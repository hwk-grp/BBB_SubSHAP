import os
os.environ['OMP_NUM_THREADS'] = "1"

import random
import torch.nn as nn
import torch.optim as optim
import torch
from torch_geometric.loader import DataLoader

import utils.train_gnn as tr_gnn
import utils.inputs.Graph_Input as chem
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from utils.models.GNN import GCN, EGCN, GAT
from utils.checkpoint import save_best_checkpoint, load_best_result
from utils.hyperparameter import GNN_hyperparams

torch.set_num_threads(1)

# ======================================
# Experimental setting
# ======================================
dataset_name = 'bbbp'
architecture = 'gnn'
gnn_types = ['GCN', 'EGCN', 'GAT']
max_epochs = 500
task = 'clf'
rand_seed = 2025
n_models = 5

# ======================================
# Feature setting
# ======================================
num_atom_feats = 58
n_fp = 0
n_radius = 3
num_mol_feats = n_fp + 188

# ======================================
# Load dataset
# ======================================
dataset, _ = chem.load_dataset('data/' + dataset_name + '.xlsx', n_fp, n_radius, task)

# Shuffle dataset
random.seed(rand_seed)
random.shuffle(dataset)

# Remove duplicates
seen = set()
filtered_dataset = []
for entry in dataset:
    smiles = entry[0]
    if smiles not in seen:
        seen.add(smiles)
        filtered_dataset.append(entry)
dataset = filtered_dataset

smiles = [x[0] for x in dataset]
mols = [x[1] for x in dataset]
targets = [x[2] for x in dataset]

# ======================================
# Split data into train/valid/test
# ======================================
num_train_mols = int(len(dataset) * 0.8)
num_valid_mols = int(len(dataset) * 0.1)
num_test_mols = int(len(dataset) * 0.1)

train_dataset = mols[:num_train_mols]
valid_dataset = mols[num_train_mols:(num_train_mols + num_valid_mols)]
test_dataset = mols[(num_train_mols + num_valid_mols):]

train_targets = targets[:num_train_mols]
valid_targets = targets[num_train_mols:(num_train_mols + num_valid_mols)]
test_targets = targets[(num_train_mols + num_valid_mols):]

# DataLoader
valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ======================================
# Define GNN model mapping
# ======================================
GNN_class_map = {
    'GCN': lambda: GCN(num_atom_feats, Best_dims_hidden, Best_dims_embedding, 2),
    'EGCN': lambda: EGCN(num_atom_feats, num_mol_feats, Best_dims_hidden, Best_dims_embedding, 2),
    'GAT': lambda: GAT(num_atom_feats, Best_dims_hidden, Best_dims_embedding, 2)
}

# ======================================
# Main training loop
# ======================================
for gnn_idx, gnn_type in enumerate(gnn_types):
    print(f"\n=== Now running GNN type: {gnn_type} ===")
    models, models_best = [], []

    # Hyperparameter setting
    params = GNN_hyperparams[gnn_type]
    Best_dims_hidden = params['hidden']
    Best_dims_embedding = params['embedding']
    Best_Weight_Decay = params['weight_decay']
    Best_Batch_Size = params['batch_size']
    Best_Max_Learning_Rate = params['max_lr']
    Best_T0 = params['T0']
    Best_Early_Stop_Limit = params['early_stop']

    # Define models
    for _ in range(n_models):
        models.append(GNN_class_map[gnn_type]().cuda())

    # Train each model
    for j in range(n_models):

        # Initialize weights
        for name, child in models[j].named_children():
            if name in ['gc1', 'gc2', 'gc3']:
                if gnn_type in ['GCN', 'EGCN']:
                    nn.init.kaiming_normal_(child.lin.weight, mode='fan_in', nonlinearity='relu')
                else:  # GAT
                    nn.init.kaiming_normal_(child.lin_src.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(child.att_src, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(child.att_dst, mode='fan_in', nonlinearity='relu')
            elif name in ['fc1', 'fc2']:
                nn.init.kaiming_normal_(child.weight, mode='fan_in', nonlinearity='relu')

        # Loss, optimizer, scheduler
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(models[j].parameters(), lr=1e-5, weight_decay=Best_Weight_Decay)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=Best_T0, T_mult=1,
            eta_max=Best_Max_Learning_Rate, T_up=5, gamma=0.9
        )

        print(models[j])
        print(f"Total number of parameters: {sum(p.numel() for p in models[j].parameters())}")

        best_score, early_stop_cnt = 0, 0

        for epoch in range(max_epochs):
            train_data_loader = DataLoader(train_dataset, batch_size=Best_Batch_Size, shuffle=True)

            # =============================
            # Train and evaluate
            # =============================
            # train/test functions now return 8 metrics:
            # loss, acc, f1, auc, p_auc, sensitivity, specificity, fpr
            (train_loss, train_acc, train_f1, train_auc, train_p_auc,
             train_sens, train_spec, train_fpr) = tr_gnn.train(models[j], train_data_loader, criterion, optimizer, scheduler)

            (valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc,
             valid_sens, valid_spec, valid_fpr) = tr_gnn.test(models[j], valid_data_loader, criterion)

            (test_loss, test_acc, test_f1, test_auc, test_p_auc,
             test_sens, test_spec, test_fpr) = tr_gnn.test(models[j], test_data_loader, criterion)

            # =============================
            # Early stopping
            # =============================
            if valid_p_auc > best_score:
                best_score = valid_p_auc
                save_best_checkpoint(models[j], optimizer, epoch, best_score, j, architecture)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt > Best_Early_Stop_Limit:
                break

            # =============================
            # Print epoch metrics
            # =============================
            print(
                f"Epoch {epoch+1} || Train loss : {train_loss:.4f} | acc :  {train_acc:.4f} | f1 : {train_f1:.4f} | "
                f"AUC : {train_auc:.4f} | p-AUC {train_p_auc:.4f} || "
                f"Valid p-AUC : {valid_p_auc:.4f} || "
                f"Test p-AUC : {test_p_auc:.4f}"
            )

        # Load best result for this model
        models_best.append(load_best_result(models[j], j, architecture))

    # =============================
    # Select best model among n_models
    # =============================
    best_ensemble_idx = 0
    best_ensemble_score = 0
    for k in range(n_models):
        (_, _, _, _, valid_p_auc, _, _, _) = tr_gnn.test(models_best[k], valid_data_loader, criterion)
        if valid_p_auc > best_ensemble_score:
            best_ensemble_idx = k
            best_ensemble_score = valid_p_auc

    best_model = models_best[best_ensemble_idx]

    # =============================
    # Final evaluation
    # =============================
    train_metrics = tr_gnn.test(best_model, train_data_loader, criterion)
    valid_metrics = tr_gnn.test(best_model, valid_data_loader, criterion)
    test_metrics = tr_gnn.test(best_model, test_data_loader, criterion)

    print(f"\n==== [{gnn_type}] FINAL RESULTS ====")
    print(f"Train -> Loss : {train_metrics[0]:.4f}, ACC : {train_metrics[1]:.4f}, F1 : {train_metrics[2]:.4f}, "
          f"AUC : {train_metrics[3]:.4f}, p-AUC : {train_metrics[4]:.4f}, Sensitivity : {train_metrics[5]:.4f}, "
          f"Specificity : {train_metrics[6]:.4f}, FPR : {train_metrics[7]:.4f}")
    print(f"Valid -> Loss : {valid_metrics[0]:.4f}, ACC : {valid_metrics[1]:.4f}, F1 : {valid_metrics[2]:.4f}, "
          f"AUC : {valid_metrics[3]:.4f}, p-AUC : {valid_metrics[4]:.4f}, Sensitivity : {valid_metrics[5]:.4f}, "
          f"Specificity : {valid_metrics[6]:.4f}, FPR : {valid_metrics[7]:.4f}")
    print(f"Test  -> Loss : {test_metrics[0]:.4f}, ACC : {test_metrics[1]:.4f}, F1 : {test_metrics[2]:.4f}, "
          f"AUC : {test_metrics[3]:.4f}, p-AUC : {test_metrics[4]:.4f}, Sensitivity : {test_metrics[5]:.4f}, "
          f"Specificity : {test_metrics[6]:.4f}, FPR : {test_metrics[7]:.4f}")

    # Save the best model
    torch.save(best_model.state_dict(), f'preds/saved_models/GNN_seed{rand_seed}_{gnn_type}.pt')
