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

# Experimental setting
dataset_name = 'bbbp'
gnn_types = ['GCN', 'EGCN', 'GAT']
max_epochs = 500
task = 'clf'
rand_seed = 2025
n_models = 5

# Feature setting
num_atom_feats = 58
n_fp = 0
n_radius = 3
num_mol_feats = n_fp + 188

# Excel data
dataset, _ = chem.load_dataset('data/' + dataset_name + '.xlsx', n_fp, n_radius, task)

# Data shuffling
random.seed(rand_seed)
random.shuffle(dataset)

# Duplicate remove
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

# Data splitting
num_train_mols = int(len(dataset) * 0.8)
num_valid_mols = int(len(dataset) * 0.1)
num_test_mols = int(len(dataset) * 0.1)

train_dataset = mols[:num_train_mols]
valid_dataset = mols[num_train_mols:(num_train_mols + num_valid_mols)]
test_dataset = mols[(num_train_mols + num_valid_mols):]

train_smiles = smiles[:num_train_mols]
valid_smiles = smiles[num_train_mols:(num_train_mols + num_valid_mols)]
test_smiles = smiles[(num_train_mols + num_valid_mols):]

train_targets = targets[:num_train_mols]
valid_targets = targets[num_train_mols:(num_train_mols + num_valid_mols)]
test_targets = targets[(num_train_mols + num_valid_mols):]

# DataLoader
valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Class mapping of GNN models
GNN_class_map = {
    'GCN': lambda: GCN(num_atom_feats, Best_dims_hidden, Best_dims_embedding, 2),
    'EGCN': lambda: EGCN(num_atom_feats, num_mol_feats, Best_dims_hidden, Best_dims_embedding, 2),
    'GAT': lambda: GAT(num_atom_feats, Best_dims_hidden, Best_dims_embedding, 2)
}

for gnn_idx, gnn_type in enumerate(gnn_types):
    print(f"Now running GNN type: {gnn_type}")
    models = []
    models_best = []

    # Hyperparameter setting
    params = GNN_hyperparams[gnn_type]
    Best_dims_hidden = params['hidden']
    Best_dims_embedding = params['embedding']
    Best_Weight_Decay = params['weight_decay']
    Best_Batch_Size = params['batch_size']
    Best_Max_Learning_Rate = params['max_lr']
    Best_T0 = params['T0']
    Best_Early_Stop_Limit = params['early_stop']

    # Defining the models
    for _ in range(n_models):
        models.append(GNN_class_map[gnn_type]().cuda())

    # 5 iteration of learning process
    for j in range(0, n_models):

        # HE weight initialization
        for name, child in models[j].named_children():
            if name == 'gc1' or name == 'gc2' or name == 'gc3':
                if gnn_types[gnn_idx] == 'GCN' or gnn_types[gnn_idx] == 'EGCN':
                    nn.init.kaiming_normal_(child.lin.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(child.lin_src.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(child.att_src, mode='fan_in', nonlinearity='relu')
                    nn.init.kaiming_normal_(child.att_dst, mode='fan_in', nonlinearity='relu')
            elif name == 'fc1' or name == 'fc2':
                nn.init.kaiming_normal_(child.weight, mode='fan_in', nonlinearity='relu')
  
        # Loss function, Optimizer, Learning rate schedular
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(models[j].parameters(), lr=1e-5, weight_decay=Best_Weight_Decay)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=Best_T0, T_mult=1, eta_max=Best_Max_Learning_Rate, T_up=5, gamma=0.9)
  
        # Training
        best_epoch = 0
        best_score = 0
        early_stop_cnt = 0
        early_stop_limit = Best_Early_Stop_Limit
  
        print(models[j])
        total_params = sum(p.numel() for p in models[j].parameters())
        print(f"Total number of parameters: {total_params}")

        for epoch in range(0, max_epochs):
            train_data_loader = DataLoader(train_dataset, batch_size=Best_Batch_Size, shuffle=True)

            train_loss, train_acc, train_f1, train_auc, train_p_auc = tr_gnn.train(models[j], train_data_loader, criterion, optimizer, scheduler)
            valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_gnn.test(models[j], valid_data_loader, criterion)
            test_loss, test_acc, test_f1, test_auc, test_p_auc = tr_gnn.test(models[j], test_data_loader, criterion)

            # Early stopping
            if valid_p_auc > best_score:
                best_score, best_epoch = valid_p_auc, epoch
                save_best_checkpoint(models[j], optimizer, best_epoch, best_score, j)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stop_cnt > early_stop_limit:
                break

            print('Epoch {}\ttrain loss {:.4f}\tvalid loss {:.4f}\ttest loss {:.4f}\tvalid p-auc {:.4f}\ttest p-auc {:.4f}'.format(epoch + 1, train_loss, valid_loss, test_loss, valid_p_auc, test_p_auc))
  
        models_best.append(load_best_result(models[j], j))
      
    best_ensemble_idx = 0
    best_ensemble_score = 0

    # Select best performance model during 5 iterations
    for k in range(0, n_models): 
        valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_gnn.test(models_best[k], valid_data_loader, criterion)
        if valid_p_auc > best_ensemble_score:
            best_ensemble_idx = k
            best_ensemble_score = valid_p_auc

    train_loss, train_acc, train_f1, train_auc, train_p_auc = tr_gnn.test(models_best[best_ensemble_idx], train_data_loader, criterion)
    valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_gnn.test(models_best[best_ensemble_idx], valid_data_loader, criterion)
    test_loss, test_acc, test_f1, test_auc, test_p_auc = tr_gnn.test(models_best[best_ensemble_idx], test_data_loader, criterion)

    print('Result of {}\ttrain loss {:.4f}\tvalid loss {:.4f}\ttest loss {:.4f}\t'.format(gnn_types[gnn_idx], train_loss, valid_loss, test_loss))
    print('Result of {}\ttrain acc {:.4f}\tvalid acc {:.4f}\ttest acc {:.4f}\t'.format(gnn_types[gnn_idx], train_acc, valid_acc, test_acc))
    print('Result of {}\ttrain f1 {:.4f}\tvalid f1 {:.4f}\ttest f1 {:.4f}\t'.format(gnn_types[gnn_idx], train_f1, valid_f1, test_f1))
    print('Result of {}\ttrain auc {:.4f}\tvalid auc {:.4f}\ttest auc {:.4f}\t'.format(gnn_types[gnn_idx], train_auc, valid_auc, test_auc))
    print('Result of {}\ttrain p-auc {:.4f}\tvalid p-auc {:.4f}\ttest p-auc {:.4f}\t'.format(gnn_types[gnn_idx], train_p_auc, valid_p_auc, test_p_auc))

    # Save model
    torch.save(models_best[best_ensemble_idx].state_dict(), 'preds/saved_models/GNN_seed' + str(rand_seed) + '_' + gnn_types[gnn_idx] + '.pt')



