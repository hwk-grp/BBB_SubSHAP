import os
os.environ['OMP_NUM_THREADS'] = "1"

import random
import torch.nn as nn
import torch.optim as optim
import torch
import numpy
from torch.utils.data import TensorDataset, DataLoader

import utils.train_dnn as tr_dnn
import utils.inputs.Graph_Input as chem
import utils.inputs.Fingerprint as fp
import utils.inputs.Molecular_descriptor as md
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from utils.models.DNN import DNN
from utils.Imblearn import return_emb_after_Imblearn
from utils.checkpoint import save_best_checkpoint, load_best_result
from utils.hyperparameter import DNN_hyperparams
from utils.util import *

torch.set_num_threads(1)

# ======================================
# Experimental set up
# ======================================
dataset_name = 'bbbp'
architecture = 'dnn'
feature_types = ['maccs_FP', 'morgan_FP', 'pubchem_FP', 'rdkit_MD', 'mordred_MD']
max_epochs = 500
task = 'clf'
rand_seed = 2025
n_models = 5

# ======================================
# Feature set up
# ======================================
n_fp = 0
n_radius = 3
num_atom_feats = 58
num_mol_feats = n_fp + 188

# ======================================
# Imbalance algorithm types
# ======================================
Imblearn_kind = [
    'Original', 'SMOTE', 'RandomOversampling', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSMOTE',
    'RandomUndersampler', 'TomekLinks', 'ENN', 'SMOTETomek', 'SMOTEENN'
]

# ======================================
# Load dataset
# ======================================
dataset, _ = chem.load_dataset('data/' + dataset_name + '.xlsx', n_fp, n_radius, task)

# Shuffle and remove duplicates
random.seed(rand_seed)
random.shuffle(dataset)

seen = set()
filtered_dataset = []
for entry in dataset:
    smiles = entry[0]
    if smiles not in seen:
        seen.add(smiles)
        filtered_dataset.append(entry)
dataset = filtered_dataset

smiles = [x[0] for x in dataset]
targets = [x[2] for x in dataset]

# Split data into train/valid/test
num_train_mols = int(len(dataset) * 0.8)
num_valid_mols = int(len(dataset) * 0.1)
num_test_mols = int(len(dataset) * 0.1)

train_smiles = smiles[:num_train_mols]
valid_smiles = smiles[num_train_mols:(num_train_mols + num_valid_mols)]
test_smiles = smiles[(num_train_mols + num_valid_mols):]

train_targets = targets[:num_train_mols]
valid_targets = numpy.array(targets[num_train_mols:(num_train_mols + num_valid_mols)]).reshape(-1, 1)
test_targets = numpy.array(targets[(num_train_mols + num_valid_mols):]).reshape(-1, 1)

# ======================================
# Molecular feature encoding
# ======================================
Train_maccs_FP, Train_morgan_FP, Train_pubchem_FP = fp.generate_fingerprints(train_smiles)
Valid_maccs_FP, Valid_morgan_FP, Valid_pubchem_FP = fp.generate_fingerprints(valid_smiles)
Test_maccs_FP, Test_morgan_FP, Test_pubchem_FP = fp.generate_fingerprints(test_smiles)

Train_rdkit_MD = md.RDkit_descriptors(train_smiles)
Valid_rdkit_MD = md.RDkit_descriptors(valid_smiles)
Test_rdkit_MD = md.RDkit_descriptors(test_smiles)

Train_rdkit_MD = numpy.nan_to_num(numpy.array([[convert_to_number(x) for x in row] for row in Train_rdkit_MD]), nan=0.0)
Valid_rdkit_MD = numpy.nan_to_num(numpy.array([[convert_to_number(x) for x in row] for row in Valid_rdkit_MD]), nan=0.0)
Test_rdkit_MD = numpy.nan_to_num(numpy.array([[convert_to_number(x) for x in row] for row in Test_rdkit_MD]), nan=0.0)

Train_mordred_MD = md.Mordred_descriptors(train_smiles)
Valid_mordred_MD = md.Mordred_descriptors(valid_smiles)
Test_mordred_MD = md.Mordred_descriptors(test_smiles)

Train_mordred_MD = numpy.nan_to_num(numpy.array([[convert_to_number(x) for x in row] for row in Train_mordred_MD]), nan=0.0)
Valid_mordred_MD = numpy.nan_to_num(numpy.array([[convert_to_number(x) for x in row] for row in Valid_mordred_MD]), nan=0.0)
Test_mordred_MD = numpy.nan_to_num(numpy.array([[convert_to_number(x) for x in row] for row in Test_mordred_MD]), nan=0.0)

feature_dict = {
    'maccs_FP': (Train_maccs_FP, Valid_maccs_FP, Test_maccs_FP),
    'morgan_FP': (Train_morgan_FP, Valid_morgan_FP, Test_morgan_FP),
    'pubchem_FP': (Train_pubchem_FP, Valid_pubchem_FP, Test_pubchem_FP),
    'rdkit_MD': (Train_rdkit_MD, Valid_rdkit_MD, Test_rdkit_MD),
    'mordred_MD': (Train_mordred_MD, Valid_mordred_MD, Test_mordred_MD)
}

# ======================================
# Training process
# ======================================
for feature_idx, feature in enumerate(feature_types):
    print(f"Feature index: {feature_idx}, Feature name: {feature}")
    params = DNN_hyperparams[feature]

    Best_dims_hidden = params['dims_hidden']
    Best_dropout = params['dropout']
    Best_Weight_Decay = params['weight_decay']
    Best_Batch_Size = params['batch_size']
    Best_Max_Learning_Rate = params['max_lr']
    Best_T0 = params['T0']
    Best_Early_Stop_Limit = params['early_stop']

    train_dataset, valid_dataset, test_dataset = feature_dict[feature]
    Imblearn_embeddings = return_emb_after_Imblearn(train_dataset, train_targets, rand_seed, feature_types[feature_idx])

    valid_dataset = numpy.hstack([valid_dataset, valid_targets])
    test_dataset = numpy.hstack([test_dataset, test_targets])

    for imb_idx in range(len(Imblearn_kind)):
        models, models_best = [], []

        train_data = numpy.array(Imblearn_embeddings[imb_idx])
        valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for _ in range(n_models):
            models.append(DNN(len(train_dataset[0]), Best_dims_hidden, Best_dropout).cuda())

        for j in range(n_models):
            # Initialize weights
            for name, child in models[j].named_children():
                if hasattr(child, 'weight') and child.weight.dim() >= 2:
                    nn.init.kaiming_normal_(child.weight, mode='fan_in', nonlinearity='relu')

            print(models[j])
            print(f"Total parameters: {sum(p.numel() for p in models[j].parameters())}")

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(models[j].parameters(), lr=1e-10, weight_decay=Best_Weight_Decay)
            scheduler = CosineAnnealingWarmUpRestarts(
                optimizer, T_0=Best_T0, T_mult=1,
                eta_max=Best_Max_Learning_Rate, T_up=5, gamma=0.9
            )

            best_score, early_stop_cnt = 0, 0

            for epoch in range(max_epochs):
                train_data_loader = DataLoader(train_data, batch_size=Best_Batch_Size, shuffle=True)

                # train_dnn now returns 8 metrics including sensitivity/specificity/fpr
                (train_loss, train_acc, train_f1, train_auc, train_p_auc,
                 train_sens, train_spec, train_fpr) = tr_dnn.train(models[j], train_data_loader, criterion, optimizer, scheduler)

                (valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc,
                 valid_sens, valid_spec, valid_fpr) = tr_dnn.test(models[j], valid_data_loader, criterion)

                (test_loss, test_acc, test_f1, test_auc, test_p_auc,
                 test_sens, test_spec, test_fpr) = tr_dnn.test(models[j], test_data_loader, criterion)

                if valid_p_auc > best_score:
                    best_score = valid_p_auc
                    save_best_checkpoint(models[j], optimizer, epoch, best_score, j, architecture)
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1

                if early_stop_cnt > Best_Early_Stop_Limit:
                    break

                print(f'Epoch : {epoch+1} | Train loss : {train_loss:.4f} | Valid p-AUC : {valid_p_auc:.4f} | Test p-AUC : {test_p_auc:.4f}')

            models_best.append(load_best_result(models[j], j, architecture))

        # Select best performing model
        best_ensemble_idx = 0
        best_ensemble_score = 0
        for k in range(n_models):
            (_, _, _, _, valid_p_auc, _, _, _) = tr_dnn.test(models_best[k], valid_data_loader, criterion)
            if valid_p_auc > best_ensemble_score:
                best_ensemble_idx = k
                best_ensemble_score = valid_p_auc

        best_model = models_best[best_ensemble_idx]

        # Final evaluation
        (train_loss, train_acc, train_f1, train_auc, train_p_auc,
         train_sens, train_spec, train_fpr) = tr_dnn.test(best_model, train_data_loader, criterion)
        (valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc,
         valid_sens, valid_spec, valid_fpr) = tr_dnn.test(best_model, valid_data_loader, criterion)
        (test_loss, test_acc, test_f1, test_auc, test_p_auc,
         test_sens, test_spec, test_fpr) = tr_dnn.test(best_model, test_data_loader, criterion)

        # Print all metrics
        print(f'\n==== [{feature}.{Imblearn_kind[imb_idx]}] FINAL RESULTS ====')
        print(f'Train -> Loss: {train_loss:.4f}, ACC: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, p-AUC: {train_p_auc:.4f}, Sensitivity: {train_sens:.4f}, Specificity: {train_spec:.4f}, FPR: {train_fpr:.4f}')
        print(f'Valid -> Loss: {valid_loss:.4f}, ACC: {valid_acc:.4f}, F1: {valid_f1:.4f}, AUC: {valid_auc:.4f}, p-AUC: {valid_p_auc:.4f}, Sensitivity: {valid_sens:.4f}, Specificity: {valid_spec:.4f}, FPR: {valid_fpr:.4f}')
        print(f'Test  -> Loss: {test_loss:.4f}, ACC: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, p-AUC: {test_p_auc:.4f}, Sensitivity: {test_sens:.4f}, Specificity: {test_spec:.4f}, FPR: {test_fpr:.4f}')

        # Save best model
        torch.save(best_model.state_dict(), f'preds/saved_models/DNN_seed{rand_seed}_{feature}_{Imblearn_kind[imb_idx]}.pt')
