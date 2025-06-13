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
from utils.checkpoint import save_best_checkpoint, load_checkpoint, load_best_result
from utils.hyperparameter import DNN_hyperparams
from utils.util import *

torch.set_num_threads(1)


# Experimental setting
dataset_name = 'bbbp'
feature_types = ['maccs_FP', 'morgan_FP', 'pubchem_FP', 'rdkit_MD', 'mordred_MD']
max_epochs = 500
task = 'clf'
rand_seed = 2025
n_models = 5

# Feature setting
n_fp = 0
n_radius = 3
num_atom_feats = 58
num_mol_feats = n_fp + 188

# Imbalance algorithm variety
Imblearn_kind = ['Original', 'SMOTE', 'RandomOversampling', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', 'KMeansSMOTE',
                 'RandomUndersampler', 'TomekLinks', 'ENN',
                 'SMOTETomek', 'SMOTEENN'
                 ]
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

train_smiles = smiles[:num_train_mols]
valid_smiles = smiles[num_train_mols:(num_train_mols + num_valid_mols)]
test_smiles = smiles[(num_train_mols + num_valid_mols):]

train_targets = targets[:num_train_mols]
valid_targets = targets[num_train_mols:(num_train_mols + num_valid_mols)]
test_targets = targets[(num_train_mols + num_valid_mols):]

valid_targets = numpy.array(valid_targets).reshape(-1, 1)
test_targets = numpy.array(test_targets).reshape(-1, 1)

# Molecular Descriptors, Fingerprints encoding
Train_maccs_FP, Train_morgan_FP, Train_pubchem_FP = fp.generate_fingerprints(train_smiles)
Valid_maccs_FP, Valid_morgan_FP, Valid_pubchem_FP = fp.generate_fingerprints(valid_smiles)
Test_maccs_FP, Test_morgan_FP, Test_pubchem_FP = fp.generate_fingerprints(test_smiles)

Train_rdkit_MD = md.RDkit_descriptors(train_smiles)
Valid_rdkit_MD = md.RDkit_descriptors(valid_smiles)
Test_rdkit_MD = md.RDkit_descriptors(test_smiles)

Train_rdkit_MD = numpy.array([[convert_to_number(x) for x in row] for row in Train_rdkit_MD])
Train_rdkit_MD = numpy.nan_to_num(Train_rdkit_MD, nan=0.0)
Valid_rdkit_MD = numpy.array([[convert_to_number(x) for x in row] for row in Valid_rdkit_MD])
Valid_rdkit_MD = numpy.nan_to_num(Valid_rdkit_MD, nan=0.0)
Test_rdkit_MD = numpy.array([[convert_to_number(x) for x in row] for row in Test_rdkit_MD])
Test_rdkit_MD = numpy.nan_to_num(Test_rdkit_MD, nan=0.0)

Train_mordred_MD = md.Mordred_descriptors(train_smiles)
Valid_mordred_MD = md.Mordred_descriptors(valid_smiles)
Test_mordred_MD = md.Mordred_descriptors(test_smiles)

Train_mordred_MD = numpy.array([[convert_to_number(x) for x in row] for row in Train_mordred_MD])
Train_mordred_MD = numpy.nan_to_num(Train_mordred_MD, nan=0.0)
Valid_mordred_MD = numpy.array([[convert_to_number(x) for x in row] for row in Valid_mordred_MD])
Valid_mordred_MD = numpy.nan_to_num(Valid_mordred_MD, nan=0.0)
Test_mordred_MD = numpy.array([[convert_to_number(x) for x in row] for row in Test_mordred_MD])
Test_mordred_MD = numpy.nan_to_num(Test_mordred_MD, nan=0.0)

# Molecular feature dictionary
feature_dict = {
    'maccs_FP': (Train_maccs_FP, Valid_maccs_FP, Test_maccs_FP),
    'morgan_FP': (Train_morgan_FP, Valid_morgan_FP, Test_morgan_FP),
    'pubchem_FP': (Train_pubchem_FP, Valid_pubchem_FP, Test_pubchem_FP),
    'rdkit_MD': (Train_rdkit_MD, Valid_rdkit_MD, Test_rdkit_MD),
    'mordred_MD': (Train_mordred_MD, Valid_mordred_MD, Test_mordred_MD)
}

# Hyperparameter setting
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

    # Data resampling (Imbalance algorithm)
    Imblearn_embeddings = return_emb_after_Imblearn(train_dataset, train_targets, rand_seed, feature_types[feature_idx])

    valid_dataset = numpy.hstack([valid_dataset, valid_targets])
    valid_dataset = numpy.array(valid_dataset, dtype=numpy.float32)
    test_dataset = numpy.hstack([test_dataset, test_targets])
    test_dataset = numpy.array(test_dataset, dtype=numpy.float32)


    for imb_idx in range(0, len(Imblearn_kind)):
        
        models = list()
        models_best = list()
        
        # DataLoader
        train_data = numpy.array(Imblearn_embeddings[imb_idx])
        valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Defining a model
        for i in range(0, n_models):
            models.append(DNN(len(train_dataset[0]), Best_dims_hidden, Best_dropout).cuda())

        # 5 iteration of learning process
        for j in range(0, n_models):
            # HE weight initialization
            for name, child in models[j].named_children():
                if hasattr(child, 'weight') and child.weight.dim() >= 2:
                    nn.init.kaiming_normal_(child.weight, mode='fan_in', nonlinearity='relu')

            print(models[j])
            total_params = sum(p.numel() for p in models[j].parameters())
            print(f"Total number of parameters: {total_params}")

            # Loss function, Optimizer, Learning rate schedular
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(models[j].parameters(), lr=1e-10, weight_decay=Best_Weight_Decay)
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=Best_T0, T_mult=1, eta_max=Best_Max_Learning_Rate, T_up=5, gamma=0.9)

            # Training
            best_epoch = 0
            best_score = 0
            early_stop_cnt = 0
            early_stop_limit = Best_Early_Stop_Limit

            for epoch in range(0, max_epochs):
                train_data_loader = DataLoader(train_data, batch_size=Best_Batch_Size, shuffle=True)
                train_loss, train_acc, train_f1, train_auc, train_p_auc = tr_dnn.train(models[j], train_data_loader, criterion, optimizer, scheduler)
                valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_dnn.test(models[j], valid_data_loader, criterion)
                test_loss, test_acc, test_f1, test_auc, test_p_auc = tr_dnn.test(models[j], test_data_loader, criterion)

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
            valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_dnn.test(models_best[k], valid_data_loader, criterion)
            if valid_p_auc > best_ensemble_score:
                best_ensemble_idx = k
                best_ensemble_score = valid_p_auc
        
        train_loss, train_acc, train_f1, train_auc, train_p_auc = tr_dnn.test(models_best[best_ensemble_idx], train_data_loader, criterion)
        valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_dnn.test(models_best[best_ensemble_idx], valid_data_loader, criterion)
        test_loss, test_acc, test_f1, test_auc, test_p_auc = tr_dnn.test(models_best[best_ensemble_idx], test_data_loader, criterion)
        
        print('Result of {}.{}\ttrain loss {:.4f}\tvalid loss {:.4f}\ttest loss {:.4f}\t'.format(feature_types[feature_idx], Imblearn_kind[imb_idx], train_loss, valid_loss, test_loss))
        print('Result of {}.{}\ttrain acc {:.4f}\tvalid acc {:.4f}\ttest acc {:.4f}\t'.format(feature_types[feature_idx], Imblearn_kind[imb_idx], train_acc, valid_acc, test_acc))
        print('Result of {}.{}\ttrain f1 {:.4f}\tvalid f1 {:.4f}\ttest f1 {:.4f}\t'.format(feature_types[feature_idx], Imblearn_kind[imb_idx], train_f1, valid_f1, test_f1))
        print('Result of {}.{}\ttrain auc {:.4f}\tvalid auc {:.4f}\ttest auc {:.4f}\t'.format(feature_types[feature_idx], Imblearn_kind[imb_idx], train_auc, valid_auc, test_auc))
        print('Result of {}.{}\ttrain p-auc {:.4f}\tvalid p-auc {:.4f}\ttest p-auc {:.4f}\t'.format(feature_types[feature_idx], Imblearn_kind[imb_idx], train_p_auc, valid_p_auc, test_p_auc))

        # Save model
        torch.save(models_best[best_ensemble_idx].state_dict(), 'preds/saved_models/DNN_seed' + str(rand_seed) + '_' + feature_types[feature_idx] + '_' + Imblearn_kind[imb_idx] + '.pt')

