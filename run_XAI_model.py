import random
import torch.optim as optim
import torch
import numpy
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import utils.train_XAI as tr_dnn
import utils.inputs.Graph_Input as chem
import utils.inputs.Fingerprint as fp
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts
from utils.models.XAI_model import DNN
from utils.checkpoint import save_best_checkpoint, load_best_result

# Experimental setting
dataset_name = 'bbbp'  # ['bbbp', 'efflux', 'influx', 'pampa']
feature_type = 'emaccs_FP'
task = 'clf'
seed_dict = {'bbbp': 2030, 'efflux': 2028, 'influx': 2026, 'pampa': 2027}
rand_seed = seed_dict[dataset_name]
max_epochs = 500

# Feature setting
n_fp = 0
n_radius = 3
num_atom_feats = 63
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

train_smiles = smiles[:num_train_mols]
valid_smiles = smiles[num_train_mols:(num_train_mols + num_valid_mols)]
test_smiles = smiles[(num_train_mols + num_valid_mols):]

train_targets = targets[:num_train_mols]
valid_targets = targets[num_train_mols:(num_train_mols + num_valid_mols)]
test_targets = targets[(num_train_mols + num_valid_mols):]

train_targets = numpy.array(train_targets).reshape(-1, 1)
valid_targets = numpy.array(valid_targets).reshape(-1, 1)
test_targets = numpy.array(test_targets).reshape(-1, 1)

# Molucular fingerprint encoding
Train_emaccs_FP, _ = fp.EMACCS_fingerprints(train_smiles)
Valid_emaccs_FP, _ = fp.EMACCS_fingerprints(valid_smiles)
Test_emaccs_FP, _ = fp.EMACCS_fingerprints(test_smiles)

train_dataset = numpy.array(Train_emaccs_FP)
valid_dataset = numpy.array(Valid_emaccs_FP)
test_dataset = numpy.array(Test_emaccs_FP)

train_dataset = numpy.hstack([train_dataset, train_targets])
train_dataset = numpy.array(train_dataset, dtype=numpy.float32)
valid_dataset = numpy.hstack([valid_dataset, valid_targets])
valid_dataset = numpy.array(valid_dataset, dtype=numpy.float32)
test_dataset = numpy.hstack([test_dataset, test_targets])
test_dataset = numpy.array(test_dataset, dtype=numpy.float32)

# DataLoader
valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Defining a model
model = DNN(len(train_dataset[0])-1, n_hidden=512, dropout=0.4).cuda()
print(model)

# Loss function, Optimizer, Learning rate schedular
criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=1e-10, weight_decay=1e-7)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=25, T_mult=1, eta_max=0.00075, T_up=5, gamma=0.9)

# Training
best_epoch = 0
best_score = 0
early_stop_cnt = 0
early_stop_limit = 25

for epoch in range(0, max_epochs):
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_loss, train_acc, train_f1, train_auc, train_p_auc = tr_dnn.train(model, train_data_loader, criterion, optimizer, scheduler)
    valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_dnn.test(model, valid_data_loader, criterion)
    test_loss, test_acc, test_f1, test_auc, test_p_auc = tr_dnn.test(model, test_data_loader, criterion)

    # Early stopping
    if valid_p_auc > best_score:
        best_score, best_epoch = valid_p_auc, epoch
        save_best_checkpoint(model, optimizer, best_epoch, best_score, rand_seed)
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1

    if early_stop_cnt > early_stop_limit:
        break

    print('Epoch {}\ttrain loss {:.4f}\tvalid loss {:.4f}\ttest loss {:.4f}\tvalid p-auc {:.4f}\ttest p-auc {:.4f}'
          .format(epoch + 1, train_loss, valid_loss, test_loss, valid_p_auc, test_p_auc))

model_best = load_best_result(model, rand_seed)
train_loss, train_acc, train_f1, train_auc, train_p_auc = tr_dnn.test(model_best, train_data_loader, criterion)
valid_loss, valid_acc, valid_f1, valid_auc, valid_p_auc = tr_dnn.test(model_best, valid_data_loader, criterion)
test_loss, test_acc, test_f1, test_auc, test_p_auc = tr_dnn.test(model_best, test_data_loader, criterion)

print('Result of {}\ttrain loss {:.4f}\tvalid loss {:.4f}\ttest loss {:.4f}\t'.format(feature_type, train_loss, valid_loss, test_loss))
print('Result of {}\ttrain acc {:.4f}\tvalid acc {:.4f}\ttest acc {:.4f}\t'.format(feature_type, train_acc, valid_acc, test_acc))
print('Result of {}\ttrain f1 {:.4f}\tvalid f1 {:.4f}\ttest f1 {:.4f}\t'.format(feature_type, train_f1, valid_f1, test_f1))
print('Result of {}\ttrain auc {:.4f}\tvalid auc {:.4f}\ttest auc {:.4f}\t'.format(feature_type, train_auc, valid_auc, test_auc))
print('Result of {}\ttrain p-auc {:.4f}\tvalid p-auc {:.4f}\ttest p-auc {:.4f}\t'.format(feature_type, train_p_auc, valid_p_auc, test_p_auc))

# Save model
torch.save(model.state_dict(), 'preds/saved_models/XAI_model_' + dataset_name + '_seed_' + str(rand_seed) + '_' + feature_type + '.pt')


