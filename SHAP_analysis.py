import pandas
import torch
import numpy
import random
import shap
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

import utils.inputs.Fingerprint as fp
import utils.inputs.Graph_Input as chem
from utils.models.XAI_model import DNN

# Experimental setting
dataset_name = 'bbbp'  # ['bbbp', 'efflux', 'influx', 'pampa']
feature_type = 'emaccs_FP'
task = 'clf'
seed_dict = {'bbbp': 2030, 'efflux': 2028, 'influx': 2026, 'pampa': 2027}
rand_seed = seed_dict[dataset_name]

# Feature setting
n_fp = 0
n_radius = 3
num_atom_feats = 58
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

# Molucular fingerprints encoding
Train_emaccs_FP, _ = fp.EMACCS_fingerprints(train_smiles)
Valid_emaccs_FP, _ = fp.EMACCS_fingerprints(valid_smiles)
Test_emaccs_FP, _ = fp.EMACCS_fingerprints(test_smiles)

train_dataset = numpy.array(Train_emaccs_FP)
valid_dataset = numpy.array(Valid_emaccs_FP)
test_dataset = numpy.array(Test_emaccs_FP)

train_dataset_load = numpy.hstack([train_dataset, train_targets])
train_dataset_load = numpy.array(train_dataset_load, dtype=numpy.float32)
valid_dataset_load = numpy.hstack([valid_dataset, valid_targets])
valid_dataset_load = numpy.array(valid_dataset_load, dtype=numpy.float32)
test_dataset_load = numpy.hstack([test_dataset, test_targets])
test_dataset_load = numpy.array(test_dataset_load, dtype=numpy.float32)

# Dataloader
train_data_loader = DataLoader(train_dataset_load, batch_size=32, shuffle=False)
valid_data_loader = DataLoader(valid_dataset_load, batch_size=32, shuffle=False)
test_data_loader = DataLoader(test_dataset_load, batch_size=32, shuffle=False)

# Recall saved model
model = DNN(len(train_dataset[0]), n_hidden=512, dropout=0.4)
model.load_state_dict(torch.load('preds/saved_models/XAI_model_inpaper_' + dataset_name + '_seed_' + str(rand_seed) + '_' + feature_type + '.pt'))
model.eval()

# The tensor that used to extract SHAP value
explain_tensor = torch.tensor(train_dataset, dtype=torch.float32)
print(f"Shape of dataset used in explanation: {explain_tensor.shape}")

# Extracting SHAP value array
explainer = shap.GradientExplainer(model, explain_tensor)
SHAP_values = explainer.shap_values(explain_tensor, rseed=rand_seed)
SHAP_values = SHAP_values.squeeze(axis=-1)

# Saving SHAP value array
df = pandas.DataFrame(SHAP_values)
df.to_excel("preds/shap_values/shap_values" + "_" + dataset_name + "_" + feature_type + ".xlsx")

# The range of SHAP value
print(f"Shape of shap value: {SHAP_values.shape}")
max_value = numpy.max(SHAP_values)  
min_value = numpy.min(SHAP_values)  
print(f"Maximum shap value: {max_value}")
print(f"Minimum shap value: {min_value}")

# Show shap analysis results
shap.initjs()
f = plt.figure()
shap.summary_plot(SHAP_values, explain_tensor.numpy(), max_display=30, feature_names=[f"bit_{i}" for i in range(train_dataset.shape[1])])
f.savefig("preds/draws/" + dataset_name + "/" + feature_type + "/shap_analysis/shap_summary_plot" + ".png", bbox_inches="tight")

f = plt.figure()
shap.summary_plot(SHAP_values, explain_tensor.numpy(), max_display=30, feature_names=[f"bit_{i}" for i in range(train_dataset.shape[1])], plot_type='bar')
f.savefig("preds/draws/" + dataset_name + "/" + feature_type + "/shap_analysis/shap_summary_bar_plot" + ".png", bbox_inches="tight")

# Find important substructure contribute to prediction
absolute_shap = []  # list to store the sum of the absolute shap values
contribution = []  # Which label each substructure contributes to

for col in range(train_dataset.shape[1]):
    indices = numpy.where(train_dataset[:, col] == 1)[0]  # Only using present substructure
    SHAP_value = SHAP_values[indices, col]
    mean_value = numpy.mean(SHAP_value)
    contribution.append(1 if mean_value >= 0 else 0)  # Determine contribution direction
    abs_sum = numpy.sum(numpy.abs(SHAP_value))
    absolute_shap.append(abs_sum)

sorted_indices = numpy.argsort(absolute_shap)[::-1]
sorted_shap_values = numpy.array(absolute_shap)[sorted_indices]
sorted_contributions = numpy.array(contribution)[sorted_indices]

for idx, shap_value, contrib in zip(sorted_indices, sorted_shap_values, sorted_contributions):
    print(f"Bit index: {idx}, SHAP Sum: {shap_value:.4f}, Contribution: {contrib}")

# 20 Highest contibuting substructures for label : 1
top_20_indices_contrib_1 = sorted_indices[sorted_contributions == 1][:20]
# 20 Highest contibuting substructures for label : 0
top_20_indices_contrib_0 = sorted_indices[sorted_contributions == 0][:20]

print("Top 20 substructure indices contribution to label 1:", top_20_indices_contrib_1.tolist())
print("Top 20 substructure indices contribution to label 0:", top_20_indices_contrib_0.tolist())

# Retrieve SHAP Sum values corresponding to the top 20 indices
shap_values_contrib_1 = sorted_shap_values[sorted_contributions == 1][:20]
shap_values_contrib_0 = sorted_shap_values[sorted_contributions == 0][:20]