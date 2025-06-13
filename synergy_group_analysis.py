import pandas
import torch
import numpy
import random
from pathlib import Path

import utils.inputs.Fingerprint as fp
import utils.inputs.Graph_Input as chem
from utils.inputs import EMACCSkeys
from utils.util import *

# Experimental setting
dataset_name = 'bbbp'  # ['bbbp', 'efflux', 'influx', 'pampa']
feature_type = 'emaccs_FP'
task = 'clf'
seed_dict = {'bbbp': 2030, 'efflux': 2028, 'influx': 2026, 'pampa': 2027}
rand_seed = seed_dict[dataset_name]

# Synergy group analysis (Setting index of interesting substructure)
interest_substructure_id = 151

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

# The tensor that used to extract SHAP value
explain_tensor = torch.tensor(train_dataset, dtype=torch.float32)

# Recall shap values
SHAP_values = numpy.array(pandas.read_excel("preds/shap_values/shap_values" + "_" + dataset_name + "_" + feature_type + ".xlsx", index_col=0))

# The range of SHAP value
print(f"Shape of SHAP values : {SHAP_values.shape}")
max_value = numpy.max(SHAP_values)
min_value = numpy.min(SHAP_values)
print(f"Maximum SHAP value: {max_value}")
print(f"Minimum SHAP value: {min_value}")
print("\n")

# Create a folder to save synergy group results
folder_path = Path("preds/draws/" + dataset_name + "/" + feature_type + "/synergy_group_analysis/" + "Bit." + str(interest_substructure_id) + '/')
folder_path.mkdir(parents=True, exist_ok=True)  
print(f"Folder is generated - path: {folder_path}")

# Draw interesting substructure
interest_smart = EMACCSkeys.smartsPatts[interest_substructure_id][0]
draw_smarts(interest_smart, 'preds/draws/' + dataset_name + "/" + feature_type + '/synergy_group_analysis/' + 'Bit.' + str(interest_substructure_id) + '/' + 'interesting_substructure_bit.' + str(interest_substructure_id) + '.png')

# Extracting subset if interesting substructure is presented
present = train_dataset[:, interest_substructure_id] == 1
indices = numpy.where(present)[0]
SHAP_value_interest_substructure = SHAP_values[indices, interest_substructure_id]

# Calculate the mean and standard deviation
mean = numpy.mean(SHAP_value_interest_substructure)
std = numpy.std(SHAP_value_interest_substructure)
print("\n")
print(f"Mean value of SHAP: {mean}")
print(f"Standard deviation value of SHAP: {std}")
print("\n")

# Determine contribution direction of interesting substructure (0 or 1)
contribution = 1 if mean > 0 else 0
print(f"Interesting substructure Bit {interest_substructure_id} contributed to (label : {contribution})")
print("\n")

# Show contribution of interesting substructure
feature_names = [f"Bit_{i}" for i in range(explain_tensor.shape[1])]
value_names = ["Absent", "Present"]

single_feature_summary_plot(
    feature_index=interest_substructure_id,
    SHAP_values=SHAP_values,
    X=explain_tensor,
    feature_names=feature_names,
    value_names=value_names,
    dataset_name=dataset_name
)

# Positive and Negetive synergy group analysis
if contribution == 1:
    upper_bound = mean + std  # Positive set
    above_upper_bound = SHAP_value_interest_substructure[SHAP_value_interest_substructure >= upper_bound]
elif contribution == 0:
    upper_bound = mean - std  # Positive set
    above_upper_bound = SHAP_value_interest_substructure[SHAP_value_interest_substructure <= upper_bound]

num_upper_bound = len(above_upper_bound)

if contribution == 1:
    below_lower_bound = numpy.sort(SHAP_value_interest_substructure)[:num_upper_bound]  # Negative set
elif contribution == 0:
    below_lower_bound = numpy.sort(SHAP_value_interest_substructure)[-num_upper_bound:]  # Negative set

maccs_lb = numpy.zeros(len(train_dataset[0]))
maccs_ub = numpy.zeros(len(train_dataset[0]))

for i in range(0, len(indices)):
    for j in range(0, len(below_lower_bound)):
        if SHAP_values[indices[i], interest_substructure_id] == below_lower_bound[j]:
            maccs_lb = maccs_lb + train_dataset[indices[i]]
for i in range(0, len(indices)):
    for j in range(0, len(above_upper_bound)):
        if SHAP_values[indices[i], interest_substructure_id] == above_upper_bound[j]:
            maccs_ub = maccs_ub + train_dataset[indices[i]]
mean_maccs_ub = maccs_ub/len(above_upper_bound)
mean_maccs_lb = maccs_lb/len(below_lower_bound)

synergistic_effect = numpy.nan_to_num((mean_maccs_ub - mean_maccs_lb))

# The top 20 and bottom 20 entries were extracted based on values
positive_synergy_group = numpy.argsort(synergistic_effect)[-20:][::-1]
negative_synergy_group = numpy.argsort(synergistic_effect)[:20]

# Drawing structure of synergy groups
for i in range(0, len(positive_synergy_group)):
    key = positive_synergy_group[i]
    smart = EMACCSkeys.smartsPatts[key][0]
    diff = synergistic_effect[key]
    diff = int(diff * 100)
    draw_smarts(smart, 'preds/draws/' + dataset_name + "/" + feature_type + '/synergy_group_analysis/' + 'Bit.' + str(interest_substructure_id) + '/' + 'positive_synergy_group_top' + str(i) + '_diff(' + str(diff) + ')' + '_bit.' + str(key) + '.png')

for i in range(0, len(negative_synergy_group)):
    key = negative_synergy_group[i]
    smart = EMACCSkeys.smartsPatts[key][0]
    diff = synergistic_effect[key]
    diff = int(diff * 100)
    draw_smarts(smart, 'preds/draws/' + dataset_name + "/" + feature_type + '/synergy_group_analysis/' + 'Bit.' + str(interest_substructure_id) + '/' + 'negative_synergy_group_top' + str(i) + '_diff(' + str(diff) + ')' + '_bit.' + str(key) + '.png')
