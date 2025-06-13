import pandas
import numpy

from utils.inputs.Fingerprint import EMACCS_fingerprints
from utils.util import *

# Dataset from excel
dataset_name = 'bbbp'  # ['bbbp', 'efflux', 'influx', 'pampa']
data_pd = pandas.read_excel('data/' + dataset_name + '.xlsx')
data = numpy.array(data_pd)

# Duplicate remove
data_cleaned = data_pd.drop_duplicates(subset=[data_pd.columns[0]], keep='first')

# Extracting SMILES and target
smiles = data_cleaned.iloc[:, 0].values
targets = data_cleaned.iloc[:, 2].values

# Encoding into EMACCS fingerprint vector
emaccs_keys, valid_smiles = EMACCS_fingerprints(smiles)

# Print label distribution for all molecules
all_molecules = match_smiles_and_targets(valid_smiles, data)
count_0_all, count_1_all = count_target_values(all_molecules)
print("Count of molecules in label 0:", count_0_all)
print("Count of molecules in label 1:", count_1_all)
print("\n")

# Defining bit indices of substructure combination
combination_bit_indices = [151, 113, 117, 315, 194, 208]

# Filtering molecules that having reference substructure combination
filtered_smiles = filter_smiles_by_bits(valid_smiles, emaccs_keys, combination_bit_indices)
print(f"Total number of filtered molecules: {len(filtered_smiles)}")

filtered_molecules = match_smiles_and_targets(filtered_smiles, data)

# Filtered_molecules: {smiles: label}
label_0_smiles = []
label_1_smiles = []

for smi, label in filtered_molecules.items():
    if label == 0:
        label_0_smiles.append(smi)
    elif label == 1:
        label_1_smiles.append(smi)

# The number of filtered molecules in each label
print(f"Count of filtered molecules in label 0: {len(label_0_smiles)}")
print(f"Count of filtered molecules in label 1: {len(label_1_smiles)}")

# Print SMILES of filtered molecules in each label
print("\nSMILES with label 0:")
print(label_0_smiles)

print("\nSMILES with label 1:")
print(label_1_smiles)








