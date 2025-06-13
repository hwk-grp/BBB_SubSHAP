import pandas as pd
import numpy as np
import seaborn as sns
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from rdkit import Chem


# Converts a string to a float, returns 0 if conversion fails.
def convert_to_number(value):
    try:
        return float(value)
    except ValueError:
        return 0


# Applies the softmax function row-wise to a Numpy array.
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Creates a SHAP summary plot for a single feature (one substructure).
def single_feature_summary_plot(feature_index, SHAP_values, X, feature_names, value_names, dataset_name):
    feature_values = X[:, feature_index]
    shap_vals = SHAP_values[:, feature_index]
    value_mapping = {0: "Absent", 1: "Present"}

    plt.figure(figsize=(8, 3))

    data = pd.DataFrame({
        "SHAP Value": shap_vals,
        "Feature Value": feature_values
    })
    data["Feature Label"] = data["Feature Value"].replace(value_mapping)

    sns.set_style("whitegrid")
    sns.set_context("talk")

    sns.stripplot(
        x=data["SHAP Value"],
        y=data["Feature Label"],
        hue=feature_values,
        palette="coolwarm",
        size=4, jitter=True,
        orient="h", color="black", alpha=0.8,
        order=["Absent", "Present"]
    )

    legend = plt.legend(bbox_to_anchor=(0.03, 0.95), loc='upper left', borderaxespad=0., fontsize=10, title_fontsize=14)
    legend.get_title().set_fontweight("bold")

    for text in legend.get_texts():
        label = text.get_text()
        text.set_text(value_mapping[int(float(label))])
        text.set_fontweight("bold")
        text.set_fontsize(fontsize=14)

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("SHAP value", fontsize=16, fontweight="bold")
    plt.ylabel("")
    plt.xticks(fontsize=12)
    plt.tick_params(axis='x', pad=2)
    plt.yticks(fontsize=14, fontweight="bold")
    plt.xlim(-0.4, 0.4)
    plt.grid(alpha=0.8)
    plt.tight_layout()
    plt.show()


# Draws and saves the structure of a functional group from a SMARTS pattern.
def draw_smarts(smarts: str, filename: str = "functional_group.png"):
    try:
        mol = Chem.MolFromSmarts(smarts)

        if mol is None:
            raise ValueError("The given SMARTS pattern is not valid.")

        Draw.MolToFile(mol, filename)
        print(f"The substructure image has been saved as {filename}.")

    except Exception as e:
        print(f"Error type: {e}")


# Filters a list of SMILES based on the presence of specified MACCS key bits.
def filter_smiles_by_bits(smiles_list, maccs_list, bit_list):
    filtered_smiles = []
    for smiles, maccs in zip(smiles_list, maccs_list):
        if all(maccs.GetBit(bit) for bit in bit_list):
            filtered_smiles.append(smiles)
    return filtered_smiles


# Matches a list of SMILES with corresponding target values from an array.
def match_smiles_and_targets(smiles_list, smiles_targets_array):
    matched_molecules = {}

    smiles_array = smiles_targets_array[:, 0]
    targets_array = smiles_targets_array[:, 2]

    for smiles in smiles_list:
        if smiles in smiles_array:
            index = np.where(smiles_array == smiles)[0][0]
            matched_molecules[smiles] = targets_array[index]

    return matched_molecules  # Return dictionary {SMILES : Targets}


# Counts the number of samples for each class (0 and 1) in the matched targets.
def count_target_values(matched_targets):
    values = list(matched_targets.values())
    count_0 = values.count(0)
    count_1 = values.count(1)
    return count_0, count_1
