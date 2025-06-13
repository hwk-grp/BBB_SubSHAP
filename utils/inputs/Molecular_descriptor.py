from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import pandas as pd
from mordred import Calculator, descriptors


def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    rdkit_descriptors_list = []
    for mol in mols:
        # add hydrogens to molecules
        mol = Chem.AddHs(mol)
        rdkit_descriptors = calc.CalcDescriptors(mol)
        rdkit_descriptors_list.append(rdkit_descriptors)


    df_rdkit_descriptors = pd.DataFrame(rdkit_descriptors_list, columns=desc_names)

    # Select column which you interest
    desired_columns_df = pd.read_excel('data/desired_columns_rdkit.xlsx')
    desired_columns = desired_columns_df.iloc[:, 0].tolist()
    clean_desired_columns = [col.strip() for col in desired_columns]

    desired_df_rdkit_descriptors = df_rdkit_descriptors[clean_desired_columns]
    desired_rdkit_descriptors = desired_df_rdkit_descriptors.to_numpy()

    return desired_rdkit_descriptors


def Mordred_descriptors(smiles):

    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    df_mordred_descriptors = calc.pandas(mols)

    # Select column which you interest
    desired_columns_df = pd.read_excel('data/desired_columns_mordred.xlsx')
    desired_columns = desired_columns_df.iloc[:, 0].tolist()
    clean_desired_columns = [col.strip() for col in desired_columns]

    desired_df_mordred_descriptors = df_mordred_descriptors[clean_desired_columns]
    desired_mordred_descriptors = desired_df_mordred_descriptors.to_numpy()

    return desired_mordred_descriptors

