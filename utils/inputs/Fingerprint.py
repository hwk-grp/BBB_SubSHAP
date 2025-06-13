from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
import numpy as np

from utils.inputs import EMACCSkeys

def generate_fingerprints(smiles_list):
    maccs_fingerprints = []
    morgan_fingerprints = []
    pubchem_fingerprints = []

    for smiles in smiles_list:
        # SMILES to MOL type
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue

        # Generating MACCS Fingerprint
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_fingerprints.append(np.array(maccs_fp))

        # Generating Morgan Fingerprint (radius=3, dimension of bitvector=2048)
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        morgan_fingerprints.append(np.array(morgan_fp))

        # Generating PubChem Fingerprint (Avalon Fingerprint)
        pubchem_fp = pyAvalonTools.GetAvalonFP(mol, nBits=881)
        pubchem_fingerprints.append(np.array(pubchem_fp))

    maccs_fingerprints = np.array(maccs_fingerprints)
    morgan_fingerprints = np.array(morgan_fingerprints)
    pubchem_fingerprints = np.array(pubchem_fingerprints)

    return maccs_fingerprints, morgan_fingerprints, pubchem_fingerprints

def EMACCS_fingerprints(smiles_list):
    emaccs_fingerprints = []
    valid_smiles = []

    for smiles in smiles_list:
        # SMILES to MOL type
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue

        # Generating EMACCS Fingerprint
        emaccs_fp = EMACCSkeys._pyGenMACCSKeys(mol)
        emaccs_fingerprints.append(emaccs_fp)
        valid_smiles.append(smiles)  # 유효한 SMILES 저장

    return emaccs_fingerprints, valid_smiles
