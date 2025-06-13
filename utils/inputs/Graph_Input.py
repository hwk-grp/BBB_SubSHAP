import numpy
import pandas
import torch
from tqdm import tqdm
from mendeleev.fetch import fetch_table
from sklearn import preprocessing
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

elem_feat_names = ['atomic_weight', 'atomic_radius', 'dipole_polarizability',
                   'fusion_heat', 'evaporation_heat', 'thermal_conductivity', 'vdw_radius', 'covalent_radius_bragg',
                   'en_pauling']

n_atom_feats = 58
n_bond_feats = 6


def get_elem_feats():
    tb_atom_feats = fetch_table('elements')
    elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))

    return preprocessing.scale(elem_feats)


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# ---------------------------------------------
# Scaffold and pharmacophore information utils
# ---------------------------------------------
# tag pharmoco features to each atom
fun_smarts = {
    'Hbond_donor': '[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]',
    'Hbond_acceptor': '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&X2&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]',
    'Basic': '[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))]),$([n;X2;+0;-0])]',
    'Acid': '[C,S](=[O,S,P])-[O;H1,-1]',
    'Halogen': '[F,Cl,Br,I]'
}
FunQuery = dict([(pharmaco, Chem.MolFromSmarts(s)) for (pharmaco, s) in fun_smarts.items()])


def tag_pharmacophore(mol):
    for fungrp, qmol in FunQuery.items():
        matches = mol.GetSubstructMatches(qmol)
        match_idxes = []
        for mat in matches:
            match_idxes.extend(mat)
        for i, atom in enumerate(mol.GetAtoms()):
            tag = '1' if i in match_idxes else '0'
            atom.SetProp(fungrp, tag)
    return mol


# tag scaffold information to each atom
def tag_scaffold(mol):
    core = MurckoScaffold.GetScaffoldForMol(mol)
    match_idxes = mol.GetSubstructMatch(core)
    for i, atom in enumerate(mol.GetAtoms()):
        tag = '1' if i in match_idxes else '0'
        atom.SetProp('Scaffold', tag)
    return mol


def load_dataset(path_user_dataset, n_fp, n_radius, task):
    elem_feats = get_elem_feats()
    list_mols = list()
    id_target = numpy.array(pandas.read_excel(path_user_dataset))
    list_atom_types = list()

    for i in tqdm(range(0, id_target.shape[0])):

        mol = smiles_to_mol_graph(elem_feats, n_fp, n_radius, id_target[i, 0], idx=i, target=id_target[i, 2],
                                  list_atom_types=list_atom_types, task=task)

        if mol is not None:
            list_mols.append((id_target[i, 0], mol, id_target[i, 2]))

    return list_mols, list_atom_types


def smiles_to_mol_graph(elem_feats, n_fp, n_radius, smiles, idx, target, list_atom_types, task, explicit_H=False,
                        use_chirality=True, pharmaco=True, scaffold=True):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if pharmaco:
            mol = tag_pharmacophore(mol)
        if scaffold:
            mol = tag_scaffold(mol)

        atom_feats = []
        atom_types = list()
        bonds = list()
        bond_feats = list()

        # number of atoms
        n_atoms = mol.GetNumAtoms()
        atoms = mol.GetAtoms()
        atom_nums = [atom.GetAtomicNum() for atom in atoms]

        # adjacency matrix
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        adj_mat = adj_mat + numpy.eye(adj_mat.shape[0], dtype=int)

        # find atoms in ring structures
        atom_in_ring = numpy.array([mol.GetAtomWithIdx(i).IsInRing() for i in range(mol.GetNumAtoms())], dtype=int)

        # atomistic contributions to partition coefficient
        logp_contrib = numpy.array(rdMolDescriptors._CalcCrippenContribs(mol))

        # Zhu et al. atomic features (46 dimension)
        for i, atom in enumerate(mol.GetAtoms()):
            results = onehot_encoding_unk(
                atom.GetSymbol(),
                ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
                 ]) + onehot_encoding_unk(atom.GetDegree(),
                                          [0, 1, 2, 3, 4, 5, 'other']) + \
                      [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                      onehot_encoding_unk(atom.GetHybridization(), [
                          Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                          Chem.rdchem.HybridizationType.SP3D2, 'other'
                      ]) + [atom.GetIsAromatic()]
            if not explicit_H:
                results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                        [0, 1, 2, 3, 4])
            if use_chirality:
                try:
                    results = results + onehot_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
                except:
                    results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
            if pharmaco:
                results = results + [int(atom.GetProp('Hbond_donor'))] + [int(atom.GetProp('Hbond_acceptor'))] + \
                          [int(atom.GetProp('Basic'))] + [int(atom.GetProp('Acid'))] + \
                          [int(atom.GetProp('Halogen'))]
            if scaffold:
                results = results + [int(atom.GetProp('Scaffold'))]

            # additional 12 atomic features
            atom_feats_rdkit = numpy.array([atom_in_ring[i], logp_contrib[i][0], logp_contrib[i][1]])
            tmp_feats = numpy.append(elem_feats[atom.GetAtomicNum() - 1, :], atom_feats_rdkit)

            feat = numpy.concatenate((results, tmp_feats))
            atom_feats.append(feat)

        # bond feature

        for bond in mol.GetBonds():
            bond_feat = numpy.zeros(4)
            bond_type = bond.GetBondType()

            if bond_type == 'SINGLE':
                bond_feat[0] = 1
            elif bond_type == 'DOUBLE':
                bond_feat[1] = 1
            elif bond_type == 'TRIPLE':
                bond_feat[2] = 1
            elif bond_type == 'AROMATIC':
                bond_feat[3] = 1

            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bonds.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            bond_feats.append(bond_feat)

        if len(bonds) == 0:
            return None

        # list -> numpy array -> pytorch tensor
        atom_feats = torch.tensor(numpy.array(atom_feats), dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).cuda()

        bond_feats = torch.tensor(numpy.array(bond_feats), dtype=torch.float)
        if task == 'reg':
            y = torch.tensor(target, dtype=torch.float).view(-1, 1)
        elif task == 'clf':
            y = torch.tensor(target, dtype=torch.long).view(1)
        else:
            print('task {} is not available'.format(task))
            exit()
        atom_types = torch.tensor(numpy.array(atom_types), dtype=torch.int)

        mol_feats = list()
        mol_feats.append(ExactMolWt(mol))
        mol_feats.append(mol.GetRingInfo().NumRings())
        mol_feats.append(Descriptors.MolLogP(mol))
        mol_feats.append(Descriptors.MolMR(mol))
        mol_feats.append(Descriptors.NumHAcceptors(mol))
        mol_feats.append(Descriptors.NumHDonors(mol))
        mol_feats.append(Descriptors.NumHeteroatoms(mol))
        mol_feats.append(Descriptors.NumRotatableBonds(mol))
        mol_feats.append(Descriptors.TPSA(mol))
        mol_feats.append(Descriptors.qed(mol))

        mol_feats.append(rdMolDescriptors.CalcLabuteASA(mol))
        mol_feats.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
        mol_feats.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
        mol_feats.append(rdMolDescriptors.CalcNumAmideBonds(mol))
        mol_feats.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
        mol_feats.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
        mol_feats.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        mol_feats.append(rdMolDescriptors.CalcNumHeterocycles(mol))
        mol_feats.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
        mol_feats.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))

        mol_feats.append(rdMolDescriptors.CalcNumSaturatedRings(mol))

        mol_feats = numpy.append(mol_feats, MACCSkeys.GenMACCSKeys(mol))
        mol_feats = torch.tensor(numpy.array(mol_feats), dtype=torch.float).view(1, 188)

        return Data(x=atom_feats.cuda(), y=y.cuda(), edge_index=bonds.t().contiguous(), edge_attr=bond_feats.cuda(), id=idx, mol_feats=mol_feats.cuda(),
                    atom_types=atom_types, n_atoms=n_atoms, atom_nums=torch.tensor(atom_nums, dtype=torch.long))
    except:
        return None
