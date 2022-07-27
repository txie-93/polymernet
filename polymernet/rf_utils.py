import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from .data import process_smiles

def smiles2morgan(smiles):
    mol = process_smiles(smiles, form_ring=True, has_H=False)
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features


def dataset2morganfeatures(dataset):
    data = [dataset[i] for i in range(len(dataset))]
    features = np.array([smiles2morgan(data[i].smiles) for i in range(len(dataset))])
    targets = np.array([data[i].y.item() for i in range(len(dataset))])
    poly_ids = [data[i].poly_id for i in range(len(dataset))]
    return features, targets, poly_ids
