import argparse
import sys
import os.path as osp
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from polymernet.data import PolymerDataset
from polymernet.model import SingleTaskNet
from polymernet.rf_utils import dataset2morganfeatures
from torch_geometric.utils import softmax
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor


parser = argparse.ArgumentParser('Graph Network for polymers')
parser.add_argument('root_dir', help='path to directory that stores data')
parser.add_argument('--split', type=int, default=0,
                    help='CV split that is used for validation (default: 0)')
parser.add_argument('--pred-path', default=None, help='path to prediction csv')
parser.add_argument('--fea-len', type=int, default=16, help='feature length '
                    'for the network (default: 16)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--n-layers', type=int, default=4,
                    help='number of graph convolution layers (default: 3)')
parser.add_argument('--n-h', type=int, default=2,
                    help='number of hidden layers after pool (default: 2)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs (default: 200)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size (default: 16)')
parser.add_argument('--has-h', type=int, default=0,
                    help='whether to have explicit H (default: 0)')
parser.add_argument('--form-ring', type=int, default=1,
                    help='whether to form ring for molecules (default: 1)')
parser.add_argument('--log10', type=int, default=1,
                    help='whether to use the log10 of the property')


def write_results(results, fname):
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        for r in zip(*results):
            writer.writerow(r)


def write_attentions(poly_ids, smiles, attentions, fname):
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        for poly_id, s, a in zip(poly_ids, smiles, attentions):
            writer.writerow([poly_id, s] + a)


def normalization(dataset):
    ys = np.array([data.y for data in dataset])
    return ys.mean(), ys.std()


def main(args):
    has_H, form_ring = bool(args.has_h), bool(args.form_ring)
    log10 = bool(args.log10)
    train_dataset = PolymerDataset(
        args.root_dir, 'train', args.split, form_ring=form_ring, has_H=has_H,
        log10=log10)
    val_dataset = PolymerDataset(
        args.root_dir, 'val', args.split, form_ring=form_ring, has_H=has_H,
        log10=log10)
    test_dataset = PolymerDataset(
        args.root_dir, 'test', args.split, form_ring=form_ring, has_H=has_H,
        log10=log10)

    train_features, train_targets, _ = dataset2morganfeatures(train_dataset)
    val_features, val_targets, _ = dataset2morganfeatures(val_dataset)
    test_features, test_targets, test_poly_ids = dataset2morganfeatures(test_dataset)

    model = RandomForestRegressor()

    model.fit(train_features, train_targets)

    val_preds = model.predict(val_features)
    test_preds = model.predict(test_features)

    val_mae = np.mean(np.abs(val_preds - val_targets))
    test_mae = np.mean(np.abs(test_preds - test_targets))

    print('Validation MAE: {} test MAE: {}'.format(val_mae, test_mae))
    write_results([test_poly_ids, test_targets, test_preds], 'test_results.csv')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)
