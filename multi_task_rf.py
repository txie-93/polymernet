import argparse
import sys
import os.path as osp
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from polymernet.data import PolymerDataset, MultiDataset
from polymernet.model import MultiTaskNet
from polymernet.rf_utils import dataset2morganfeatures
from torch_geometric.utils import softmax
from sklearn.ensemble import RandomForestRegressor


parser = argparse.ArgumentParser('Graph Network for polymers')
parser.add_argument('root_dir', help='path to directory that stores data')
parser.add_argument('root_dir_exp', help='path to directory that stores data')
parser.add_argument('--split', type=int, default=0,
                    help='CV split that is used for validation (default: 0)')
parser.add_argument('--pred-path', default=None, help='path to prediction csv')
parser.add_argument('--fea-len', type=int, default=8, help='feature length '
                    'for the network (default: 8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--n-layers', type=int, default=4,
                    help='number of graph convolution layers (default: 4)')
parser.add_argument('--n-h', type=int, default=2,
                    help='number of hidden layers after pool (default: 2)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs (default: 200)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='batch size (default: 8)')
parser.add_argument('--has-h', type=int, default=0,
                    help='whether to have explicit H (default: 0)')
parser.add_argument('--form-ring', type=int, default=1,
                    help='whether to form ring for molecules (default: 1)')
parser.add_argument('--exp-weight', type=float, default=0.1, help='weight of '
                    'experiment dataset in loss. (default: 0.1)')
parser.add_argument('--use-sim-pred', type=int, default=1,
                    help='whether to use simulation prediction (default: 1')


def append_label_feature(features, label):
    return np.concatenate([features, np.ones((features.shape[0], 1)) * label], axis=1)


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
    ys = np.array([data_sim.y for data_exp, data_sim in dataset])
    return ys.mean(), ys.std()


def convert_exp_features(model, exp_features):
    sim_preds = model.predict(exp_features)
    exp_features = np.concatenate([exp_features, sim_preds.reshape(-1, 1)], axis=1)
    return exp_features


def main(args):
    has_H, form_ring = bool(args.has_h), bool(args.form_ring)
    train_sim_dataset = PolymerDataset(
        args.root_dir, 'train', args.split, form_ring=form_ring, has_H=has_H)
    val_sim_dataset = PolymerDataset(
        args.root_dir, 'val', args.split, form_ring=form_ring, has_H=has_H)
    test_sim_dataset = PolymerDataset(
        args.root_dir, 'test', args.split, form_ring=form_ring, has_H=has_H)
    train_exp_dataset = PolymerDataset(
        args.root_dir_exp, 'train', args.split, form_ring=form_ring, has_H=has_H)
    val_exp_dataset = PolymerDataset(
        args.root_dir_exp, 'val', args.split, form_ring=form_ring, has_H=has_H)
    test_exp_dataset = PolymerDataset(
        args.root_dir_exp, 'test', args.split, form_ring=form_ring, has_H=has_H)

    train_sim_features, train_sim_targets, _ = dataset2morganfeatures(train_sim_dataset)
    val_sim_features, val_sim_targets, _ = dataset2morganfeatures(val_sim_dataset)
    test_sim_features, test_sim_targets, test_sim_poly_ids = dataset2morganfeatures(test_sim_dataset)

    train_exp_features, train_exp_targets, _ = dataset2morganfeatures(train_exp_dataset)
    val_exp_features, val_exp_targets, _ = dataset2morganfeatures(val_exp_dataset)
    test_exp_features, test_exp_targets, test_exp_poly_ids = dataset2morganfeatures(test_exp_dataset)

    # train_sim_features = append_label_feature(train_sim_features, 0.)
    # val_sim_features = append_label_feature(val_sim_features, 0.)
    # test_sim_features = append_label_feature(test_sim_features, 0.)

    # train_exp_features = append_label_feature(train_exp_features, 1.)
    # val_exp_features = append_label_feature(val_exp_features, 1.)
    # test_exp_features = append_label_feature(test_exp_features, 1.)

    # all_train_features = np.concatenate([train_sim_features, train_exp_features], axis=0)
    # all_train_targets = np.concatenate([train_sim_targets, train_exp_targets], axis=0)

    # model = RandomForestRegressor()

    # model.fit(all_train_features, all_train_targets)

    # val_sim_preds = model.predict(val_sim_features)
    # val_exp_preds = model.predict(val_exp_features)
    # test_sim_preds = model.predict(test_sim_features)
    # test_exp_preds = model.predict(test_exp_features)

    model_sim = RandomForestRegressor()
    model_sim.fit(train_sim_features, train_sim_targets)
    val_sim_preds = model_sim.predict(val_sim_features)
    test_sim_preds = model_sim.predict(test_sim_features)

    train_exp_features = convert_exp_features(model_sim, train_exp_features)
    val_exp_features = convert_exp_features(model_sim, val_exp_features)
    test_exp_features = convert_exp_features(model_sim, test_exp_features)

    model_exp = RandomForestRegressor()
    model_exp.fit(train_exp_features, train_exp_targets)
    val_exp_preds = model_exp.predict(val_exp_features)
    test_exp_preds = model_exp.predict(test_exp_features)

    val_sim_mae = np.mean(np.abs(val_sim_preds - val_sim_targets))
    val_exp_mae = np.mean(np.abs(val_exp_preds - val_exp_targets))
    test_sim_mae = np.mean(np.abs(test_sim_preds - test_sim_targets))
    test_exp_mae = np.mean(np.abs(test_exp_preds - test_exp_targets))

    print('Validation sim MAE: {} exp MAE: {}'.format(val_sim_mae, val_exp_mae))
    print('Test sim MAE: {} exp MAE: {}'.format(test_sim_mae, test_exp_mae))

    write_results([test_exp_poly_ids, test_exp_targets, test_exp_preds], 'exp_test_results.csv')
    write_results([test_sim_poly_ids, test_sim_targets, test_sim_preds], 'sim_test_results.csv')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)
