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
from torch_geometric.utils import softmax


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
    # Combine simulation and experiment datasets
    train_dataset = MultiDataset(train_exp_dataset, train_sim_dataset)
    val_dataset = MultiDataset(val_exp_dataset, val_sim_dataset)
    test_dataset = MultiDataset(test_exp_dataset, test_sim_dataset)
    data_example = train_dataset[0][1]

    mean, std = normalization(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    pred_sim_dataset = PolymerDataset(
        args.root_dir, 'pred', args.split, form_ring=form_ring, has_H=has_H)
    pred_dataset = MultiDataset(pred_sim_dataset, pred_sim_dataset)
    pred_loader = DataLoader(
        pred_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskNet(
        data_example.num_features, data_example.num_edge_features,
        args.fea_len, args.n_layers, args.n_h,
        use_sim_pred=bool(args.use_sim_pred)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=20,
                                                           min_lr=1e-5)

    def train(epoch):
        model.train()
        loss_all = 0

        for data in train_loader:
            data = [d.to(device) for d in data]
            optimizer.zero_grad()
            exp_out, sim_out = model(data)
            # Normalize y
            exp_y, sim_y = (data[0].y - mean) / std, (data[1].y - mean) / std
            assert 0. <= args.exp_weight <= 1.
            loss = (F.mse_loss(exp_out, exp_y) * args.exp_weight +
                    F.mse_loss(sim_out, sim_y) * (1 - args.exp_weight))
            loss.backward()
            assert data[0].num_graphs == data[1].num_graphs
            loss_all += loss.item() * data[0].num_graphs
            optimizer.step()
        return loss_all / len(train_loader.dataset)

    def test(loader):
        model.eval()
        exp_error = 0.
        sim_error = 0.
        exp_poly_ids = []
        sim_poly_ids = []
        exp_preds = []
        exp_targets = []
        sim_preds = []
        sim_targets = []

        for data in loader:
            data = [d.to(device) for d in data]
            exp_pred, sim_pred = model(data)
            # De-normalize prediction
            exp_pred = exp_pred * std + mean
            sim_pred = sim_pred * std + mean
            exp_error += (exp_pred - data[0].y).abs().sum().item()  # MAE
            sim_error += (sim_pred - data[1].y).abs().sum().item()  # MAE
            exp_preds.append(exp_pred.cpu().detach().numpy())
            exp_targets.append(data[0].y.cpu().detach().numpy())
            sim_preds.append(sim_pred.cpu().detach().numpy())
            sim_targets.append(data[1].y.cpu().detach().numpy())
            exp_poly_ids += data[0].poly_id
            sim_poly_ids += data[1].poly_id
        exp_preds = np.concatenate(exp_preds)
        exp_targets = np.concatenate(exp_targets)
        sim_preds = np.concatenate(sim_preds)
        sim_targets = np.concatenate(sim_targets)
        return (exp_error / len(loader.dataset),
                sim_error / len(loader.dataset),
                (exp_poly_ids, exp_targets, exp_preds,
                 sim_poly_ids, sim_targets, sim_preds))

    best_val_error = None
    for epoch in range(args.epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_exp_error, val_sim_error, _ = test(val_loader)
        scheduler.step(val_exp_error)  # TODO: decide which error to use

        if best_val_error is None or val_exp_error <= best_val_error:
            test_exp_error, test_sim_error, test_results = test(test_loader)
            best_val_error = val_exp_error
            write_results(test_results[:3], 'exp_test_results.csv')
            write_results(test_results[3:], 'sim_test_results.csv')
            torch.save(model.state_dict(), 'best_model.pth')

        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation exp MAE: '
              '{:.7f},  Validation sim MAE: {:.7f}, '
              'Best Validation MAE: {:.7f}, Test exp MAE: {:.7f}, '
              'Test sim MAE: {:.7f}'.format(
                  epoch, lr, loss, val_exp_error, val_sim_error,
                  best_val_error, test_exp_error, test_sim_error))

    # Predict on pred dataset
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    exp_poly_ids = []
    sim_poly_ids = []
    exp_preds = []
    exp_targets = []
    sim_preds = []
    sim_targets = []
    exp_smiles = []
    sim_smiles = []
    # attentions = []
    for data in pred_loader:
        data = [d.to(device) for d in data]
        exp_pred, sim_pred = model(data)
        # De-normalize prediction
        exp_pred = exp_pred * std + mean
        sim_pred = sim_pred * std + mean
        exp_preds.append(exp_pred.cpu().detach().numpy())
        exp_targets.append(data[0].y.cpu().detach().numpy())
        sim_preds.append(sim_pred.cpu().detach().numpy())
        sim_targets.append(data[1].y.cpu().detach().numpy())
        exp_poly_ids += data[0].poly_id
        sim_poly_ids += data[1].poly_id
        exp_smiles += data[0].smiles
        sim_smiles += data[1].smiles
        # attentions += get_attention(model, data)
    exp_preds = np.concatenate(exp_preds)
    exp_targets = np.concatenate(exp_targets)
    sim_preds = np.concatenate(sim_preds)
    sim_targets = np.concatenate(sim_targets)
    write_results((exp_poly_ids, exp_targets, exp_preds, exp_smiles),
                  'exp_pred_results.csv')
    write_results((sim_poly_ids, sim_targets, sim_preds, sim_smiles),
                  'sim_pred_results.csv')
    # write_attentions(poly_ids, smiles, attentions, 'attentions.csv')


def get_attention(model, data):
    """Get attention using layers in the model."""
    out = F.leaky_relu(model.node_embed(data.x))
    edge_attr = F.leaky_relu(model.edge_embed(data.edge_attr))

    for cgconv in model.cgconvs:
        out = cgconv(out, data.edge_index, edge_attr)

    size = data.batch[-1].item() + 1

    gate = model.pool.gate_nn(out).view(-1, 1)
    out = model.pool.nn(out) if model.pool.nn is not None else out
    assert gate.dim() == out.dim() and gate.size(0) == out.size(0)

    gate = softmax(gate, data.batch, num_nodes=size)
    gate = gate.squeeze(dim=-1)

    gate = gate.cpu().detach().numpy()
    batch = data.batch.cpu().detach().numpy()

    attentions = [[] for _ in range(size)]
    for g, b in zip(gate, batch):
        attentions[b].append(g)
    return attentions



if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)
