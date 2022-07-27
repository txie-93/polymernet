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
from torch_geometric.utils import softmax


parser = argparse.ArgumentParser('Graph Network for polymers')
parser.add_argument('root_dir', help='path to directory that stores data')
parser.add_argument('--split', type=int, default=0,
                    help='CV split that is used for validation (default: 0)')
parser.add_argument('--total-split', type=int, default=10,
                    help='Total number of CV splits (default: 10)')
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
parser.add_argument('--size-limit', type=int, default=None,
                    help='limit the size of training data (default: None)')


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
        log10=log10, total_split=args.total_split, size_limit=args.size_limit)
    val_dataset = PolymerDataset(
        args.root_dir, 'val', args.split, form_ring=form_ring, has_H=has_H,
        log10=log10, total_split=args.total_split)
    test_dataset = PolymerDataset(
        args.root_dir, 'test', args.split, form_ring=form_ring, has_H=has_H,
        log10=log10, total_split=args.total_split)
    data_example = train_dataset[0]

    mean, std = normalization(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    pred_dataset = PolymerDataset(
        args.root_dir, 'pred', args.split, form_ring=form_ring, has_H=has_H,
        total_split=args.total_split)
    pred_loader = DataLoader(
        pred_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SingleTaskNet(
        data_example.num_features, data_example.num_edge_features,
        args.fea_len, args.n_layers, args.n_h).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=20,
                                                           min_lr=1e-5)

    def train(epoch):
        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # Normalize y
            loss = F.mse_loss(model(data), ((data.y - mean) / std))
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_loader.dataset)

    def test(loader):
        model.eval()
        error = 0
        poly_ids = []
        preds = []
        targets = []

        for data in loader:
            data = data.to(device)
            pred = model(data)
            # De-normalize prediction
            pred = pred * std + mean
            error += (pred - data.y).abs().sum().item()  # MAE
            preds.append(pred.cpu().detach().numpy())
            targets.append(data.y.cpu().detach().numpy())
            poly_ids += data.poly_id
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        return error / len(loader.dataset), (poly_ids, targets, preds)

    best_val_error = None
    for epoch in range(args.epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_error, _ = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, test_results = test(test_loader)
            best_val_error = val_error
            write_results(test_results, 'test_results.csv')
            torch.save(model.state_dict(), 'best_model.pth')

        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Best Validation MAE: {:.7f}, Test MAE: {:.7f}'.format(
                  epoch, lr, loss, val_error, best_val_error, test_error))

    # Predict on pred dataset
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    poly_ids = []
    preds = []
    targets = []
    smiles = []
    attentions = []
    for data in pred_loader:
        data = data.to(device)
        pred = model(data)
        # De-normalize prediction
        pred = pred * std + mean
        preds.append(pred.cpu().detach().numpy())
        targets.append(data.y.cpu().detach().numpy())
        poly_ids += data.poly_id
        smiles += data.smiles
        attentions += get_attention(model, data)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    write_results((poly_ids, targets, preds, smiles), 'pred_results.csv')
    write_attentions(poly_ids, smiles, attentions, 'attentions.csv')


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
