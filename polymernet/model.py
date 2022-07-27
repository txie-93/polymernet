import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.nn import CGConv, GlobalAttention, NNConv, Set2Set


class MultiTaskNet(torch.nn.Module):
    """MultiTask network to train both simulation and experiment data."""

    def __init__(self, node_in_len, edge_in_len, fea_len, n_layers, n_h,
                 use_sim_pred=False):
        super(MultiTaskNet, self).__init__()

        self.use_sim_pred = use_sim_pred

        self.node_embed = Linear(node_in_len, fea_len)
        self.edge_embed = Linear(edge_in_len, fea_len)
        self.cgconvs = nn.ModuleList([
            CGConv(fea_len, fea_len, aggr='mean', batch_norm=True)
            for _ in range(n_layers)])

        self.pool = GlobalAttention(
            gate_nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, 1)),
            nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, fea_len)))
        self.hs = nn.ModuleList(
            [Linear(fea_len, fea_len) for _ in range(n_h - 1)])
        self.out = Linear(fea_len, 1)
        if self.use_sim_pred:
            self.h_exp = Linear(fea_len + 1, 1)
        else:
            self.h_exp = Linear(fea_len, 1)

    def forward(self, data):
        exp_data, sim_data = data

        exp_out = F.leaky_relu(self.node_embed(exp_data.x))
        exp_edge_attr = F.leaky_relu(self.edge_embed(exp_data.edge_attr))
        sim_out = F.leaky_relu(self.node_embed(sim_data.x))
        sim_edge_attr = F.leaky_relu(self.edge_embed(sim_data.edge_attr))

        for cgconv in self.cgconvs:
            exp_out = cgconv(exp_out, exp_data.edge_index, exp_edge_attr)
            sim_out = cgconv(sim_out, sim_data.edge_index, sim_edge_attr)

        exp_out = self.pool(exp_out, exp_data.batch)
        sim_out = self.pool(sim_out, sim_data.batch)

        # Make a copy, send it to simulation branch
        exp_by_sim_out = exp_out

        for hidden in self.hs:
            sim_out = F.leaky_relu(hidden(sim_out))
            exp_by_sim_out = F.leaky_relu(hidden(exp_by_sim_out))
        sim_out = self.out(sim_out)
        exp_by_sim_out = self.out(exp_by_sim_out)

        # Transfer learning
        if self.use_sim_pred:
            exp_out = torch.cat([exp_out, exp_by_sim_out], axis=-1)
        exp_out = self.h_exp(exp_out)

        return (torch.squeeze(exp_out, dim=-1), torch.squeeze(sim_out, dim=-1))


class SingleTaskNet(torch.nn.Module):
    """CGConv + Global attention pooling."""

    def __init__(self, node_in_len, edge_in_len, fea_len, n_layers, n_h):
        super(SingleTaskNet, self).__init__()
        self.node_embed = Linear(node_in_len, fea_len)
        self.edge_embed = Linear(edge_in_len, fea_len)
        self.cgconvs = nn.ModuleList([
            CGConv(fea_len, fea_len, aggr='mean', batch_norm=True)
            for _ in range(n_layers)])

        self.pool = GlobalAttention(
            gate_nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, 1)),
            nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, fea_len)))
        self.hs = nn.ModuleList(
            [Linear(fea_len, fea_len) for _ in range(n_h - 1)])
        self.out = Linear(fea_len, 1)

    def forward(self, data):
        out = F.leaky_relu(self.node_embed(data.x))
        edge_attr = F.leaky_relu(self.edge_embed(data.edge_attr))

        for cgconv in self.cgconvs:
            out = cgconv(out, data.edge_index, edge_attr)

        out = self.pool(out, data.batch)

        for hidden in self.hs:
            out = F.leaky_relu(hidden(out))
        out = self.out(out)
        return torch.squeeze(out, dim=-1)