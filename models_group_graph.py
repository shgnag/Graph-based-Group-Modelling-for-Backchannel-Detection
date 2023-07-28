import torch
import torch.nn as nn
import torch.nn.parameter

from torch_geometric.nn import EdgeConv, DynamicEdgeConv, GraphConv, GCNConv


class LinearPathPreact(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(LinearPathPreact, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu(x1)
        x1 = self.fc1(x1)
        return x1

class GroupGraph(torch.nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(GroupGraph, self).__init__()
        self.dim_reduction_audio = nn.Linear(in_feats, hidden_size)
        self.dim_reduction_main = nn.Linear(in_feats, hidden_size)
        self.dim_reduction_context = nn.Linear(in_feats, hidden_size)
        self.dim_reduction_v2 = nn.Linear(in_feats, hidden_size)

        self.next1 = EdgeConv(LinearPathPreact(hidden_size*8, hidden_size*2))
        self.next2 = EdgeConv(LinearPathPreact(hidden_size*2*2, hidden_size*2))
        self.next3 = EdgeConv(LinearPathPreact(hidden_size*2*2, hidden_size*2))

        self.gconv1 = GCNConv(hidden_size, hidden_size*2)
        self.gconv2 = GCNConv(hidden_size*2, hidden_size*2)
        self.gconv3 = GCNConv(hidden_size*2, hidden_size*4)

        self.fc = nn.Linear(hidden_size*2, 1)
        self.dropout = nn.Dropout(0.3)
        self.act = nn.Tanh()

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x.shape)
        x0_main = self.dim_reduction_main(x)
        x0 = self.dim_reduction_context(x)
        x0[0::4, :] = x0_main[0::4, :].clone()

        x_gconv1 = self.gconv1(x0, edge_index)
        x_gconv2 = self.gconv2(x_gconv1, edge_index)
        x_gconv3 = self.gconv3(x_gconv2, edge_index)


        x1 = self.next1(x_gconv3, edge_index)

        x2 = self.next2(x1, edge_index)

        x3 = self.next3(x2, edge_index)

        return self.fc(x3)
