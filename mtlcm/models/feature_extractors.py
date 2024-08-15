import dgl
from dgl.nn.pytorch import EGNNConv
import torch.nn as nn
import torch.nn.functional as F


class EGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_hidden):
        super(EGNN, self).__init__()
        self.conv1 = EGNNConv(in_feats, h_feats, out_size=h_feats, edge_feat_size=4)
        self.conv2 = EGNNConv(h_feats, num_hidden, num_hidden, edge_feat_size=4)

    def forward(self, g):
        node_feat = g.ndata["attr"]
        coord_feat = g.ndata["pos"]
        edge_feat = g.edata["edge_attr"]
        h, x = self.conv1(g, node_feat, coord_feat, edge_feat)
        h = F.relu(h)
        x = F.relu(x)
        h, x = self.conv2(g, h, x, edge_feat)

        g.ndata["h"] = h
        g.ndata["x"] = x

        out = dgl.sum_nodes(g, "h")

        return out
