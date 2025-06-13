import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
#from model.modules.NGAT_Conv import GATConv

class GCN(nn.Module):
    def __init__(self, num_node_feats, d_hidden, d_emb, dim_out):
        super(GCN, self).__init__()

        self.input = None

        self.gc1 = GCNConv(num_node_feats, d_hidden)
        self.bn1 = BatchNorm(d_hidden)
        self.gc2 = GCNConv(d_hidden, d_hidden)
        self.bn2 = BatchNorm(d_hidden)
        self.gc3 = GCNConv(d_hidden, d_hidden)
        self.fc1 = nn.Linear(d_hidden, d_emb)
        self.fc2 = nn.Linear(d_emb, dim_out)

    def forward(self, g):
        h1 = F.selu(self.gc1(g.x, g.edge_index))
        h1 = self.bn1(h1)
        h2 = F.selu(self.gc2(h1, g.edge_index))
        h2 = self.bn2(h2)
        h3 = F.relu(self.gc3(h2, g.edge_index))

        hg = global_add_pool(h3, g.batch)

        h = F.selu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def prt_emb(self, g):
        h1 = F.selu(self.gc1(g.x, g.edge_index))
        h1 = self.bn1(h1)
        h2 = F.selu(self.gc2(h1, g.edge_index))
        h2 = self.bn2(h2)
        h3 = F.selu(self.gc3(h2, g.edge_index))

        hg = global_add_pool(h3, g.batch)

        h = F.selu(self.fc1(hg))

        return h

class EGCN(nn.Module):
    def __init__(self, num_node_feats, num_graph_feats, d_hidden, d_emb, dim_out):
        super(EGCN, self).__init__()

        self.input = None

        self.gc1 = GCNConv(num_node_feats, d_hidden)
        self.bn1 = BatchNorm(d_hidden)
        self.gc2 = GCNConv(d_hidden, d_hidden)
        self.bn2 = BatchNorm(d_hidden)
        self.gc3 = GCNConv(d_hidden, d_hidden)

        self.fc1 = nn.Linear(d_hidden + num_graph_feats, d_emb)
        self.fc2 = nn.Linear(d_emb, dim_out)

    def forward(self, g):
        h1 = F.selu(self.gc1(g.x, g.edge_index))
        h1 = self.bn1(h1)
        h2 = F.selu(self.gc2(h1, g.edge_index))
        h2 = self.bn2(h2)
        h3 = F.selu(self.gc3(h2, g.edge_index))

        hg = global_add_pool(h3, g.batch)
        
        h = F.selu(self.fc1(torch.cat([hg, g.mol_feats], dim=1)))
        out = self.fc2(h)

        return out

    def prt_emb(self, g):
        h1 = F.selu(self.gc1(g.x, g.edge_index))
        h1 = self.bn1(h1)
        h2 = F.selu(self.gc2(h1, g.edge_index))
        h2 = self.bn2(h2)
        h3 = F.selu(self.gc3(h2, g.edge_index))

        hg = global_add_pool(h3, g.batch)

        h = F.selu(self.fc1(torch.cat([hg, g.mol_feats], dim=1)))

        return h

class GAT(nn.Module):
    def __init__(self, num_node_feats, d_hidden, d_emb, dim_out):
        super(GAT, self).__init__()

        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        self.gc1 = GATConv(num_node_feats, d_hidden, heads=4, concat=True)
        self.bn1 = BatchNorm(d_hidden * 4)
        self.gc2 = GATConv(d_hidden * 4, d_hidden, heads=4, concat=True)
        self.bn2 = BatchNorm(d_hidden * 4)
        self.gc3 = GATConv(d_hidden * 4, d_hidden, heads=4, concat=False)
        self.fc1 = nn.Linear(d_hidden, d_emb)
        self.fc2 = nn.Linear(d_emb, dim_out)

    def forward(self, g):
        h1 = F.selu(self.gc1(g.x, g.edge_index))
        h1 = self.bn1(h1)
        h2 = F.selu(self.gc2(h1, g.edge_index))
        h2 = self.bn2(h2)
        h3 = F.selu(self.gc3(h2, g.edge_index))

        hg = global_add_pool(h3, g.batch)

        h = F.selu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def prt_emb(self, g):
        h1 = F.selu(self.gc1(g.x, g.edge_index))
        h1 = self.bn1(h1)
        h2 = F.selu(self.gc2(h1, g.edge_index))
        h2 = self.bn2(h2)
        h3 = F.selu(self.gc3(h2, g.edge_index))

        hg = global_add_pool(h3, g.batch)

        h = F.selu(self.fc1(hg))

        return h
