import torch
from torch.nn import Sequential as Seq
from torch.nn import Linear, ReLU, Sigmoid, Conv2d, Conv1d, Tanh
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import DataLoader

class GraphOperators(torch.nn.Module):
    def __init__(self, power = 2, num_edge_attr = 33, hidden_nodes = 16):
        super(GraphOperators, self).__init__()
        self.power = power
        # self.edge_attr_mlp = Seq(Linear(num_edge_attr, hidden_nodes), ReLU(), Linear(hidden_nodes, 1), Tanh())
    
    def forward(self, edge_index, edge_attr):
        # learned_attr = self.edge_attr_mlp(edge_attr)
        adj = to_dense_adj(edge_index=edge_index).type(torch.float32) # convert sparse to adjacency matrix
        adj = torch.squeeze(adj) # remove dimension with lenth 1
        adj_0 = torch.eye(adj.shape[0])
        powers_adj = adj_0
        powers_adj = torch.stack((powers_adj, adj))
        for p in range(2,self.power+1):
            adj_p = torch.matrix_power(adj, p)
            adj_p = torch.unsqueeze(adj_p, 0)
            powers_adj = torch.cat((powers_adj, adj_p))
        return powers_adj

class Embedding(torch.nn.Module):
    def __init__(self, n_feature, num_embedding):
        super(Embedding, self).__init__()
        self.emb = Seq(Linear(n_feature, num_embedding), Tanh())

    def forward(self, x):
        return self.emb(x)

class AdjacencyLearning(torch.nn.Module):
    def __init__(self, num_features, hidden_nodes = 3):
        # power = 2 adjacency matrixs
        super(AdjacencyLearning, self).__init__()
        self.mlp_0 = Seq(Linear(num_features, hidden_nodes), ReLU(), Linear(hidden_nodes, 1), Sigmoid())
        self.mlp_1 = Seq(Linear(num_features, hidden_nodes), ReLU(), Linear(hidden_nodes, 1), Sigmoid())
        self.mlp_2 = Seq(Linear(num_features, hidden_nodes), ReLU(), Linear(hidden_nodes, 1), Sigmoid())
        self.mlp = [self.mlp_0, self.mlp_1, self.mlp_2]

    def forward(self, x, powers_adj):
        # x is powers adjacency matrixs
        # output is list of tuple of edge_index and weight
        edge_index_powers = [dense_to_sparse(adj)[0] for adj in powers_adj]
        edge_weight_powers = [self._learn_adjacencies(x, edge_index, i) for i, edge_index in enumerate(edge_index_powers)]

        return [(i, w) for i, w in zip(edge_index_powers, edge_weight_powers)]

    def _learn_adjacencies(self, x, edge_index, mlp_index):

        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        abs_dif = torch.abs(x_j - x_i)
        edge_weight = self.mlp[mlp_index](abs_dif).view(-1)
        return edge_weight

class AdjacencyLearningClassifier(torch.nn.Module):
    def __init__(self, num_features, hidden_nodes = 4):
        super(AdjacencyLearningClassifier, self).__init__()
        self.mlp = Seq(Linear(num_features, hidden_nodes), ReLU(), Linear(hidden_nodes, 2))

    def forward(self, x, edge_index):
        edge_weight = self._learn_adjacencies(x, edge_index)

        return edge_weight

    def _learn_adjacencies(self, x, edge_index):
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        abs_dif = torch.abs(x_j - x_i)
        edge_weight = self.mlp(abs_dif).view((-1, 2))
        return edge_weight

class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.gcn_0 = GCNConv(num_features, hidden_channels, add_self_loops=False)
        self.gcn_1 = GCNConv(num_features, hidden_channels, add_self_loops=False)
        self.gcn_2 = GCNConv(num_features, hidden_channels, add_self_loops=False)
    
    def forward(self, x, powers_weight_edge):
        B0 = powers_weight_edge[0]
        B1 = powers_weight_edge[1]
        B2 = powers_weight_edge[2]
        h_0 = self.gcn_0(x, B0[0], B0[1])
        h_1 = self.gcn_0(x, B1[0], B1[1])
        h_2 = self.gcn_0(x, B2[0], B2[1])
        x = torch.sum(torch.stack([h_0, h_1, h_2]), dim = 0)
        out = F.relu(x)

        return out

class GraphResidualBlock(torch.nn.Module):
    def __init__(self, num_in_feature, num_hidden_feature):
        super(GraphResidualBlock, self).__init__()
        self.AL = AdjacencyLearning(num_in_feature, num_hidden_feature)
        self.GNN_0 = GNN(num_in_feature, num_hidden_feature)
        self.GNN_1 = GNN(num_hidden_feature, num_in_feature)
        self.batch_norm_0 = BatchNorm(num_hidden_feature)
        self.batch_norm_1 = BatchNorm(num_in_feature)

    def forward(self, x, powers_adj):
        residual = x
        learned_adj = self.AL(x, powers_adj)
        out = self.GNN_0(x, learned_adj)
        out = self.batch_norm_0(out).relu()
        out = self.GNN_1(out, learned_adj)
        out = self.batch_norm_1(out)
        out += residual

        return out


class ResGraph(torch.nn.Module):
    def __init__(self, num_in_feature, num_embedding, num_hidden_feature, num_class):
        super(ResGraph, self).__init__()
        self.num_in_feature = num_in_feature
        self.num_hidden_feature = num_hidden_feature
        self.num_class = num_class
        self.num_embedding = num_embedding

        self.graph_operator = GraphOperators(2)
        self.embedding = Embedding(self.num_in_feature, self.num_embedding)
        self.grb_0 = GraphResidualBlock(self.num_embedding, self.num_hidden_feature)
        self.grb_1 = GraphResidualBlock(self.num_embedding, self.num_hidden_feature)
        self.grb_2 = GraphResidualBlock(self.num_embedding, self.num_hidden_feature)
        self.adj_classifier = AdjacencyLearningClassifier(self.num_embedding ,self.num_hidden_feature)
        self.linear = Linear(self.num_embedding, num_class)

    def forward(self, data):
        powers_adj = self.graph_operator(data.edge_index, data.edge_attr)
        embedded = self.embedding(data.x)
        out = self.grb_0(embedded, powers_adj)
        out = self.grb_1(out, powers_adj)
        out = self.grb_2(out, powers_adj)
        edge_learned_weight = self.adj_classifier(out, data.edge_index)
        out = self.linear(out)
        return out, edge_learned_weight
