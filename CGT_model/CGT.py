import torch
from torch.nn import Linear, Embedding, ModuleList, Sequential, ReLU
from torch_geometric.nn import GINConv
from CGT_model.gps_conv import GPSConv
from aug.DropPath import DropPath


class CGT(torch.nn.Module):
    def __init__(self, fea_dim, channels: int, num_layers: int, num_tasks, num_features):
        super().__init__()
        self.node_emb = Linear(fea_dim, channels)
        self.pe_lin = Linear(20, channels)
        self.edge_emb = Embedding(4, channels)
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINConv(nn), heads=4, attn_dropout=0.5, attn_drophead=0.0)
            self.convs.append(conv)

        self.lin = Linear(channels, num_tasks)
        self.drop_path_prob = 0.0
        self.drop_path = DropPath(self.drop_path_prob)
        self.gumbel = torch.nn.Linear(num_features, 1)

    def forward(self, x, x1, x2, edge_index1, edge_index2, pe, edge_index, edge_attr, batch):
        x = self.node_emb(x) + self.pe_lin(pe)
        x1 = self.node_emb(x1) + self.pe_lin(pe)
        x2 = self.node_emb(x2) + self.pe_lin(pe)
        # edge_attr = self.edge_emb(edge_attr)

        edge_attr = None

        for conv in self.convs:
            x = x + self.drop_path(conv(x, edge_index, batch))
            x1 = x1 + self.drop_path(conv(x1, edge_index1, batch))
            x2 = x2 + self.drop_path(conv(x2, edge_index2, batch))

        x = global_add_pool(x, batch)
        x1 = global_add_pool(x1, batch)
        x2 = global_add_pool(x2, batch)
        return x, x1, x2