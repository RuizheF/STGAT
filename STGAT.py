import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(TemporalConv, self).__init__()
        padding = (0, kernel_size // 2)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        X = X.permute(0, 3, 2, 1)
        P = self.conv_1(X)
        Q = torch.sigmoid(self.conv_2(X))
        PQ = P * Q
        H = F.relu(PQ + self.conv_3(X))
        H = F.dropout(H, p=0.5, training=self.training)
        H = H.permute(0, 3, 2, 1)
        return H

class GAT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        num_layers: int
    ):
        super(GAT, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_layers = num_layers
        self.conv_0 = GATConv(20, hidden_channels, heads=heads, concat=True, dropout=0.5, add_self_loops=True, edge_dim=1, fill_value=0, bias=True)
        self.bn_0 = nn.BatchNorm1d(hidden_channels * heads, eps=1e-5, momentum=0.1)

        # self.conv_1 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=0.5, add_self_loops=True, edge_dim=1, fill_value=0, bias=True)
        # self.bn_1 = nn.BatchNorm1d(hidden_channels * heads, eps=1e-5, momentum=0.1)
        # self.conv_2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False, dropout=0.5, add_self_loops=True, edge_dim=1, fill_value=0, bias=True)
        # self.bn_2 = nn.BatchNorm1d(hidden_channels, eps=1e-5, momentum=0.1)

        self.layers = nn.ModuleList([
            GATConv(20 if i == 0 else hidden_channels * heads,
                    hidden_channels, heads=heads, concat=True,
                    dropout=0.5, add_self_loops=True, edge_dim=1)
            for i in range(num_layers)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels * heads)
            for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_channels * heads, hidden_channels)
        self.activation = F.relu

    def forward(self, data, y):

        if y == 't': x = data.t
        elif y == 's': x = data.s

        edge_index = data.edge_index
        edge_weight = data.edge_weight

        x_0 = self.conv_0(x, edge_index, edge_weight)
        x_0 = self.bn_0(x_0)
        x_0 = self.activation(x_0)

        for i in range(self.num_layers - 1):
            x = self.layers[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, 0.5, training=self.training)

        x = self.layers[self.num_layers-1](x, edge_index, edge_weight)
        x = x + x_0
        x = self.bns[self.num_layers-1](x)
        x = self.activation(x)
        x = self.fc(x)
        x = F.dropout(x, 0.5, training=self.training)

        return x

class GAT_TCN(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 heads: int = 3,
                 num_layers: int = 2,
                 kernel_size: int = 3
    ):
        super(GAT_TCN, self).__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.kernel_size = kernel_size

        self.tcn_1 = TemporalConv(in_channels, hidden_channels, kernel_size=kernel_size)
        self.tcn_2 = TemporalConv(hidden_channels, out_channels, kernel_size=kernel_size)
        self.gat_1 = GAT(num_nodes, 20, hidden_channels, out_channels, heads, num_layers)
        self.gat_2 = GAT(num_nodes, 20, hidden_channels, out_channels, heads, num_layers)

        self.linear_residual = nn.Linear(1, hidden_channels)

        self.bn = nn.BatchNorm2d(num_nodes)
        self.fc = nn.Linear(self.num_nodes * (20 + 20), self.out_channels * 2048)
        self.fc1 = nn.Linear(2048, 300)

    def forward(self, data) -> torch.FloatTensor:
        # 对残差进行处理
        r = data.r  # Assuming 'r' contains the residual data
        r = r.reshape(-1, 20, self.num_nodes, self.in_channels)  # Shape: [batch, num_nodes, time_steps, in_channels]

        # Apply linear layer to each time step of r
        r = self.linear_residual(r)
        r = F.relu(r)
        r = F.dropout(r, p=0.5, training=self.training)

        t = self.gat_1(data, 't')
        t = t.permute(1, 0).reshape(-1, 1, self.num_nodes, self.hidden_channels)
        s = self.gat_2(data, 's')
        s = s.permute(1, 0).reshape(-1, 1, self.num_nodes, self.hidden_channels)

        fused_features_list = []
        for t_idx in range(r.size(1)):
            r_t = r[:, t_idx, :, :]

            fused_t = r_t + t.squeeze(1)
            fusion = fused_t + s.squeeze(1)

            fused_features_list.append(fusion)

        fusions = torch.stack(fused_features_list, dim=1)


        x = data.x.reshape(-1, 20, self.num_nodes, self.in_channels)
        x = self.tcn_1(x)

        # Combine the outputs of TCN and GAT
        res = torch.cat((x, fusions), dim=1)
        res = self.tcn_2(res)

        res = res.permute(0, 2, 1, 3)  # Adjust the shape for batch normalization
        res = self.bn(res)
        res = res.permute(0, 2, 1, 3)
        res = res.reshape(res.size(0), -1)  # Flatten for FC layer

        # Fully connected layers for final prediction
        res = F.relu(self.fc(res))
        res = self.fc1(res)

        return res
