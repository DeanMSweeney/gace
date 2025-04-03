import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing

class FlowGNN(MessagePassing, nn.Module): # MessagePassing inheritance 
    """
    Graph Neural Network.
    """
    def __init__(self, node_dim=3, edge_dim=1, hidden_dim=64):
        super().__init__(aggr="mean")  

        self.node_encoder = nn.Linear(node_dim + 1, hidden_dim)  # +1 for time
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Linear(hidden_dim, node_dim)

    def forward(self, x, t, edge_index, edge_attr):
        """
        Forward pass of positions, velocity through the the network
        """
        x_time = torch.cat([x, t], dim=-1)
        x_encoded = self.node_encoder(x_time)
        edge_encoded = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=x_encoded, edge_attr=edge_encoded)
        return self.decoder(out)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return self.processor(aggr_out)

    def step(self, xt, t_start, t_end, edge_index, edge_attr):
        """
        Performs one integration step from t_start to t_end
        """
        # Predict velocity at t_start
        t = t_start * torch.ones(len(xt), 1)  # Expand time to match batch
        u_start = self(xt, t, edge_index, edge_attr)
        
        # Update to midpoint (t_mid = t_start + Δt/2)
        xt_mid = xt + u_start * (t_end - t_start) / 2
        
        # Predict velocity at midpoint
        t_mid = (t_start + t_end) / 2 * torch.ones(len(xt), 1)
        u_mid = self(xt_mid, t_mid, edge_index, edge_attr)
        
        # Full step update
        return xt + u_mid * (t_end - t_start)


