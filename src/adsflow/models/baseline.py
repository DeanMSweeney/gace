import torch
import torch.nn as nn
from torch_geometric.data import Data

# 1) Define the tiny MLP baseline
class BaselineVel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dim = flattened x_t (2 atoms × 3 dims) + t (1 dim) = 7
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 6),   # output = 2 atoms × 3 dims
        )

    def forward(self, data):
        x_flat = data.pos.view(1, -1)  # [1,6]
        t_flat = data.t.view(1, 1)     # [1,1]
        out = self.net(torch.cat([x_flat, t_flat], dim=1))
        return out.view(-1, 3)         # [2,3]