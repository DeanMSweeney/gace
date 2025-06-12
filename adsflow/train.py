import torch
from torch import Tensor, nn, optim
import numpy as np
from models.gemnet.gemnet_flow import GemNetOC
from utils.atoms_to_graphs import AtomsToGraphs


"""



# Initialize model and optimizer
model = FlowGNN(node_dim=3, edge_dim=1, hidden_dim=64)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def train(model, train_loader, val_loader, epochs=1000):
    model.train()
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Sample noise and time
            x0 = torch.randn_like(batch.pos)  # Noise
            x1 = batch.pos                    # Ground truth
            t = torch.rand(len(x1), 1)        # Random time in [0, 1]
            
            # Flow matching loss
            xt = (1 - t) * x0 + t * x1
            dx_dt = x1 - x0
            u_theta = model(xt, t, batch.edge_index, batch.distances.view(-1, 1))
            loss = loss_fn(u_theta, dx_dt)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation (optional)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x0 = torch.randn_like(batch.pos)
                x1 = batch.pos
                t = torch.rand(len(x1), 1)
                xt = (1 - t) * x0 + t * x1
                dx_dt = x1 - x0
                u_theta = model(xt, t, batch.edge_index, batch.distances.view(-1, 1))
                val_loss += loss_fn(u_theta, dx_dt).item()
        model.train()
        
        # Print progress
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss / len(train_loader):.6f} | "
                f"Val Loss: {val_loss / len(val_loader):.6f}"
            )

"""