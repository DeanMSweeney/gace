import numpy as np

import torch 
from torch import optim
from torch import nn
from torch_geometric.data import Data
from torch import Tensor

from ase.io import read
from ase import Atoms
from ase.visualize import view

from collections import deque

from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.gemnet.gemnet_base import GemNetOC


def train(
    model,
    positions,
    atomic_numbers,
    cell,
    optimizer,
    loss_fn,
    num_epochs: int = 100,
    print_every: int = 10,
    tol: float = 1e-2,
    device: torch.device = torch.device("cpu"),
):
    
    model.train()
    # Move static tensors once
    x1 = positions.to(device)
    atomic_numbers = atomic_numbers.to(device)
    cell = cell.to(device)
    batch = torch.zeros(len(x1), dtype=torch.long, device=device)

    # Keep a fixed‐size buffer for running avg
    recent_losses = deque(maxlen=print_every)

    for epoch in range(num_epochs):
        # build data object 
        data = Data(
            atomic_numbers=atomic_numbers,    
            batch=batch,
            cell = cell, 
            pbc = torch.tensor([[True, True, True]]), 
            natoms = torch.tensor([len(positions)], dtype=torch.long), 
            tags = torch.zeros(len(positions), dtype=torch.long)
        )
        # interpolate between noise and sample: optimal transport
        x0 = torch.randn_like(x1, device=device)
        t  = torch.rand(1, device=device)
        xt = (1 - t) * x0 + t * x1
        data.t = t
        data.pos = xt

        optimizer.zero_grad()
        u_theta = model(data)
        target  = x1 - x0
        loss    = loss_fn(u_theta, target)
        loss.backward()
        optimizer.step()

        recent_losses.append(loss.item())
        if epoch % print_every == 0:
            avg = sum(recent_losses) / len(recent_losses)
            print(f"[{epoch:6d}] loss = {loss.item()}, avg({print_every}) = {avg}")

        if loss.item() < tol:
            print(f"Converged at epoch {epoch} (loss {loss.item()})")
            break

    return 

