from ase.io import read
from ase import Atoms
from ase.visualize import view

import numpy as np
<<<<<<< HEAD:adsflow/scratch/scratch.py
=======

>>>>>>> e45138ff87540d2e0d736b1905944637b7703a8c:adsflow/scratch/play/scratch.py
import torch 
from torch import optim
from torch import nn
from torch_geometric.data import Data
from torch import Tensor

from adsflow.models.gemnet.gemnet_flow import GemNetFlow
from adsflow.models.gemnet.gemnet_base import GemNetOC
from adsflow.models import baseline
from ase.io import read
print("Imported packages") 

'''
Read atomic positions and convert to torch objects 
'''
contcar = read("POSCAR.vasp")
poscar = read("POSCAR0.vasp")
print("Atomic position read")

positions = torch.tensor(contcar.get_positions(), dtype=torch.float32)
atomic_numbers = torch.tensor(contcar.get_atomic_numbers(), dtype=torch.long)

"""
Compile GemNet-OC for flow matching 
"""
# GemNet-OC params
model_cfg = {
    "num_targets": 1,
    "num_spherical": 4,
    "num_radial": 8,
    "num_blocks": 1,
    "emb_size_atom": 32,
    "emb_size_edge": 64,
    "emb_size_trip_in": 64,
    "emb_size_trip_out": 64,
    "emb_size_quad_in": 32,
    "emb_size_quad_out": 32,
    "emb_size_aint_in": 64,
    "emb_size_aint_out": 64,
    "emb_size_rbf": 16,
    "emb_size_cbf": 16,
    "emb_size_sbf": 32,
    "num_before_skip": 2,
    "num_after_skip": 2,
    "num_concat": 1,
    "num_atom": 3,
    "num_output_afteratom": 3,
    "num_atom_emb_layers": 0,
    "num_global_out_layers": 2,
    "flow_matching": True,
    "dummy": True,
    "use_pbc": True,
    "scale_backprop_forces": False,
    "cutoff": 12.0,
    "cutoff_qint": 12.0,
    "cutoff_aeaint": 12.0,
    "cutoff_aint": 12.0,
    "max_neighbors": 30,
    "max_neighbors_qint": 8,
    "max_neighbors_aeaint": 20,
    "max_neighbors_aint": 1000,
    "enforce_max_neighbors_strictly": True,
    "rbf": {"name": "gaussian"},
    "rbf_spherical": None,
    "envelope": {"name": "polynomial", "exponent": 5},
    "cbf": {"name": "spherical_harmonics"},
    "sbf": {"name": "legendre_outer"},
    "extensive": True,
    "forces_coupled": True,
    "output_init": "HeOrthogonal",
    "activation": "silu",
    "scale_file": None,
    "quad_interaction": True,
    "atom_edge_interaction": True,
    "edge_atom_interaction": True,
    "atom_interaction": True,
    "scale_basis": True,
}

GemFlow = GemNetFlow(**model_cfg)
print("Compiled model")

optimizer = optim.Adam(GemFlow.parameters(), 1e-3)
loss_fn = nn.MSELoss()

# Training loop
def train():
    '''
    Defines the training procedure for the neural network
    '''
<<<<<<< HEAD:adsflow/scratch/scratch.py
    l = []
    for epoch in range(1): 
    
=======
    
    for epoch in range(10000000): 
        l = [] 
>>>>>>> e45138ff87540d2e0d736b1905944637b7703a8c:adsflow/scratch/play/scratch.py
        x1 = positions 
        x0 = torch.randn_like(x1) 
        t = torch.rand(1)

        xt = (1 - t) * x0 + t * x1
        
        batch = torch.zeros(len(xt), dtype=torch.long)
        
        # Build data objects
        data = Data(pos=xt, atomic_numbers=atomic_numbers, batch=batch)
        data.cell = torch.tensor(contcar.get_cell().array, dtype=torch.float32).unsqueeze(0) 
        data.pbc = torch.tensor([[True, True, True]])
        data.natoms = torch.tensor([len(positions)], dtype=torch.long)
        data.t = t
        data.tags = torch.zeros(len(positions), dtype=torch.long)

        # Compute u_theta 
        u_theta = GemFlow(data)
    
        # Compute loss 
        loss = loss_fn(u_theta, x1 - x0)
        l.append(loss.item())
<<<<<<< HEAD:adsflow/scratch/scratch.py
        l_avg = np.mean(np.array(l[-50:]))
        print(f"Loss at epoch {epoch}: {loss.item()} & avg. loss: {l_avg}")
        

=======
        l_avg = np.mean(np.array(l[-50:])) # compute moving average
        
>>>>>>> e45138ff87540d2e0d736b1905944637b7703a8c:adsflow/scratch/play/scratch.py
        if loss.item() < 0.01:
            break 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

<<<<<<< HEAD:adsflow/scratch/scratch.py
       
=======
        print(f"Avg loss at epoch {epoch}: {l_avg}")
>>>>>>> e45138ff87540d2e0d736b1905944637b7703a8c:adsflow/scratch/play/scratch.py

print("Begin model training")
train()
