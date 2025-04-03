"""
Copyright (c) Facebook, Inc. and its affiliates.

Base model class for atomistic/molecular simulations with graph-based representations.
Handles both periodic (PBC) and non-periodic systems with on-the-fly graph generation.
"""

import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

from ads_flow.utils.utils import (
    get_pbc_distances,
    radius_graph_pbc,
)

class BaseModel(nn.Module):
    """Base class for graph-based atomistic models.
    
    Args:
        num_atoms (int, optional): Maximum number of atoms in system.
        bond_feat_dim (int, optional): Dimension of bond features.
        num_targets (int, optional): Number of prediction targets (e.g., energy, forces).
    """
    def __init__(
        self, 
        num_atoms: Optional[int] = None,
        bond_feat_dim: Optional[int] = None, 
        num_targets: Optional[int] = None
    ) -> None:
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass (to be implemented by child classes)."""
        raise NotImplementedError("Child classes must implement forward()")

    def generate_graph(
        self,
        data: Data,
        cutoff: Optional[float] = None,
        max_neighbors: Optional[int] = None,
        use_pbc: Optional[bool] = None,
        otf_graph: Optional[bool] = None,
        enforce_max_neighbors_strictly: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Generates graph representation of atomic system.
        
        Args:
            data: Torch Geometric Data object with:
                - pos: Atomic positions [num_atoms, 3]
                - cell: Unit cell vectors [3, 3] (if PBC)
                - batch: Batch indices [num_atoms] (if multiple systems)
            cutoff: Maximum interatomic distance for edges.
            max_neighbors: Maximum allowed neighbors per atom.
            use_pbc: Whether to use periodic boundary conditions.
            otf_graph: Generate graph on-the-fly (else use precomputed).
            enforce_max_neighbors_strictly: Whether to enforce max_neighbors strictly.
            
        Returns:
            Tuple containing:
                - edge_index: Edge indices [2, num_edges]
                - edge_dist: Interatomic distances [num_edges]
                - distance_vec: Distance vectors [num_edges, 3]
                - cell_offsets: PBC offsets [num_edges, 3]
                - cell_offset_distances: Offset distances [num_edges, 3]
                - neighbors: Number of neighbors per atom
        """
        # Set defaults from class attributes if not provided
        cutoff = cutoff or getattr(self, 'cutoff', None)
        max_neighbors = max_neighbors or getattr(self, 'max_neighbors', None)
        use_pbc = use_pbc or getattr(self, 'use_pbc', False)
        otf_graph = otf_graph or getattr(self, 'otf_graph', False)

        # Handle neighbor enforcement policy
        if enforce_max_neighbors_strictly is None:
            enforce_max_neighbors_strictly = getattr(
                self, 'enforce_max_neighbors_strictly', True
            )

        # Check for precomputed edges if not generating on-the-fly
        if not otf_graph:
            try:
                edge_index = data.edge_index
                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors
            except AttributeError:
                logging.warning(
                    "Precomputed edges not found. Falling back to on-the-fly generation."
                )
                otf_graph = True

        # Handle periodic boundary conditions
        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data=data,
                    cutoff=cutoff,
                    max_neighbors=max_neighbors,
                    enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
                )

            # Compute distances with PBC corrections
            out: Dict[str, torch.Tensor] = get_pbc_distances(
                pos=data.pos,
                edge_index=edge_index,
                cell=data.cell,
                cell_offsets=cell_offsets,
                neighbors=neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]

        # Non-periodic case
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            # Compute distances directly (no PBC)
            j, i = edge_index  # Source and target nodes
            distance_vec = data.pos[j] - data.pos[i]
            edge_dist = distance_vec.norm(dim=-1)
            
            # Dummy values for non-PBC consistency
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(cell_offsets)
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns parameter names excluded from L2 regularization.
        
        Commonly excludes:
            - Embeddings (e.g., atom type embeddings)
            - Frequency-based parameters
            - Bias terms
        """
        no_wd_list = []
        for name, _ in self.named_parameters():
            if any(k in name for k in ["embedding", "frequencies", "bias"]):
                no_wd_list.append(name)
        return no_wd_list