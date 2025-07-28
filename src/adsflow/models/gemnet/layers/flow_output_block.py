"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

#----------------------------------------------------------------------------------
# Modifications by Dean M. Sweeney, 2025-07-24
# - Removed energy predicitons from output block 
# - Replaced force prediciton with velocity predicitions (variable change)
#----------------------------------------------------------------------------------

from __future__ import annotations

import torch

from fairchem.core.modules.scaling import ScaleFactor

from fairchem.core.models.gemnet_oc.layers.base_layers import Dense, ResidualLayer

class FlowOutputBlock(torch.nn.Module):
    """
    Combines the atom update block and subsequent final dense layer.

    Arguments
    ---------
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_rbf: int
        Embedding size of the radial basis.
    nHidden: int
        Number of residual blocks before adding the atom embedding.
    activation: str
        Name of the activation function to use in the dense layers.

    """

    def __init__(
        self,
        emb_size_edge: int,
        emb_size_rbf: int,
        nHidden: int,
        activation: str | None = None,
    ) -> None:
        super().__init__() 

        self.scale_rbf_v = ScaleFactor()
        self.seq_velocity = self.get_mlp(
            emb_size_edge, emb_size_edge, nHidden, activation
        )
        self.dense_rbf_v = Dense(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )
        
    def get_mlp(self, units_in: int, units: int, nHidden: int, activation):
        if units_in != units:
            dense1 = Dense(units_in, units, activation=activation, bias=False)
            mlp = [dense1]
        else:
            mlp = []
        res = [
            ResidualLayer(units, nLayers=2, activation=activation)
            for _ in range(nHidden)
        ]
        mlp += res
        return torch.nn.ModuleList(mlp)
    
    def forward(self, h: torch.Tensor, m: torch.Tensor, basis_rad, idx_atom):
        """
        Returns
        -------
        torch.Tensor, shape=(nAtoms, emb_size_atom)
            Output atom embeddings.
        torch.Tensor, shape=(nEdges, emb_size_edge)
            Output edge embeddings.
        """
        nAtoms = h.shape[0]

        # ------------------------- Edge embeddings ------------------------ #
        x_v = m
        for _, layer in enumerate(self.seq_velocity):
            x_v = layer(x_v)  # (nEdges, emb_size_edge)

        basis_emb_v = self.dense_rbf_v(basis_rad)
        # (nEdges, emb_size_edge)
        x_v_basis = x_v * basis_emb_v
        x_v = self.scale_rbf_v(x_v_basis, ref=x_v)
        # ------------------------------------------------------------------ #

        return x_v
