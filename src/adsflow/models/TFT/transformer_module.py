"""Adapted from https://github.com/carlosinator/tabasco."""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple


import torch
import torch.nn as nn
from torch import Tensor

from .architectures.common import SwiGLU
from .architectures.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from .architectures.transformer import (
    ModernTransformer,
    ModernTransformerDecoderBlock,
    Transformer,
)
from adsflow.utils.tft.pylogger import RankedLogger
from adsflow.utils.tft.typing_utils import Bool, Float, Int, typecheck

log = RankedLogger(__name__, rank_zero_only=True)


class TransformerModule(nn.Module):
    """Basic Transformer model for molecule and material representation learning.

    Args:
        spatial_dim: Dimension of spatial coordinates (e.g., 3 for 3D).
        atom_dim: Number of atom types.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        num_aux_layers: Number of auxiliary transformer layers.
        num_aux_mlip_layers: Number of auxiliary energy and force prediction transformer layers.
        aux_hidden_dim: Dimension of auxiliary transformer layers.
        aux_mlip_hidden_dim: Dimension of auxiliary energy and force prediction transformer layers.
        aux_layer: Layer at which to extract representations for auxiliary tasks.
        hidden_dim: Dimension of the hidden layers.
        num_properties: Number of global properties to predict.
        dataset_embedder: The dataset embedder module.
        spacegroup_embedder: The spacegroup embedder module.
        activation: Activation function to use ("SiLU", "ReLU", "SwiGLU").
        implementation: Implementation type ("transformer", "modern_transformer").
        context_length: Maximum context length for positional encoding.
        rope_base: Base for rotary positional encoding.
        qk_layernorm: Whether to apply layer normalization to query and key in attention.
        use_sdpa: Whether to use scaled dot-product attention.
        jvp_attn: Whether to use JVP attention.
        cross_attention: Whether to use cross-attention layers.
        add_sinusoid_posenc: Whether to add sinusoidal positional encoding.
        concat_combine_input: Whether to concatenate and combine inputs.
        is_conservative: Whether to enforce conservative atomic forces as negative gradients of global energy.
        sum_pool_energy: Whether to sum-pool atom representations for energy prediction.
        separate_energy_force_transformers: Whether to use separate transformers for energy and force predictions.
        mask_material_coords: Whether or not material coordinates are masked in forward().
        custom_weight_init: Custom weight initialization method (None, "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal").
            NOTE: "uniform" does not work well.
    """

    @typecheck
    def __init__(
        self,
        spatial_dim: int,
        atom_dim: int,
        num_heads: int,
        num_layers: int,
        num_aux_layers: int,
        num_aux_mlip_layers: int,
        aux_hidden_dim: int,
        aux_mlip_hidden_dim: int,
        aux_layer: int,
        hidden_dim: int,
        num_properties: int,
        activation: Literal["SiLU", "ReLU", "SwiGLU"] = "SiLU",
        implementation: Literal["transformer", "modern_transformer"] = "modern_transformer",
        context_length: int = 2048,
        rope_base: Optional[int] = None,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
        cross_attention: bool = True,
        add_sinusoid_posenc: bool = True,
        concat_combine_input: bool = False,
        is_conservative: bool = False,
        sum_pool_energy: bool = False,
        separate_energy_force_transformers: bool = True,
        mask_material_coords: bool = True,
        num_rbf: int = 20,
        rbf_max: float = 5.0,
        custom_weight_init: Optional[
            Literal["none", "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal"]
        ] = None,
        **kwargs,
    ):
        super().__init__()

        if is_conservative and not jvp_attn:
            log.warning("JVP Flash Attention is recommended when predicting conservative forces.")

        # Coerce `custom_weight_init` to `None` if it is passed as a string
        if isinstance(custom_weight_init, str) and custom_weight_init.lower() == "none":
            custom_weight_init = None

        self.input_dim = spatial_dim + atom_dim
        self.time_dim = 1
        self.prim_cond_dim = 9   # atom_types, frac_coords, lengths, angles, T_matrix, supercell_lattice, posenc, conditions, type
        self.ads_cond_dim = 6    # ads_atom_types, ads_pos, ads_ref_centered_pos, posenc, conditions, type
        self.comb_input_dim = self.input_dim + self.time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_aux_layers = num_aux_layers
        self.num_aux_mlip_layers = num_aux_mlip_layers
        self.implementation = implementation
        self.context_length = context_length
        self.jvp_attn = jvp_attn
        self.cross_attention = cross_attention
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.is_conservative = is_conservative
        self.sum_pool_energy = sum_pool_energy
        self.separate_energy_force_transformers = separate_energy_force_transformers
        self.mask_material_coords = mask_material_coords
        self.custom_weight_init = custom_weight_init

        # Embeddings for each the primitive cell
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)  # shared with adsorbate
        self.frac_coords_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.lengths_scaled_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.angles_radians_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.transformation_matrix_embed = nn.Linear(9, hidden_dim, bias=False) # matrix is 3 x 3 = 9 total values
        self.supercell_lattice_embed = nn.Linear(9, hidden_dim, bias=False)      # S @ L, also 3 x 3 = 9

        # Embeddings for the adsorbate
        self.pos_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)       # absolute positions (B, N,  3) → (B, N,  C)
        self.ads_ref_centered_pos_embed = nn.Linear(3, hidden_dim, bias=False)  # (B, N, 3) → (B, N, C)

        # type embedding: 0 -> slab, 1 -> adsorbate
        self.type_embed = nn.Embedding(2, hidden_dim)

        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=context_length
            )

        if self.concat_combine_input:
            self.combine_input_prim = nn.Linear(self.prim_cond_dim * hidden_dim, hidden_dim)
            self.combine_input_ads = nn.Linear(self.ads_cond_dim * hidden_dim, hidden_dim)

        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)
        self.ads_binding_energy_embed = nn.Linear(1, hidden_dim)

        if activation == "SiLU":
            activation = nn.SiLU(inplace=False)
        elif activation == "ReLU":
            activation = nn.ReLU(inplace=False)
        elif activation == "SwiGLU":
            activation = SwiGLU()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        transformer_class = (
            partial(
                ModernTransformer,
                context_length=context_length,
                rope_base=rope_base,
                qk_layernorm=qk_layernorm,
                use_sdpa=use_sdpa,
                jvp_attn=jvp_attn,
            )
            if implementation == "modern_transformer"
            else Transformer
        )
        if self.implementation in ("transformer", "modern_transformer"):
            self.transformer_norm = (
                nn.RMSNorm(hidden_dim)
                if self.implementation == "modern_transformer"
                else nn.Identity()
            )
            self.transformer = transformer_class(
                dim=hidden_dim,
                depth=num_layers,
                num_heads=num_heads,
                repr_layer=aux_layer,
            )
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")
        
        # ── Prim output heads ────────────────────────────────────────────────
        self.out_atom_types = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )
        self.out_frac_coords = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_lengths_scaled = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_angles_radians = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        self.out_transformation_matrix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 9, bias=False),
        )

        # ── Ads output head (atom types are conditioning — not predicted) ────
        self.out_pos = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )

        # ── Cross attention layers ───────────────────────────────────────────
        if self.cross_attention:
            decoder_layer = partial(
                nn.TransformerDecoderLayer,
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )
            if self.implementation == "modern_transformer":
                decoder_layer = partial(
                    ModernTransformerDecoderBlock,
                    dim=hidden_dim,
                    n_heads=num_heads,
                    context_length=context_length,
                    rope_base=rope_base,
                    qk_layernorm=qk_layernorm,
                    use_sdpa=use_sdpa,
                    jvp_attn=jvp_attn,
                )

            # Prim cross-attention
            self.atom_types_cross_attention = decoder_layer()
            self.frac_coords_cross_attention = decoder_layer()
            self.lengths_scaled_cross_attention = decoder_layer()
            self.angles_radians_cross_attention = decoder_layer()
            self.transformation_matrix_cross_attention = decoder_layer()

            # Ads cross-attention (atom types are conditioning — no cross-attn head)
            self.pos_cross_attention = decoder_layer()

        # Add auxiliary task heads
        self.global_property_proj = (
            nn.Linear(hidden_dim, aux_hidden_dim, bias=False)
            if hidden_dim != aux_hidden_dim
            else nn.Identity()
        )
        self.global_energy_proj = (
            nn.Linear(hidden_dim, aux_mlip_hidden_dim, bias=False)
            if hidden_dim != aux_mlip_hidden_dim
            else nn.Identity()
        )
        self.atomic_forces_proj = (
            nn.Linear(hidden_dim, aux_mlip_hidden_dim, bias=False)
            if hidden_dim != aux_mlip_hidden_dim and self.separate_energy_force_transformers
            else nn.Identity()
        )

        if self.num_aux_layers > 0:
            if self.cross_attention:
                self.global_property_cross_attention = decoder_layer()

            self.global_property_transformer = transformer_class(
                dim=aux_hidden_dim,
                num_heads=num_heads,
                depth=num_aux_layers,
            )

        if self.num_aux_mlip_layers > 0:
            if self.cross_attention:
                self.global_energy_cross_attention = decoder_layer()
                if self.separate_energy_force_transformers:
                    self.atomic_forces_cross_attention = decoder_layer()

            self.global_energy_transformer = transformer_class(
                dim=aux_mlip_hidden_dim,
                num_heads=num_heads,
                depth=num_aux_mlip_layers,
            )
            if self.separate_energy_force_transformers:
                self.atomic_forces_transformer = transformer_class(
                    dim=aux_mlip_hidden_dim,
                    num_heads=num_heads,
                    depth=num_aux_mlip_layers,
                )

        self.global_property_head = nn.Linear(aux_hidden_dim, num_properties, bias=True)
        self.global_energy_head = nn.Linear(aux_mlip_hidden_dim, 1, bias=True)
        if not self.is_conservative:
            self.atomic_forces_head = nn.Linear(aux_mlip_hidden_dim, 3, bias=False)

        self.auxiliary_tasks = ["global_property", "global_energy", "atomic_forces"]

        # Initialize weights
        if self.custom_weight_init is not None:
            log.info(f"Initializing weights via {self.custom_weight_init} method.")
            self.apply(self._custom_weight_init)

    @typecheck
    def _custom_weight_init(self, module: nn.Module):
        """Initialize the weights of the module with a custom method.

        Args:
            module: The module to initialize.
        """
        for name, param in module.named_parameters():
            if "weight" in name and param.data.dim() >= 2:
                if self.custom_weight_init == "xavier":
                    nn.init.xavier_uniform_(param)
                elif self.custom_weight_init == "kaiming":
                    nn.init.kaiming_uniform_(param)
                elif self.custom_weight_init == "orthogonal":
                    nn.init.orthogonal_(param)
                elif self.custom_weight_init == "uniform":
                    nn.init.uniform_(param)
                elif self.custom_weight_init == "eye":
                    nn.init.eye_(param)
                elif self.custom_weight_init == "normal":
                    nn.init.normal_(param)
                else:
                    raise ValueError(f"Invalid custom weight init: {self.custom_weight_init}")

    @typecheck
    def forward(
        self,
        x_prim: (
            Tuple[
                Float["b m v"],  # type: ignore - atom_types (soft one-hot simplex)
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
                Float["b 3 3"],  # type: ignore - transformation_matrix
                Float["b 3 3"],  # type: ignore - supercell_lattice (S @ L)
            ]
            | List[torch.Tensor]
        ),
        x_ads: (
            Tuple[
                Int["b n"],      # type: ignore - ads_atom_types (integer indices, conditioning)
                Float["b n 3"],  # type: ignore - ads_pos (absolute Cartesian positions)
                Float["b n 3"],  # type: ignore - ads_ref_centered_pos
            ]
            | List[torch.Tensor]
        ),
        t_prim: (
            Tuple[
                Float[" b"],  # type: ignore - atom_types_t
                Float[" b"],  # type: ignore - frac_coords_t
                Float[" b"],  # type: ignore - lengths_scaled_t
                Float[" b"],  # type: ignore - angles_radians_t
            ]
            | List[torch.Tensor]
        ),
        t_ads: (
            Tuple[
                Float[" b"],  # type: ignore - ads_pos_t
            ]
            | List[torch.Tensor]
        ),
        feats: Dict[str, Tensor],
        prim_padding_mask: Bool["b m"],  # type: ignore
        ads_padding_mask: Bool["b n"],  # type: ignore
        **kwargs: Any,
    ) -> Tuple[
        Tuple[
            Float["b m v"],  # type: ignore - atom_types
            Float["b m 3"],  # type: ignore - frac_coords
            Float["b 1 3"],  # type: ignore - lengths_scaled
            Float["b 1 3"],  # type: ignore - angles_radians
            Float["b 3 3"],  # type: ignore - transformation_matrix
        ],
        Tuple[
            Float["b n 3"],  # type: ignore - ads_pos
        ],
        Tuple[
            Float["b 1 p"],  # type: ignore - global_property
            Float["b 1 1"],  # type: ignore - global_energy
            Float["b s 3"],  # type: ignore - atomic_forces (s = m + n)
        ],
    ]:
        """Forward pass of the module.

        Primitive-cell and adsorbate atoms are embedded as **separate sequences**,
        concatenated along the token dimension for the shared transformer trunk,
        then split back for modality-specific cross-attention and output heads.

        Args:
            x_prim: Prim modalities (atom_types, frac_coords, lengths, angles, T-matrix).
            x_ads: Ads modalities (ads_atom_types, ads_pos).
            t_prim: Per-modality flow times for prim (atom_types_t, frac_coords_t,
                lengths_t, angles_t).
            t_ads: Per-modality flow times for ads (ads_atom_types_t, ads_pos_t).
            feats: Graph-level conditioning features.
            prim_padding_mask: True = padding for prim tokens (B, M).
            ads_padding_mask: True = padding for ads tokens (B, N).

        Returns:
            ``(pred_prim, pred_ads, pred_aux)``
        """

        # primitive cell data at time t
        atom_types, frac_coords, lengths_scaled, angles_radians, transformation_matrix, supercell_lattice = x_prim
        atom_types_t, frac_coords_t, lengths_scaled_t, angles_radians_t = t_prim


        # adsorbate data at time t
        ads_atom_types, ads_pos, ads_ref_centered_pos = x_ads
        ads_pos_t, = t_ads

        device = prim_padding_mask.device
        batch_size, prim_seq_len = prim_padding_mask.shape
        _, ads_seq_len = ads_padding_mask.shape
        total_seq_len = prim_seq_len + ads_seq_len

        # Create type embeddings: 0 = slab, 1 = adsorbate
        prim_type_ids = torch.zeros(batch_size, prim_seq_len, device=device, dtype=torch.long)  # (B, M)
        ads_type_ids  = torch.ones(batch_size, ads_seq_len,  device=device, dtype=torch.long)   # (B, N)
        
        # ── Prim real-token mask ─────────────────────────────────────────────
        prim_real_mask = 1 - prim_padding_mask.int()  # (B, M)
        ads_real_mask = 1 - ads_padding_mask.int()     # (B, N)
        combined_real_mask = torch.cat([prim_real_mask, ads_real_mask], dim=1)  # (B, M+N)
        global_mask = combined_real_mask.any(-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)

        # ── Prim embeddings ──────────────────────────────────────────────────
        embed_atom_types = self.atom_type_embed(atom_types.argmax(dim=-1))  # (B, M,  C)
        embed_frac_coords = self.frac_coords_embed(frac_coords)             # (B, M,  C)
        embed_lengths_scaled = self.lengths_scaled_embed(lengths_scaled)    # (B, 1, C)
        embed_angles_radians = self.angles_radians_embed(angles_radians)    # (B, 1, C)
        # transformation_matrix: (B, 3, 3) → flatten to (B, 9) → embed → (B, C) → unsqueeze → (B, 1, C)
        embed_transformation_matrix = self.transformation_matrix_embed(
            transformation_matrix.reshape(batch_size, 9)
        ).unsqueeze(1)  # (B, 1, C)
        # supercell_lattice: S @ L, same approach
        embed_supercell_lattice = self.supercell_lattice_embed(
            supercell_lattice.reshape(batch_size, 9)
        ).unsqueeze(1)  # (B, 1, C)

        # ── Ads embeddings ───────────────────────────────────────────────────
        embed_ads_atom_types = self.atom_type_embed(ads_atom_types)                    # (B, N, C) — shared table
        embed_ads_pos = self.pos_embed(ads_pos)                                      # (B, N, C)
        embed_ads_ref_centered_pos = self.ads_ref_centered_pos_embed(ads_ref_centered_pos)  # (B, N, C)

        # ── Positional encoding (over combined seq) ──────────────────────────
        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(batch_size=batch_size, seq_len=total_seq_len)
        else:
            embed_posenc = torch.zeros(batch_size, total_seq_len, self.hidden_dim, device=device)
        embed_posenc_prim = embed_posenc[:, :prim_seq_len, :]  # (B, M, C)
        embed_posenc_ads = embed_posenc[:, prim_seq_len:, :]   # (B, N,  C)

        # Embed type ids (prim = 0, ads = 1)
        embed_prim_type_ids = self.type_embed(prim_type_ids)  # (B, M, C)
        embed_ads_type_ids  = self.type_embed(ads_type_ids)   # (B, N, C)

        # ── Time embeddings (average prim + ads times) ───────────────────────
        prim_modals_t = torch.cat(
            [t_.unsqueeze(-1) for t_ in [atom_types_t, frac_coords_t, lengths_scaled_t, angles_radians_t]],
            dim=-1,
        )
        ads_modals_t = ads_pos_t.unsqueeze(-1)  # (B, 1)
        all_modals_t = torch.cat([prim_modals_t, ads_modals_t], dim=-1)  # (B, num_modals)
        embed_time = (
            self.time_encoding(all_modals_t.reshape(-1))
            .reshape(batch_size, all_modals_t.shape[1], -1)
            .mean(-2)
        )  # (B, C)

        if feats.get("ads_binding_energy") is not None:
            embed_binding_energy = self.ads_binding_energy_embed(
                feats["ads_binding_energy"].unsqueeze(-1)
            )       
        else:
            embed_binding_energy = torch.zeros(batch_size, self.hidden_dim, device=device)
                                                                                          # (B, C)
        embed_conditions = (embed_time + embed_binding_energy).unsqueeze(-2)  # (B, 1, C)

        # ── Build h_in for prim sequence ─────────────────────────────────────
        if self.concat_combine_input:
            # Broadcast graph-level embeddings to match prim seq length
            embed_lengths_scaled_bc = embed_lengths_scaled.expand(-1, prim_seq_len, -1)
            embed_angles_radians_bc = embed_angles_radians.expand(-1, prim_seq_len, -1)
            embed_transformation_matrix_bc = embed_transformation_matrix.expand(-1, prim_seq_len, -1)
            embed_supercell_lattice_bc = embed_supercell_lattice.expand(-1, prim_seq_len, -1)
            embed_conditions_prim = embed_conditions.expand(-1, prim_seq_len, -1)

            h_in_prim = torch.cat([
                embed_atom_types,               # (B, M,  C)
                embed_frac_coords,              # (B, M,  C)
                embed_lengths_scaled_bc,        # (B, M,  C)
                embed_angles_radians_bc,        # (B, M,  C)
                embed_transformation_matrix_bc, # (B, M,  C)
                embed_supercell_lattice_bc,     # (B, M,  C)
                embed_posenc_prim,              # (B, M,  C)
                embed_conditions_prim,          # (B, M,  C)
                embed_prim_type_ids,            # (B, M,  C)
            ], dim=-1)  # (B, M,  9*C)
            h_in_prim = self.combine_input_prim(h_in_prim)  # (B, M, C)
        else:
            h_in_prim = (
                embed_atom_types              # (B, M,  C)
                + embed_frac_coords           # (B, M,  C)
                + embed_lengths_scaled        # (B, 1, C) → broadcasts
                + embed_angles_radians        # (B, 1, C) → broadcasts
                + embed_transformation_matrix # (B, 1, C) → broadcasts
                + embed_supercell_lattice     # (B, 1, C) → broadcasts
                + embed_posenc_prim           # (B, M,  C)
                + embed_conditions            # (B, 1, C) → broadcasts
                + embed_prim_type_ids         # (B, M,  C)
            )
        h_in_prim = h_in_prim * prim_real_mask.unsqueeze(-1)  # (B, M, C)

        # ── Build h_in for ads sequence ──────────────────────────────────────
        if self.concat_combine_input:
            embed_conditions_ads = embed_conditions.expand(-1, ads_seq_len, -1)

            h_in_ads = torch.cat([
                embed_ads_atom_types,           # (B, N, C)
                embed_ads_pos,                  # (B, N, C)
                embed_ads_ref_centered_pos,     # (B, N, C)
                embed_posenc_ads,               # (B, N, C)
                embed_conditions_ads,           # (B, N, C)
                embed_ads_type_ids,             # (B, N, C)
            ], dim=-1)  # (B, N, 6*C)
            h_in_ads = self.combine_input_ads(h_in_ads)  # (B, N,  C)
        else:
            h_in_ads = (
                embed_ads_atom_types            # (B, N, C)
                + embed_ads_pos                 # (B, N, C)
                + embed_ads_ref_centered_pos    # (B, N, C)
                + embed_posenc_ads              # (B, N, C)
                + embed_conditions              # (B, 1, C) → broadcasts
                + embed_ads_type_ids            # (B, N, C)
            )
        h_in_ads = h_in_ads * ads_real_mask.unsqueeze(-1)  # (B, N,  C)

        # ── Concatenate prim + ads into single sequence ──────────────────────
        h_in = torch.cat([h_in_prim, h_in_ads], dim=1)  # (B, M+N,  C)
        padding_mask = torch.cat([prim_padding_mask, ads_padding_mask], dim=1)  # (B, M+N)

        pos_ids = torch.arange(total_seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        ads_pos_ids = pos_ids[:, prim_seq_len:]  # true positions of ads tokens in combined sequence
        attention_kwargs = {"pos_ids": pos_ids}

        if self.implementation == "transformer":
            attention_kwargs.pop("pos_ids")

        # ── Shared transformer trunk ─────────────────────────────────────────
        if self.implementation in ("transformer", "modern_transformer"):
            h_in = self.transformer_norm(h_in)
            h_out, h_aux = self.transformer(h_in, padding_mask=padding_mask, **attention_kwargs)
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")

        h_out = h_out * combined_real_mask.unsqueeze(-1)  # (B, M+N,  C)

        # ── Split transformer output back into prim / ads ────────────────────
        h_out_prim = h_out[:, :prim_seq_len, :]  # (B, M, C)
        h_out_ads = h_out[:, prim_seq_len:, :]   # (B, N,  C)

        # ── Cross-attention & output heads ────────────────────────────────────
        if self.cross_attention:
            # Prim cross-attention (query = prim output, memory = full sequence)
            h_atom = self.atom_types_cross_attention(
                h_out_prim,
                h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_frac_coords = self.frac_coords_cross_attention(
                h_out_prim,
                h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_lengths_scaled = self.lengths_scaled_cross_attention(
                h_out_prim,
                h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_angles_radians = self.angles_radians_cross_attention(
                h_out_prim,
                h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_transformation_matrix = self.transformation_matrix_cross_attention(
                h_out_prim,
                h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )

            out_atom_types = self.out_atom_types(h_atom)
            out_frac_coords = self.out_frac_coords(h_frac_coords)
            prim_mask_f = prim_real_mask.unsqueeze(-1).float()  # (B, M, 1)
            prim_n = prim_mask_f.sum(-2, keepdim=True).clamp(min=1)
            out_lengths_scaled = self.out_lengths_scaled(
                (h_lengths_scaled * prim_mask_f).sum(-2, keepdim=True) / prim_n
            )
            out_angles_radians = self.out_angles_radians(
                (h_angles_radians * prim_mask_f).sum(-2, keepdim=True) / prim_n
            )
            out_transformation_matrix = self.out_transformation_matrix(
                (h_transformation_matrix * prim_mask_f).sum(-2) / prim_n.squeeze(-2)
            ).reshape(batch_size, 3, 3)  # (B, 9) → (B, 3, 3)

            # Ads cross-attention — single head for absolute positions
            h_pos = self.pos_cross_attention(
                h_out_ads,
                h_in,
                tgt_key_padding_mask=ads_padding_mask,
                memory_key_padding_mask=padding_mask,
                query_pos_ids=ads_pos_ids,
                **attention_kwargs,
            )
            out_pos = self.out_pos(h_pos)  # (B, N,  3)

        else:
            out_atom_types = self.out_atom_types(h_out_prim)
            out_frac_coords = self.out_frac_coords(h_out_prim)
            prim_mask_f = prim_real_mask.unsqueeze(-1).float()  # (B, M, 1)
            prim_n = prim_mask_f.sum(-2, keepdim=True).clamp(min=1)
            out_lengths_scaled = self.out_lengths_scaled(
                (h_out_prim * prim_mask_f).sum(-2, keepdim=True) / prim_n
            )
            out_angles_radians = self.out_angles_radians(
                (h_out_prim * prim_mask_f).sum(-2, keepdim=True) / prim_n
            )
            out_transformation_matrix = self.out_transformation_matrix(
                (h_out_prim * prim_mask_f).sum(-2) / prim_n.squeeze(-2)
            ).reshape(batch_size, 3, 3)  # (B, 9) → (B, 3, 3)

            out_pos = self.out_pos(h_out_ads)  # (B, N,  3)

        # ── Auxiliary tasks (operate on full combined sequence) ───────────────
        h_global_property = h_aux
        h_global_energy = h_aux
        h_atomic_forces = h_aux

        if self.num_aux_layers > 0:
            if self.cross_attention:
                h_global_property = self.global_property_cross_attention(
                    h_aux, h_in,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=padding_mask,
                    **attention_kwargs,
                )

            h_global_property = self.global_property_transformer(
                self.global_property_proj(h_global_property),
                padding_mask=padding_mask,
                **attention_kwargs,
            )
        else:
            h_global_property = self.global_property_proj(h_global_property)

        if self.num_aux_mlip_layers > 0:
            if self.cross_attention:
                h_global_energy = self.global_energy_cross_attention(
                    h_aux, h_in,
                    tgt_key_padding_mask=padding_mask,
                    memory_key_padding_mask=padding_mask,
                    **attention_kwargs,
                )
                h_atomic_forces = (
                    self.atomic_forces_cross_attention(
                        h_aux, h_in,
                        tgt_key_padding_mask=padding_mask,
                        memory_key_padding_mask=padding_mask,
                        **attention_kwargs,
                    )
                    if self.separate_energy_force_transformers
                    else h_global_energy
                )

            h_global_energy = self.global_energy_transformer(
                self.global_energy_proj(h_global_energy),
                padding_mask=padding_mask,
                **attention_kwargs,
            )
            if self.separate_energy_force_transformers:
                h_atomic_forces = self.atomic_forces_transformer(
                    self.atomic_forces_proj(h_atomic_forces),
                    padding_mask=padding_mask,
                    **attention_kwargs,
                )
            else:
                h_atomic_forces = h_global_energy
        else:
            h_global_energy = self.global_energy_proj(h_global_energy)
            h_atomic_forces = self.atomic_forces_proj(h_atomic_forces)

        combined_mask_f = combined_real_mask.unsqueeze(-1).float()  # (B, M+N,  1)
        combined_n = combined_mask_f.sum(-2, keepdim=True).clamp(min=1)
        h_global_property_pooled = (
            (h_global_property * combined_mask_f).sum(-2, keepdim=True) / combined_n
        )
        h_global_energy_pooled = (
            (h_global_energy * combined_mask_f).sum(-2, keepdim=True)
            if self.sum_pool_energy
            else (h_global_energy * combined_mask_f).sum(-2, keepdim=True) / combined_n
        )

        global_property = self.global_property_head(h_global_property_pooled) * global_mask
        global_energy = self.global_energy_head(h_global_energy_pooled) * global_mask
        atomic_forces = self.atomic_forces_head(h_atomic_forces) * combined_real_mask.unsqueeze(-1)

        # ── Assemble outputs ─────────────────────────────────────────────────
        pred_prim = (
            out_atom_types * prim_real_mask.unsqueeze(-1),    # (B, M, V)
            out_frac_coords * prim_real_mask.unsqueeze(-1),   # (B, M, 3)
            out_lengths_scaled * global_mask,                 # (B, 1, 3)
            out_angles_radians * global_mask,                 # (B, 1, 3)
            out_transformation_matrix * global_mask,          # (B, 1, 3)
        )
        pred_ads = (
            out_pos * ads_real_mask.unsqueeze(-1),   # (B, N,  3)
        )
        pred_aux_outputs = (
            global_property,   # (B, 1, P)
            global_energy,     # (B, 1, 1)
            atomic_forces,     # (B, M+N,  3)
        )

        return pred_prim, pred_ads, pred_aux_outputs

    @typecheck
    def forward_with_cfg(
        self,
        x_prim: (
            Tuple[
                Float["b m v"],  # type: ignore - atom_types (soft one-hot simplex)
                Float["b m 3"],  # type: ignore - frac_coords
                Float["b 1 3"],  # type: ignore - lengths_scaled
                Float["b 1 3"],  # type: ignore - angles_radians
                Float["b 3 3"],  # type: ignore - transformation_matrix
                Float["b 3 3"],  # type: ignore - supercell_lattice
            ]
            | List[torch.Tensor]
        ),
        x_ads: (
            Tuple[
                Int["b n"],      # type: ignore - ads_atom_types (integer indices, conditioning)
                Float["b n 3"],  # type: ignore - ads_pos (absolute Cartesian positions)
                Float["b n 3"],  # type: ignore - ads_ref_centered_pos
            ]
            | List[torch.Tensor]
        ),
        t_prim: (
            Tuple[
                Float[" b"],  # type: ignore - atom_types_t
                Float[" b"],  # type: ignore - frac_coords_t
                Float[" b"],  # type: ignore - lengths_scaled_t
                Float[" b"],  # type: ignore - angles_radians_t
            ]
            | List[torch.Tensor]
        ),
        t_ads: (
            Tuple[
                Float[" b"],  # type: ignore - ads_pos_t
            ]
            | List[torch.Tensor]
        ),
        feats: Dict[str, Tensor],
        prim_padding_mask: Bool["b m"],  # type: ignore
        ads_padding_mask: Bool["b n"],  # type: ignore
        cfg_scale: float,
        **kwargs: Any,
    ) -> Tuple[
        Tuple[
            Float["b m v"],  # type: ignore - atom_types
            Float["b m 3"],  # type: ignore - frac_coords
            Float["b 1 3"],  # type: ignore - lengths_scaled
            Float["b 1 3"],  # type: ignore - angles_radians
            Float["b 1 9"],  # type: ignore - transformation_matrix
        ],
        Tuple[
            Float["b n v"],  # type: ignore - ads_atom_types
            Float["b 1 3"],  # type: ignore - ads_com
            Float["b n 3"],  # type: ignore - ads_rel_pos
        ],
        Tuple[
            Float["b 1 p"],  # type: ignore - global_property
            Float["b 1 1"],  # type: ignore - global_energy
            Float["b s 3"],  # type: ignore - atomic_forces (s = m + n)
        ],
    ]:
        """Classifier-free guidance forward pass.

        Doubles the batch (conditional + unconditional) and interpolates
        outputs with ``cfg_scale``.
        """
        half_prim = tuple(x_[: len(x_) // 2] for x_ in x_prim)
        combined_prim = tuple(torch.cat([h, h], dim=0) for h in half_prim)

        half_ads = tuple(x_[: len(x_) // 2] for x_ in x_ads)
        combined_ads = tuple(torch.cat([h, h], dim=0) for h in half_ads)

        pred_prim, pred_ads, pred_aux = self.forward(
            combined_prim,
            combined_ads,
            t_prim,
            t_ads,
            feats,
            prim_padding_mask,
            ads_padding_mask,
            **kwargs,
        )

        def _apply_cfg(outputs):
            guided = []
            for modal in outputs:
                cond, uncond = torch.split(modal, len(modal) // 2, dim=0)
                half = uncond + cfg_scale * (cond - uncond)
                guided.append(torch.cat([half, half], dim=0))
            return tuple(guided)

        return _apply_cfg(pred_prim), _apply_cfg(pred_ads), _apply_cfg(pred_aux)

