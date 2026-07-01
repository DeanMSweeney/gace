"""Adapted from https://github.com/carlosinator/tabasco."""

from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
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
from gace.utils.tft.pylogger import RankedLogger
from gace.utils.tft.typing_utils import Bool, Float, Int, typecheck

log = RankedLogger(__name__, rank_zero_only=True)


class TransformerModuleEnergy(nn.Module):
    """Energy regressor built on the zatom transformer trunk.

    Maps a (slab + adsorbate) structure to a single scalar binding energy.
    Shares the input embedding stack and shared transformer trunk of
    ``TransformerModule`` but removes the generation output heads and the
    binding-energy *conditioning* input (binding energy is the target here),
    and adds a single energy regression head at the decoder layer.

    The primitive-cell atom-type embedding is a *soft* embedding over the type
    simplex (``nn.Linear(atom_dim, hidden_dim)``): a hard one-hot input is
    identical to an ``nn.Embedding`` lookup, while a softmax distribution lets
    guidance gradients flow back into the flow model's atom-type logits.

    Original (generation) docstring follows.

    Args:
        spatial_dim: Dimension of spatial coordinates (e.g., 3 for 3D).
        atom_dim: Number of atom types.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        hidden_dim: Dimension of the hidden layers.
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
        hidden_dim: int,
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
        disable_prenorm: bool = False,
        num_rbf: int = 20,
        rbf_max: float = 5.0,
        custom_weight_init: Optional[
            Literal["none", "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal"]
        ] = None,
        **kwargs,
    ):
        super().__init__()

        # Coerce `custom_weight_init` to `None` if it is passed as a string
        if isinstance(custom_weight_init, str) and custom_weight_init.lower() == "none":
            custom_weight_init = None

        self.input_dim = spatial_dim + atom_dim
        self.time_dim = 1
        self.prim_cond_dim = 8   # atom_types, frac_coords, lengths, angles, T_matrix, supercell_lattice, posenc, type
        self.ads_cond_dim = 4    # ads_atom_types, ads_pos, posenc, type
        self.comb_input_dim = self.input_dim + self.time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.implementation = implementation
        self.context_length = context_length
        self.jvp_attn = jvp_attn
        self.cross_attention = cross_attention
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.custom_weight_init = custom_weight_init
        self.disable_prenorm = disable_prenorm
        self.atom_dim = atom_dim

        # Embeddings for each the primitive cell
        # Soft embedding over the type simplex (B, *, atom_dim) -> (B, *, C).
        # A hard one-hot input equals an nn.Embedding lookup; a softmax input
        # keeps the path differentiable w.r.t. atom-type logits. Shared with ads.
        self.atom_type_embed = nn.Linear(atom_dim, hidden_dim, bias=False)
        self.frac_coords_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.lengths_scaled_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.angles_radians_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.transformation_matrix_embed = nn.Linear(9, hidden_dim, bias=False) # matrix is 3 x 3 = 9 total values
        self.supercell_lattice_embed = nn.Linear(9, hidden_dim, bias=False)      # S @ L, also 3 x 3 = 9

        # Embeddings for the adsorbate
        self.pos_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)       # absolute positions (B, N,  3) → (B, N,  C)

        # type embedding: 0 -> slab, 1 -> adsorbate
        self.type_embed = nn.Embedding(2, hidden_dim)

        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=context_length
            )

        if self.concat_combine_input:
            self.combine_input_prim = nn.Linear(self.prim_cond_dim * hidden_dim, hidden_dim)
            self.combine_input_ads = nn.Linear(self.ads_cond_dim * hidden_dim, hidden_dim)

        # NOTE: the energy regressor scores a clean structure — there is no flow
        # time conditioning and no ads_binding_energy conditioning (binding energy
        # is the regression target here), so both embeddings are removed.

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
            )
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")
        
        # ── Energy regression head ───────────────────────────────────────────
        # The trunk is dedicated to energy (no generation task), so its final
        # per-token output already carries deep, energy-specialized token mixing.
        # When cross_attention is enabled we re-read the input embeddings via one
        # cross-attention step (query = trunk output, memory = input embeddings),
        # then mean-pool over real tokens and map to a scalar with an MLP.
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

            self.energy_cross_attention = decoder_layer()

        self.out_energy = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, 1),
        )

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
            ]
            | List[torch.Tensor]
        ),
        prim_padding_mask: Bool["b m"],  # type: ignore
        ads_padding_mask: Bool["b n"],  # type: ignore
        **kwargs: Any,
    ) -> Float[" b"]:  # type: ignore - per-graph scalar energy
        """Regress a single scalar (binding) energy per structure in the batch.

        Primitive-cell and adsorbate atoms are embedded as **separate sequences**
        and concatenated along the token dimension for the shared transformer
        trunk. The energy head pools the trunk output (optionally after one
        cross-attention re-read of the input embeddings) and maps it to a scalar.
        There is no flow-time or binding-energy conditioning — the structure is
        scored as-is, and binding energy is the regression target.

        Args:
            x_prim: Prim modalities (atom_types simplex, frac_coords, lengths,
                angles, T-matrix, supercell_lattice).
            x_ads: Ads modalities (ads_atom_types indices, ads_pos).
            prim_padding_mask: True = padding for prim tokens (B, M).
            ads_padding_mask: True = padding for ads tokens (B, N).

        Returns:
            ``energy`` of shape (B,).
        """

        # primitive cell structure
        atom_types, frac_coords, lengths_scaled, angles_radians, transformation_matrix, supercell_lattice = x_prim

        # adsorbate structure
        ads_atom_types, ads_pos, = x_ads

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

        # ── Prim embeddings ──────────────────────────────────────────────────
        # atom_types is a probability simplex (B, M, V); the soft embedding
        # `simplex @ W` equals an nn.Embedding lookup when the simplex is one-hot.
        embed_atom_types = self.atom_type_embed(atom_types)                 # (B, M,  C)
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
        # ads_atom_types are integer indices (fixed conditioning); one-hot them so
        # they pass through the same soft atom-type embedding (== a lookup).
        ads_atom_types_onehot = F.one_hot(ads_atom_types.long(), self.atom_dim).to(embed_atom_types.dtype)
        embed_ads_atom_types = self.atom_type_embed(ads_atom_types_onehot)            # (B, N, C) — shared table
        embed_ads_pos = self.pos_embed(ads_pos)                                      # (B, N, C)

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

        # NOTE: no time / binding-energy conditioning — the structure is scored as-is.

        # ── Build h_in for prim sequence ─────────────────────────────────────
        if self.concat_combine_input:
            # Broadcast graph-level embeddings to match prim seq length
            embed_lengths_scaled_bc = embed_lengths_scaled.expand(-1, prim_seq_len, -1)
            embed_angles_radians_bc = embed_angles_radians.expand(-1, prim_seq_len, -1)
            embed_transformation_matrix_bc = embed_transformation_matrix.expand(-1, prim_seq_len, -1)
            embed_supercell_lattice_bc = embed_supercell_lattice.expand(-1, prim_seq_len, -1)

            h_in_prim = torch.cat([
                embed_atom_types,               # (B, M,  C)
                embed_frac_coords,              # (B, M,  C)
                embed_lengths_scaled_bc,        # (B, M,  C)
                embed_angles_radians_bc,        # (B, M,  C)
                embed_transformation_matrix_bc, # (B, M,  C)
                embed_supercell_lattice_bc,     # (B, M,  C)
                embed_posenc_prim,              # (B, M,  C)
                embed_prim_type_ids,            # (B, M,  C)
            ], dim=-1)  # (B, M,  8*C)
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
                + embed_prim_type_ids         # (B, M,  C)
            )
        h_in_prim = h_in_prim * prim_real_mask.unsqueeze(-1)  # (B, M, C)

        # ── Build h_in for ads sequence ──────────────────────────────────────
        if self.concat_combine_input:
            h_in_ads = torch.cat([
                embed_ads_atom_types,           # (B, N, C)
                embed_ads_pos,                  # (B, N, C)
                embed_posenc_ads,               # (B, N, C)
                embed_ads_type_ids,             # (B, N, C)
            ], dim=-1)  # (B, N, 5*C)
            h_in_ads = self.combine_input_ads(h_in_ads)  # (B, N,  C)
        else:
            h_in_ads = (
                embed_ads_atom_types            # (B, N, C)
                + embed_ads_pos                 # (B, N, C)
                + embed_posenc_ads              # (B, N, C)
                + embed_ads_type_ids            # (B, N, C)
            )
        h_in_ads = h_in_ads * ads_real_mask.unsqueeze(-1)  # (B, N,  C)

        # ── Concatenate prim + ads into single sequence ──────────────────────
        h_in = torch.cat([h_in_prim, h_in_ads], dim=1)  # (B, M+N,  C)
        padding_mask = torch.cat([prim_padding_mask, ads_padding_mask], dim=1)  # (B, M+N)

        pos_ids = torch.arange(total_seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        attention_kwargs = {"pos_ids": pos_ids}

        if self.implementation == "transformer":
            attention_kwargs.pop("pos_ids")

        # ── Shared transformer trunk ─────────────────────────────────────────
        if self.implementation in ("transformer", "modern_transformer"):
            if not self.disable_prenorm:
                h_in = self.transformer_norm(h_in)
            h_out = self.transformer(h_in, padding_mask=padding_mask, **attention_kwargs)
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")

        h_out = h_out * combined_real_mask.unsqueeze(-1)  # (B, M+N,  C)

        # ── Energy head: (optional) cross-attention → masked-mean pool → MLP ──
        if self.cross_attention:
            # Re-read the input embeddings: query = trunk output, memory = h_in.
            h_energy = self.energy_cross_attention(
                h_out,
                h_in,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )  # (B, M+N, C)
        else:
            h_energy = h_out

        mask_f = combined_real_mask.unsqueeze(-1).float()        # (B, M+N, 1)
        h_pooled = (h_energy * mask_f).sum(-2) / mask_f.sum(-2).clamp(min=1)  # (B, C)

        energy = self.out_energy(h_pooled).squeeze(-1)  # (B,)
        energy = energy * combined_real_mask.any(-1).float()  # zero out fully-empty graphs

        return energy


