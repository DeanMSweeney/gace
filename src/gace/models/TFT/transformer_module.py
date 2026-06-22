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
    ConditionedModernTransformer,
    ConditionedModernTransformerDecoderBlock,
    ModernTransformer,
    ModernTransformerDecoderBlock,
    Transformer,
)
from gace.utils.tft.pylogger import RankedLogger
from gace.utils.tft.typing_utils import Bool, Float, Int, typecheck

log = RankedLogger(__name__, rank_zero_only=True)


class TransformerModule(nn.Module):
    """Basic Transformer model for molecule and material representation learning.

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
        use_adaln_conditioning: bool = False,
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
        # With AdaLN, conditioning is fed to every layer via `c` rather than concatenated
        # into the input, so the `conditions` slot is dropped (9->8 prim, 6->5 ads).
        self.prim_cond_dim = 8 if use_adaln_conditioning else 9   # atom_types, frac_coords, lengths, angles, T_matrix, supercell_lattice, posenc, [conditions,] type
        self.ads_cond_dim = 5 if use_adaln_conditioning else 6    # ads_atom_types, ads_pos, ads_ref_centered_pos, posenc, [conditions,] type
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
        self.use_adaln_conditioning = use_adaln_conditioning

        # Embeddings for each the primitive cell
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)  # for mask token; shared with adsorbate
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
        self.null_binding_energy = nn.Parameter(torch.zeros(hidden_dim))

        # AdaLN condition vector: nonlinearly mixes time + binding energy into `c` before
        # it drives the per-layer modulation. Only built when AdaLN conditioning is on.
        if use_adaln_conditioning:
            self.cond_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(inplace=False),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if activation == "SiLU":
            activation = nn.SiLU(inplace=False)
        elif activation == "ReLU":
            activation = nn.ReLU(inplace=False)
        elif activation == "SwiGLU":
            activation = SwiGLU()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        _trunk_cls = (
            ConditionedModernTransformer if use_adaln_conditioning else ModernTransformer
        )
        transformer_class = (
            partial(
                _trunk_cls,
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
        
        # ── Prim output heads ────────────────────────────────────────────────
        # Output head predicts real atom types only (0–98); mask token (99) is input-only.
        self.out_atom_types = nn.Sequential(
            nn.LayerNorm(hidden_dim),  # normalize head input (matches CatFlow/clari and the other heads); prevents logit saturation under AdaLN
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
                _decoder_cls = (
                    ConditionedModernTransformerDecoderBlock
                    if use_adaln_conditioning
                    else ModernTransformerDecoderBlock
                )
                decoder_layer = partial(
                    _decoder_cls,
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
            ``(pred_prim, pred_ads)``
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
        embed_atom_types = self.atom_type_embed(atom_types)  # (B, M) long → (B, M, C)
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

        embed_binding_energy = self.ads_binding_energy_embed(
            feats["ads_binding_energy"].unsqueeze(-1)
        )  # (B, C)

        # CFG training dropout: replace dropped samples with the learnable null embedding.
        # cfg_drop_mask is a (B,) bool tensor; True = this sample's condition is dropped.
        # Only present during training steps (injected by LightFlowMatch.forward).
        drop_mask = feats.get("cfg_drop_mask")
        if drop_mask is not None:
            embed_binding_energy = torch.where(
                drop_mask.unsqueeze(-1), self.null_binding_energy, embed_binding_energy
            )

        if self.use_adaln_conditioning:
            # AdaLN: conditioning is threaded into every layer via `c`, not added to the input.
            c = self.cond_proj(embed_time + embed_binding_energy)  # (B, C)
            embed_conditions = None
        else:
            # Additive: conditioning is summed into the input tokens once.
            c = None
            embed_conditions = (embed_time + embed_binding_energy).unsqueeze(-2)  # (B, 1, C)

        # ── Build h_in for prim sequence ─────────────────────────────────────
        if self.concat_combine_input:
            # Broadcast graph-level embeddings to match prim seq length
            embed_lengths_scaled_bc = embed_lengths_scaled.expand(-1, prim_seq_len, -1)
            embed_angles_radians_bc = embed_angles_radians.expand(-1, prim_seq_len, -1)
            embed_transformation_matrix_bc = embed_transformation_matrix.expand(-1, prim_seq_len, -1)
            embed_supercell_lattice_bc = embed_supercell_lattice.expand(-1, prim_seq_len, -1)

            prim_inputs = [
                embed_atom_types,               # (B, M,  C)
                embed_frac_coords,              # (B, M,  C)
                embed_lengths_scaled_bc,        # (B, M,  C)
                embed_angles_radians_bc,        # (B, M,  C)
                embed_transformation_matrix_bc, # (B, M,  C)
                embed_supercell_lattice_bc,     # (B, M,  C)
                embed_posenc_prim,              # (B, M,  C)
            ]
            if not self.use_adaln_conditioning:
                prim_inputs.append(embed_conditions.expand(-1, prim_seq_len, -1))  # (B, M, C)
            prim_inputs.append(embed_prim_type_ids)  # (B, M, C)
            h_in_prim = self.combine_input_prim(torch.cat(prim_inputs, dim=-1))  # (B, M, C)
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
            if not self.use_adaln_conditioning:
                h_in_prim = h_in_prim + embed_conditions  # (B, 1, C) → broadcasts
        h_in_prim = h_in_prim * prim_real_mask.unsqueeze(-1)  # (B, M, C)

        # ── Build h_in for ads sequence ──────────────────────────────────────
        if self.concat_combine_input:
            ads_inputs = [
                embed_ads_atom_types,           # (B, N, C)
                embed_ads_pos,                  # (B, N, C)
                embed_ads_ref_centered_pos,     # (B, N, C)
                embed_posenc_ads,               # (B, N, C)
            ]
            if not self.use_adaln_conditioning:
                ads_inputs.append(embed_conditions.expand(-1, ads_seq_len, -1))  # (B, N, C)
            ads_inputs.append(embed_ads_type_ids)  # (B, N, C)
            h_in_ads = self.combine_input_ads(torch.cat(ads_inputs, dim=-1))  # (B, N, C)
        else:
            h_in_ads = (
                embed_ads_atom_types            # (B, N, C)
                + embed_ads_pos                 # (B, N, C)
                + embed_ads_ref_centered_pos    # (B, N, C)
                + embed_posenc_ads              # (B, N, C)
                + embed_ads_type_ids            # (B, N, C)
            )
            if not self.use_adaln_conditioning:
                h_in_ads = h_in_ads + embed_conditions  # (B, 1, C) → broadcasts
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
            if not self.disable_prenorm:
                h_in = self.transformer_norm(h_in)
            if self.use_adaln_conditioning:
                h_out = self.transformer(h_in, c, padding_mask=padding_mask, **attention_kwargs)
            else:
                h_out = self.transformer(h_in, padding_mask=padding_mask, **attention_kwargs)
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")

        h_out = h_out * combined_real_mask.unsqueeze(-1)  # (B, M+N,  C)

        # ── Split transformer output back into prim / ads ────────────────────
        h_out_prim = h_out[:, :prim_seq_len, :]  # (B, M, C)
        h_out_ads = h_out[:, prim_seq_len:, :]   # (B, N,  C)

        # ── Cross-attention & output heads ────────────────────────────────────
        def _decode(block, query, **kw):
            # Conditioned decoder blocks take `c` as the second positional arg; the
            # baseline blocks do not. This keeps the six head call-sites identical.
            if self.use_adaln_conditioning:
                return block(query, c, **kw)
            return block(query, **kw)

        if self.cross_attention:
            # Prim cross-attention (query = prim output, memory = full sequence)
            h_atom = _decode(
                self.atom_types_cross_attention,
                h_out_prim,
                memory=h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_frac_coords = _decode(
                self.frac_coords_cross_attention,
                h_out_prim,
                memory=h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_lengths_scaled = _decode(
                self.lengths_scaled_cross_attention,
                h_out_prim,
                memory=h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_angles_radians = _decode(
                self.angles_radians_cross_attention,
                h_out_prim,
                memory=h_in,
                tgt_key_padding_mask=prim_padding_mask,
                memory_key_padding_mask=padding_mask,
                **attention_kwargs,
            )
            h_transformation_matrix = _decode(
                self.transformation_matrix_cross_attention,
                h_out_prim,
                memory=h_in,
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
            h_pos = _decode(
                self.pos_cross_attention,
                h_out_ads,
                memory=h_in,
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

        return pred_prim, pred_ads


