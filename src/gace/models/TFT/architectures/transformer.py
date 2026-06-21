"""Adapted from https://github.com/carlosinator/tabasco."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import (
    AttentionBlock,
    ModernAttention,
)
from .common import SwiGLUFeedForward
from .transition import Transition
from gace.utils.tft.typing_utils import typecheck


class TransformerBlock(nn.Module):
    """A transformer block with layer normalization and residual connections.

    This implements a standard transformer block with self-attention followed by
    a feed-forward network, with layer normalization and residual connections.

    Args:
        dim: Input and output dimension
        num_heads: Number of attention heads
        mlp_dim: Hidden dimension for the feed-forward network (defaults to 4x input dim)
        dropout: Dropout probability
        activation_type: Type of activation to use in the feed-forward network
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int = None,
        dropout: float = 0.0,
        activation_type: str = "swiglu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Self-attention block
        self.attn_block = AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_eps=norm_eps,
        )

        # Feed-forward network
        self.ff_block = Transition(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            activation_type=activation_type,
            layer_norm=True,
        )

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention with residual connection
        attn_output = self.attn_block(x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + attn_output

        # Feed-forward network with residual connection
        ff_output = self.ff_block(x)
        x = x + ff_output

        return x


class ModernTransformerBlock(nn.Module):
    """A self-attention transformer block using ModernAttention and SwiGLUFeedForward.

    Args:
        dim: Input and output dimension
        n_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        self.attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        self.attention_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the modern transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            pos_ids: Position ids for rotary embeddings
                Shape: [batch_size, seq_len]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len],
                where False indicates positions to mask or the float -inf denotes masked positions

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        h = x + self.attention(
            self.attention_norm(x), pos_ids=pos_ids, padding_mask=padding_mask, attn_mask=attn_mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class ModernTransformerDecoderBlock(nn.Module):
    """A transformer decoder block using ModernAttention and SwiGLUFeedForward.

    Can be used as a drop-in replacement for `nn.TransformerDecoderLayer`.

    Args:
        dim: Input and output dimension
        n_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        self.self_attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.cross_attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        self.self_attention_norm = nn.RMSNorm(dim)
        self.cross_attention_norm = nn.RMSNorm(dim)
        self.ffn_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        query_pos_ids: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the modern transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            memory: Optional memory tensor for cross-attention
                Shape: [batch_size, seq_len, dim]
            pos_ids: Position ids for rotary embeddings (memory / combined sequence)
                Shape: [batch_size, seq_len]
            query_pos_ids: Optional override for query token positions. When provided,
                used for self-attention pos_ids and as query_pos_ids in cross-attention,
                so that query tokens whose true positions differ from the front of pos_ids
                (e.g. adsorbate tokens offset by prim_seq_len) are rotated correctly.
                Shape: [batch_size, query_seq_len]
            tgt_key_padding_mask: Boolean mask for target padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            memory_key_padding_mask: Boolean mask for memory padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len],
                where False indicates positions to mask or the float -inf denotes masked positions

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        sa_pos_ids = query_pos_ids if query_pos_ids is not None else pos_ids
        h = x + self.self_attention(
            self.self_attention_norm(x),
            pos_ids=sa_pos_ids,
            padding_mask=tgt_key_padding_mask,
            attn_mask=attn_mask,
        )
        h = h + self.cross_attention(
            self.cross_attention_norm(h),
            memory=memory,
            pos_ids=pos_ids,
            query_pos_ids=query_pos_ids,
            padding_mask=memory_key_padding_mask,
            attn_mask=attn_mask,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """A standard Transformer model with multiple layers.

    This implements a sequence of transformer blocks, each containing
    self-attention and feed-forward components with residual connections.

    Args:
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        repr_layer: Layer at which to additionally extract intermediate representations. If None, no intermediate representation is extracted.
        mlp_dim: Hidden dimension for feed-forward networks (defaults to 4x dim)
        dropout: Dropout probability
        activation_type: Type of activation to use in feed-forward networks
        norm_eps: Epsilon value for layer normalization
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        repr_layer: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation_type: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.repr_layer = repr_layer

        if mlp_dim is None:
            mlp_dim = 4 * dim

        # Create a sequence of transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    activation_type=activation_type,
                    norm_eps=norm_eps,
                )
                for _ in range(depth)
            ]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(dim, eps=norm_eps)

        if repr_layer is not None:
            self.repr_norm = nn.LayerNorm(dim, eps=norm_eps)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the transformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            padding_mask: Boolean mask for padding tokens (True means ignore)
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions
                Shape: [seq_len, seq_len] or [batch_size, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim] or a tuple of
            (output tensor, intermediate representation) if `repr_layer` is set.
        """
        # Pass through each transformer block
        repr = None
        for i, layer in enumerate(self.layers):
            x = layer(x, padding_mask=padding_mask, attn_mask=attn_mask)
            if self.repr_layer is not None and i == self.repr_layer:
                repr = x.clone()

        # Apply final normalization
        x = self.norm(x)

        if self.repr_layer is not None:
            assert (
                repr is not None
            ), f"The specified `repr_layer` ({self.repr_layer}) was not reached during the forward pass."
            repr = self.repr_norm(repr)

            return x, repr

        return x


class ModernTransformer(nn.Module):
    """A modern Transformer model with optional RoPE embeddings, query-key normalization, and
    scaled dot-product attention.

    Args:
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        repr_layer: Layer at which to additionally extract intermediate representations. If None, no intermediate representation is extracted.
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
        is_decoder: Whether the transformer is used as a (cross-attention) decoder
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        repr_layer: Optional[int] = None,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = context_length
        self.repr_layer = repr_layer
        self.is_decoder = is_decoder

        layer_class = ModernTransformerDecoderBlock if is_decoder else ModernTransformerBlock
        self.layers = nn.ModuleList(
            [
                layer_class(
                    dim,
                    num_heads,
                    context_length=context_length,
                    rope_base=rope_base,
                    qk_layernorm=qk_layernorm,
                    use_sdpa=use_sdpa,
                    jvp_attn=jvp_attn,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(dim)

        if repr_layer is not None:
            self.repr_norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        h: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the modern transformer.

        Args:
            h: Input tensor of shape [batch_size, seq_len, dim].
            memory: Optional memory tensor for cross-attention (if decoder).
                Shape: [batch_size, seq_len, dim]
            pos_ids: Position ids for rotary embeddings.
                Shape: [batch_size, seq_len]
            padding_mask: Boolean mask for padding tokens (True means ignore).
                Shape: [batch_size, seq_len]
            tgt_key_padding_mask: Boolean mask for target padding tokens if decoder (True means ignore).
                Shape: [batch_size, seq_len]
            memory_key_padding_mask: Boolean mask for memory padding tokens if decoder (True means ignore).
                Shape: [batch_size, seq_len]
            attn_mask: Mask to prevent attention to certain positions.
                Values should be 0 for positions to attend to, and -inf for masked positions.
                To make the model causal (decoder-style), pass a causal mask.
                Shape: [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim] or a tuple of
            (output tensor, intermediate representation) if `repr_layer` is set.
        """
        _, seq_len, _ = h.shape
        assert (
            self.max_seq_len is None or seq_len <= self.max_seq_len
        ), "Sequence length exceeds model's maximum sequence length"

        attention_kwargs = {"padding_mask": padding_mask}
        if self.is_decoder:
            attention_kwargs = {
                "memory": memory,
                "tgt_key_padding_mask": tgt_key_padding_mask,
                "memory_key_padding_mask": memory_key_padding_mask,
            }

        repr = None
        for i, layer in enumerate(self.layers):
            h = layer(h, pos_ids=pos_ids, attn_mask=attn_mask, **attention_kwargs)
            if self.repr_layer is not None and i == self.repr_layer:
                repr = h.clone()

        h = self.norm(h)

        if self.repr_layer is not None:
            assert (
                repr is not None
            ), f"The specified `repr_layer` ({self.repr_layer}) was not reached during the forward pass."
            repr = self.repr_norm(repr)

            return h, repr

        return h


class Modulate(nn.Module):
    """clari-style AdaLN conditioning projection.

    Maps a condition vector ``c`` (B, dim_cond) to a per-feature scale (and optional
    shift) applied to ``x`` (B, seq_len, dim), broadcasting over the sequence.

    Initialization is the crux of this whole feature:
        - weights are zero-init  -> conditioning has *no* effect at step 0
        - scale bias = 1         -> the scale is identity at step 0
        - shift bias = 0         -> the shift is zero at step 0
    Used as a post-sublayer "gate" (``shift=False``), the scale bias of 1 keeps the
    residual branch fully ALIVE at init (unlike DiT-Zero, which zeroes the gate and
    starts every block as a dead passthrough).

    Args:
        dim: Feature dimension of ``x``.
        dim_cond: Dimension of the condition vector ``c``.
        shift: If True, also learn an additive shift; if False, scale only (gate).
    """

    @typecheck
    def __init__(self, dim: int, dim_cond: int, shift: bool = True):
        super().__init__()
        self.scale = nn.Linear(dim_cond, dim)
        nn.init.zeros_(self.scale.weight)
        nn.init.ones_(self.scale.bias)
        if shift:
            self.shift = nn.Linear(dim_cond, dim)
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        else:
            self.shift = None

    @typecheck
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply conditioned scale (and optional shift) to ``x``.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim].
            c: Condition vector of shape [batch_size, dim_cond].

        Returns:
            Modulated tensor of shape [batch_size, seq_len, dim].
        """
        x = self.scale(c).unsqueeze(1) * x
        if self.shift is not None:
            x = x + self.shift(c).unsqueeze(1)
        return x


class ConditionedModernTransformerBlock(nn.Module):
    """Twin of :class:`ModernTransformerBlock` with clari-style AdaLN conditioning.

    Identical attention / SwiGLU / RoPE behavior, but the norms are affine-free and a
    condition vector ``c`` modulates each sublayer: ``mod1`` applies scale+shift to the
    normalized input, ``mod2`` applies a scale-only gate to the sublayer output before
    the residual add. With :class:`Modulate`'s init, the block is an ordinary (live)
    transformer block at step 0 that ignores ``c`` and learns to use it.

    Args:
        dim: Input and output dimension.
        n_heads: Number of attention heads.
        dim_cond: Dimension of the condition vector (defaults to ``dim``).
        context_length: Maximum context length for rotary embeddings.
        rope_base: Base frequency for rotary embeddings.
        qk_layernorm: Whether to apply RMS normalization to queries and keys.
        use_sdpa: Whether to use PyTorch's scaled dot-product attention.
        jvp_attn: Whether to use JVP-compatible attention.
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dim_cond: Optional[int] = None,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        dim_cond = dim_cond if dim_cond is not None else dim
        self.attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        # Affine-free norms: modulation supplies the scale and shift.
        self.attention_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.ffn_norm = nn.RMSNorm(dim, elementwise_affine=False)

        self.mod1_attn = Modulate(dim, dim_cond)              # pre-attn scale + shift
        self.mod2_attn = Modulate(dim, dim_cond, shift=False)  # post-attn gate (scale only)
        self.mod1_ffn = Modulate(dim, dim_cond)
        self.mod2_ffn = Modulate(dim, dim_cond, shift=False)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN conditioning.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim].
            c: Condition vector of shape [batch_size, dim_cond].
            pos_ids: Position ids for rotary embeddings.
            padding_mask: Boolean mask for padding tokens (True means ignore).
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, dim].
        """
        attn_out = self.attention(
            self.mod1_attn(self.attention_norm(x), c),
            pos_ids=pos_ids,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        h = x + self.mod2_attn(attn_out, c)
        ff_out = self.feed_forward(self.mod1_ffn(self.ffn_norm(h), c))
        return h + self.mod2_ffn(ff_out, c)


class ConditionedModernTransformerDecoderBlock(nn.Module):
    """Twin of :class:`ModernTransformerDecoderBlock` with clari-style AdaLN conditioning.

    Same self-attention -> cross-attention -> feed-forward structure (and identical
    ``memory`` / ``query_pos_ids`` handling), but each of the three sublayers is wrapped
    with a ``mod1`` (pre, scale+shift) and ``mod2`` (post, scale-only gate) driven by the
    condition vector ``c``. Norms are affine-free.

    Args:
        dim: Input and output dimension.
        n_heads: Number of attention heads.
        dim_cond: Dimension of the condition vector (defaults to ``dim``).
        context_length: Maximum context length for rotary embeddings.
        rope_base: Base frequency for rotary embeddings.
        qk_layernorm: Whether to apply RMS normalization to queries and keys.
        use_sdpa: Whether to use PyTorch's scaled dot-product attention.
        jvp_attn: Whether to use JVP-compatible attention.
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        n_heads: int,
        dim_cond: Optional[int] = None,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        dim_cond = dim_cond if dim_cond is not None else dim
        self.self_attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.cross_attention = ModernAttention(
            dim,
            n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.feed_forward = SwiGLUFeedForward(dim=dim, hidden_dim=4 * dim)

        # Affine-free norms: modulation supplies the scale and shift.
        self.self_attention_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.cross_attention_norm = nn.RMSNorm(dim, elementwise_affine=False)
        self.ffn_norm = nn.RMSNorm(dim, elementwise_affine=False)

        self.mod1_sa = Modulate(dim, dim_cond)
        self.mod2_sa = Modulate(dim, dim_cond, shift=False)
        self.mod1_ca = Modulate(dim, dim_cond)
        self.mod2_ca = Modulate(dim, dim_cond, shift=False)
        self.mod1_ffn = Modulate(dim, dim_cond)
        self.mod2_ffn = Modulate(dim, dim_cond, shift=False)

    @typecheck
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        pos_ids: Optional[torch.Tensor] = None,
        query_pos_ids: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN conditioning.

        Args mirror :class:`ModernTransformerDecoderBlock.forward`, with an added
        condition vector ``c`` of shape [batch_size, dim_cond].

        Returns:
            Output tensor of shape [batch_size, seq_len, dim].
        """
        sa_pos_ids = query_pos_ids if query_pos_ids is not None else pos_ids
        sa_out = self.self_attention(
            self.mod1_sa(self.self_attention_norm(x), c),
            pos_ids=sa_pos_ids,
            padding_mask=tgt_key_padding_mask,
            attn_mask=attn_mask,
        )
        h = x + self.mod2_sa(sa_out, c)
        ca_out = self.cross_attention(
            self.mod1_ca(self.cross_attention_norm(h), c),
            memory=memory,
            pos_ids=pos_ids,
            query_pos_ids=query_pos_ids,
            padding_mask=memory_key_padding_mask,
            attn_mask=attn_mask,
        )
        h = h + self.mod2_ca(ca_out, c)
        ff_out = self.feed_forward(self.mod1_ffn(self.ffn_norm(h), c))
        return h + self.mod2_ffn(ff_out, c)


class ConditionedModernTransformer(nn.Module):
    """Twin of :class:`ModernTransformer` (encoder trunk) that threads a condition
    vector ``c`` through every :class:`ConditionedModernTransformerBlock`.

    Single-return (no auxiliary representation tap on this branch).

    Args:
        dim: Model dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        dim_cond: Dimension of the condition vector (defaults to ``dim``).
        context_length: Maximum context length for rotary embeddings.
        rope_base: Base frequency for rotary embeddings.
        qk_layernorm: Whether to apply RMS normalization to queries and keys.
        use_sdpa: Whether to use PyTorch's scaled dot-product attention.
        jvp_attn: Whether to use JVP-compatible attention.
    """

    @typecheck
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        dim_cond: Optional[int] = None,
        context_length: Optional[int] = 2048,
        rope_base: Optional[int] = 10_000,
        qk_layernorm: bool = True,
        use_sdpa: bool = True,
        jvp_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_seq_len = context_length

        self.layers = nn.ModuleList(
            [
                ConditionedModernTransformerBlock(
                    dim,
                    num_heads,
                    dim_cond=dim_cond,
                    context_length=context_length,
                    rope_base=rope_base,
                    qk_layernorm=qk_layernorm,
                    use_sdpa=use_sdpa,
                    jvp_attn=jvp_attn,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(dim)

    @typecheck
    def forward(
        self,
        h: torch.Tensor,
        c: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with AdaLN conditioning.

        Args:
            h: Input tensor of shape [batch_size, seq_len, dim].
            c: Condition vector of shape [batch_size, dim_cond].
            pos_ids: Position ids for rotary embeddings.
            padding_mask: Boolean mask for padding tokens (True means ignore).
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len].

        Returns:
            Output tensor of shape [batch_size, seq_len, dim].
        """
        _, seq_len, _ = h.shape
        assert (
            self.max_seq_len is None or seq_len <= self.max_seq_len
        ), "Sequence length exceeds model's maximum sequence length"

        for layer in self.layers:
            h = layer(h, c, pos_ids=pos_ids, padding_mask=padding_mask, attn_mask=attn_mask)

        return self.norm(h)
