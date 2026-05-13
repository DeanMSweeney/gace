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
from adsflow.utils.tft.typing_utils import typecheck


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN scale and shift to a normalized tensor.

    Reshapes the distribution of x based on the condition, broadcasting
    (B, C) shift/scale over the sequence dimension (B, seq_len, C).
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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


class ConditionedModernTransformerDecoderBlock(nn.Module):
    """ModernTransformerDecoderBlock with DiT-style AdaLN-Zero conditioning.

    Identical to ModernTransformerDecoderBlock but accepts a condition vector `c` and
    applies adaptive layer normalization (scale, shift, gate) at all three sub-layers:
    self-attention, cross-attention, and feedforward.

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
            dim, n_heads,
            context_length=context_length,
            rope_base=rope_base,
            use_qk_norm=qk_layernorm,
            use_sdpa=use_sdpa,
            jvp_attn=jvp_attn,
        )
        self.cross_attention = ModernAttention(
            dim, n_heads,
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

        # Projects c → 9 params: (shift, scale, gate) × (self-attn, cross-attn, ffn).
        # Zero-init so every block is an identity at the start of training (DiT-Zero).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

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

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            c: Condition vector of shape [batch_size, dim]
            memory: Optional memory tensor for cross-attention
            pos_ids: Position ids for rotary embeddings (memory / combined sequence)
            query_pos_ids: Optional override for query token positions
            tgt_key_padding_mask: Boolean mask for target padding tokens (True means ignore)
            memory_key_padding_mask: Boolean mask for memory padding tokens (True means ignore)
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_ff, scale_ff, gate_ff = (
            self.adaLN_modulation(c).chunk(9, dim=-1)
        )
        sa_pos_ids = query_pos_ids if query_pos_ids is not None else pos_ids
        h = x + gate_sa.unsqueeze(1) * self.self_attention(
            modulate(self.self_attention_norm(x), shift_sa, scale_sa),
            pos_ids=sa_pos_ids,
            padding_mask=tgt_key_padding_mask,
            attn_mask=attn_mask,
        )
        h = h + gate_ca.unsqueeze(1) * self.cross_attention(
            modulate(self.cross_attention_norm(h), shift_ca, scale_ca),
            memory=memory,
            pos_ids=pos_ids,
            query_pos_ids=query_pos_ids,
            padding_mask=memory_key_padding_mask,
            attn_mask=attn_mask,
        )
        out = h + gate_ff.unsqueeze(1) * self.feed_forward(
            modulate(self.ffn_norm(h), shift_ff, scale_ff)
        )
        return out


class ConditionedModernTransformerBlock(nn.Module):
    """ModernTransformerBlock with DiT-style AdaLN-Zero conditioning.

    Identical to ModernTransformerBlock but accepts a condition vector `c` and applies
    adaptive layer normalization (scale, shift, gate) derived from `c` at both the
    attention and feedforward sub-layers.

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

        # Projects c → (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff).
        # Zero-init so every block is an identity at the start of training (DiT-Zero).
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

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
            x: Input tensor of shape [batch_size, seq_len, dim]
            c: Condition vector of shape [batch_size, dim]
            pos_ids: Position ids for rotary embeddings
            padding_mask: Boolean mask for padding tokens (True means ignore)
            attn_mask: Attention mask of shape [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = x + gate_attn.unsqueeze(1) * self.attention(
            modulate(self.attention_norm(x), shift_attn, scale_attn),
            pos_ids=pos_ids,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
        )
        out = h + gate_ff.unsqueeze(1) * self.feed_forward(
            modulate(self.ffn_norm(h), shift_ff, scale_ff)
        )
        return out


class ConditionedModernTransformer(nn.Module):
    """Stack of ConditionedModernTransformerBlock layers with AdaLN conditioning.

    Threads the condition vector `c` through every block. Otherwise identical
    to ModernTransformer.

    Args:
        dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        context_length: Maximum context length for rotary embeddings
        rope_base: Base frequency for rotary embeddings
        repr_layer: Layer at which to extract intermediate representations
        qk_layernorm: Whether to apply RMS normalization to queries and keys
        use_sdpa: Whether to use PyTorch's scaled dot-product attention
        jvp_attn: Whether to use JVP-compatible attention
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
    ):
        super().__init__()
        self.repr_layer = repr_layer

        self.layers = nn.ModuleList(
            [
                ConditionedModernTransformerBlock(
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
        c: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with AdaLN conditioning.

        Args:
            h: Input tensor of shape [batch_size, seq_len, dim]
            c: Condition vector of shape [batch_size, dim]
            pos_ids: Position ids for rotary embeddings
            padding_mask: Boolean mask for padding tokens (True means ignore)
            attn_mask: Attention mask shape [batch_size, n_heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch_size, seq_len, dim], or a tuple of
            (output, intermediate representation) if repr_layer is set.
        """
        repr = None
        for i, layer in enumerate(self.layers):
            h = layer(h, c, pos_ids=pos_ids, padding_mask=padding_mask, attn_mask=attn_mask)
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
