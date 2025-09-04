# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings Base Class."""
from typing import Optional

import jax
import jax.numpy as jnp
import torch
from functools import partial
from flax import nnx

from vllm.model_executor.custom_op import CustomOp

from .common import apply_rotary_emb_dispatch, apply_rotary_emb_torch
def _apply_rotary_emb_jax(
    x: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> jnp.ndarray:
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)

    x1, x2 = jnp.split(x, 2, axis=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return jnp.concatenate([o1, o2], axis=-1)



def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      is_neox_style: bool) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    if current_platform.is_cuda():
        return apply_rotary_emb(x.unsqueeze(0), cos, sin,
                                not is_neox_style).squeeze(0)
    else:
        return _apply_rotary_emb_torch(x, cos, sin, is_neox_style)


@partial(jax.jit, static_argnames=("head_size", "rotary_dim"))
def _rotary_embedding(positions: jax.Array, query: jax.Array, key: Optional[jax.Array],cos_sin_cache,
                      head_size, rotary_dim):
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = jnp.take(cos_sin_cache, positions, axis=0)
    cos, sin = jnp.split(cos_sin, 2, axis=-1)

    query_shape = query.shape
    query = jnp.reshape(query, (num_tokens, -1, head_size))
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = _apply_rotary_emb_jax(query_rot, cos, sin)
    query = jnp.reshape(jnp.concatenate([query_rot, query_pass], axis=-1), query_shape)

    # key may be None in some cases, e.g. cross-layer KV sharing
    key_shape = key.shape
    key = jnp.reshape(key, (num_tokens, -1, head_size))
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = _apply_rotary_emb_jax(key_rot, cos, sin)
    key = jnp.reshape(jnp.concatenate([key_rot, key_pass], axis=-1), key_shape)
    return query, key

@CustomOp.register("rotary_embedding")
class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: jnp.dtype
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = jnp.asarray(cache, dtype=dtype)
        self.cos_sin_cache: jnp.Array
        self.cos_sin_cache = nnx.Variable(cache)

    def _compute_inv_freq(self, base: float) -> jax.Array:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> jax.Array:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)

        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        cos = jnp.cos(freqs)
        sin = jnp.sin(freqs)
        cache = jnp.concatenate([cos, sin], axis=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb_torch(query_rot, cos, sin,
                                            self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = _apply_rotary_emb_torch(key_rot, cos, sin,
                                              self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key
    
    # @nnx.jit
    def __call__(
            self,
            positions: jax.Array,
            query: jax.Array,
            key: Optional[jax.Array] = None,
            offsets: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, Optional[jax.Array]]:
        return _rotary_embedding(
            positions, query, key, self.cos_sin_cache.value, self.head_size, self.rotary_dim)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm import _custom_ops as ops

        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)

        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        from vllm._ipex_ops import ipex_ops as ops

        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
                                                   dtype=query.dtype)
        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if key is None:
            # XPU kernel doesn't support key=None so fall back to native impl
            # TODO(sarckk): add support for optional key in
            # ipex.llm.functional.rotary_embedding_batched
            return self.forward_native(positions, query, key, offsets)
        else:
            if offsets is not None:
                ops.batched_rotary_embedding(positions, query, key,
                                             self.head_size,
                                             self.cos_sin_cache,
                                             self.is_neox_style,
                                             self.rotary_dim, offsets)
            else:
                ops.rotary_embedding(positions, query, key, self.head_size,
                                     self.cos_sin_cache, self.is_neox_style)
        return query, key

    def forward_neuron(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        def _apply_rotary_emb_neuron(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            is_neox_style: bool,
        ) -> torch.Tensor:
            cos = cos.unsqueeze(-2).to(x.dtype)
            sin = sin.unsqueeze(-2).to(x.dtype)
            if is_neox_style:
                x1, x2 = torch.chunk(x, 2, dim=-1)
            else:
                # x1 = x[..., ::2]

                # x2 = x[..., 1::2]
                d = x.shape[-1] // 2
                x_reshaped = x.view(-1, x.shape[-1])
                x1 = x_reshaped[:, ::2].view(*x.shape[:-1], d)
                x2 = x_reshaped[:, 1::2].view(*x.shape[:-1], d)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if is_neox_style:
                return torch.cat((o1, o2), dim=-1)
            else:
                return torch.stack((o1, o2), dim=-1).flatten(-2)

        if offsets is not None:
            positions = positions + offsets

        self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                   dtype=query.dtype)

        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)

        if self.rotary_dim == self.head_size:
            query = _apply_rotary_emb(query, cos, sin, self.is_neox_style)
            query = query.reshape(query_shape)
            if key is not None:
                key = _apply_rotary_emb(key, cos, sin, self.is_neox_style)
                key = key.reshape(key_shape)
        else:
            head_size = query.shape[-1]
            query_reshaped = query.view(-1, head_size)
            query_pass = query_reshaped[:, self.rotary_dim:].view(
                *query.shape[:-1], head_size - self.rotary_dim)
            query_rot = query_reshaped[:, :self.rotary_dim].view(
                *query.shape[:-1], self.rotary_dim)
            query_rot = _apply_rotary_emb_neuron(query_rot, cos, sin,
                                                 self.is_neox_style)
            query = torch.cat((query_rot, query_pass),
                              dim=-1).reshape(query_shape)

            if key is not None:
                key_reshaped = key.view(-1, head_size)
                key_pass = key_reshaped[:, self.rotary_dim:].view(
                    *key.shape[:-1], head_size - self.rotary_dim)
                key_rot = key_reshaped[:, :self.rotary_dim].view(
                    *key.shape[:-1], self.rotary_dim)
                key_rot = _apply_rotary_emb_neuron(key_rot, cos, sin,
                                                   self.is_neox_style)
                key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s