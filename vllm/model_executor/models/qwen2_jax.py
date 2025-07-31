from collections.abc import Iterable
from typing import Any, Optional, Union, List, Tuple

import torch
# from torch import nn
from transformers import Qwen2Config
import time

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from functools import partial

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix, apply_name_forward, apply_name_backward)


init_fn = nnx.initializers.uniform()





def apply_rope(x: jax.Array, *_args, **_kwargs) -> jax.Array:
    # No-op placeholder for rotary embedding application
    return x


def attention(kv_cache: Optional[jax.Array],
              q: jax.Array,
              k: jax.Array,
              v: jax.Array,
              *_args, **_kwargs) -> tuple[jax.Array, jax.Array]:
    # Minimal placeholder: return kv_cache unchanged and pass-through v
    return kv_cache, v



class Qwen2MLP(nnx.Module):

    def __init__(self, config: Qwen2Config, dtype: jnp.dtype, rng: nnx.Rngs):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        act = "relu"

        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.act_fn = jax.nn.relu

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        result = self.down_proj(fuse)
        return result


class Qwen2Attention(nnx.Module):

    def __init__(self, config: Qwen2Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = config.head_dim

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.mesh = mesh

        self.q_proj = nnx.Einsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, self.head_dim),
            (self.num_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.k_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            (self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.v_proj = nnx.Einsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            (self.num_kv_heads, self.head_dim),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            bias_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.o_proj = nnx.Einsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # q: (T, N, H)
        q = self.q_proj(x)
        q = apply_rope(q, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # k: (T, K, H)
        k = self.k_proj(x)
        k = apply_rope(k, md.input_positions, self.head_dim_original,
                       self.rope_theta, self.rope_scaling)

        # v: (T, K, H)
        v = self.v_proj(x)
        # o: (T, N, H)
        q_scale = k_scale = v_scale = None
        new_kv_cache, outputs = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        # (T, D)
        o = self.o_proj(outputs)
        return new_kv_cache, o


class Qwen2DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen2Config, dtype: jnp.dtype, rng: nnx.Rngs,
                 mesh: Mesh, kv_cache_dtype: str):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.self_attn = Qwen2Attention(config=config,
                                        dtype=dtype,
                                        rng=rng,
                                        mesh=mesh,
                                        kv_cache_dtype=kv_cache_dtype)
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        self.mlp = Qwen2MLP(
            config=config,
            dtype=dtype,
            rng=rng,
        )

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: Any,
    ) -> Tuple[jax.Array, jax.Array]:
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(
            kv_cache,
            hidden_states,
            attention_metadata,
        )
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return kv_cache, outputs


class Qwen2Model(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
                 mesh: Mesh) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        # Restrict to bf16 regardless of upstream config
        dtype = jnp.bfloat16
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=hidden_size,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.layers = [
            Qwen2DecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype)
            for _ in range(hf_config.num_hidden_layers)
        ]
        self.norm = nnx.RMSNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
        )
        if model_config.hf_config.tie_word_embeddings:
            self.lm_head = self.embed.embedding
        else:
            self.lm_head = nnx.Param(
                init_fn(rng.params(), (hidden_size, vocab_size), dtype),
                sharding=(None, "model"),
            )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i]
            kv_cache, x = layer(
                kv_cache,
                x,
                attention_metadata,
            )
            kv_caches[i] = kv_cache
        x = self.norm(x)
        return kv_caches, x


class Qwen2ForCausalLM(nnx.Module):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        self.rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Qwen2Model(
            vllm_config=vllm_config,
            rng=self.rng,
            mesh=mesh,
        )

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: Optional[jax.Array],
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if self.vllm_config.model_config.hf_config.tie_word_embeddings:
            logits = jnp.dot(hidden_states, self.model.lm_head.value.T)
        else:
            logits = jnp.dot(hidden_states, self.model.lm_head.value)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   jax.Array]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.vllm_config.model_config.hf_config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
