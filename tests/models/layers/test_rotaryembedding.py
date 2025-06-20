import pytest
import jax
import jax.numpy as jnp
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

# Helper function for reference implementation in JAX
def reference_rotary_embedding_jax(
    query: jnp.ndarray,
    key: jnp.ndarray,
    head_size: int,
    base: float,
    is_neox_style: bool,
    rotary_dim: int,
    positions: jnp.ndarray,
):
    inv_freq = 1.0 / (base**(
        jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim))
    t = positions.astype(jnp.float32)
    freqs = jnp.einsum("i,j -> ij", t, inv_freq)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)

    def apply_rotary(x: jnp.ndarray) -> jnp.ndarray:
        dtype = x.dtype
        cos_exp = jnp.expand_dims(cos, -2).astype(dtype)
        sin_exp = jnp.expand_dims(sin, -2).astype(dtype)

        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        if is_neox_style:
            x1, x2 = jnp.split(x_rot, 2, axis=-1)
            o1 = x1 * cos_exp - x2 * sin_exp
            o2 = x2 * cos_exp + x1 * sin_exp
            x_rot = jnp.concatenate((o1, o2), axis=-1)
        else:
            x_reshaped = x_rot.reshape(*x_rot.shape[:-1], -1, 2)
            x1 = x_reshaped[..., 0]
            x2 = x_reshaped[..., 1]
            o1 = x1 * cos_exp - x2 * sin_exp
            o2 = x2 * cos_exp + x1 * sin_exp
            x_rot = jnp.stack((o1, o2), axis=-1).reshape(x_rot.shape)

        if x_pass.shape[-1] > 0:
            return jnp.concatenate((x_rot, x_pass), axis=-1)
        else:
            return x_rot

    rotated_query = apply_rotary(query)
    rotated_key = apply_rotary(key)
    return rotated_query, rotated_key

@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim_fraction", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("seed", [0])
def test_rotary_embedding_jax_only(
    is_neox_style: bool,
    rotary_dim_fraction: float,
    dtype: jnp.dtype,
    seed: int,
):
    key = jax.random.PRNGKey(seed)
    head_size = 64
    rotary_dim = int(head_size * rotary_dim_fraction)
    max_position_embeddings = 4096
    base = 10000.0
    num_tokens = 128
    num_heads = 4

    # Generate synthetic inputs
    k1, k2, k3 = jax.random.split(key, 3)
    query_shape = (num_tokens, num_heads, head_size)
    key_shape = (num_tokens, num_heads, head_size)

    query = jax.random.normal(k1, query_shape, dtype=dtype)
    key_ = jax.random.normal(k2, key_shape, dtype=dtype)
    positions = jax.random.randint(k3, (num_tokens,), 0, max_position_embeddings)

    # Reference output (ground truth)
    ref_query, ref_key = reference_rotary_embedding_jax(
        query=query,
        key=key_,
        head_size=head_size,
        base=base,
        is_neox_style=is_neox_style,
        rotary_dim=rotary_dim,
        positions=positions,
    )

    # Actual implementation from vLLM (JAX version)
    rotary_emb = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )

    out_query, out_key = rotary_emb(
        positions=positions,
        query=query,
        key=key_
    )

    # Compare outputs
    assert jnp.allclose(out_query, ref_query, atol=1e-3, rtol=1e-5), "Query mismatch"
    assert jnp.allclose(out_key, ref_key, atol=1e-3, rtol=1e-5), "Key mismatch"
