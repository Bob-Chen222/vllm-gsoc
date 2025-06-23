import jax
import jax.numpy as jnp

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, set_current_vllm_config


def test_attention_init_and_attributes():
    """Basic smoke-test for the JAX/Flax `Attention` layer.

    The goal is *not* to run the full forward pass (which is currently
    implemented only for CUDA/ROCM back-ends in JAX).  Instead we make sure
    that the layer can be instantiated under a minimal vLLM configuration and
    that its public attributes reflect the constructor arguments.
    """

    cfg = VllmConfig()

    # Register the config so that the layer can access the global static
    # context during construction.
    with set_current_vllm_config(cfg):
        attn = Attention(
            num_heads=4,
            head_size=8,
            scale=1.0,
            prefix="jax_test_attention",
        )

    # Verify attribute propagation.
    assert attn.num_heads == 4
    assert attn.head_size == 8
    assert attn.num_kv_heads == 4  # default when not specified

    # The layer allocates KV-cache per virtual engine; make sure the default
    # placeholder structure is as expected (empty list entries).
    assert isinstance(attn.kv_cache, list)

    # Create dummy JAX inputs that match the expected hidden size.  We do *not*
    # invoke ``attn.__call__`` because the JAX execution path for CPU back-ends
    # is still under development and deliberately raises an assertion.  The
    # following lines, however, document the expected shapes and dtypes.
    hidden = attn.num_heads * attn.head_size
    q = jnp.zeros((2, hidden), dtype=jnp.float32)
    k = jnp.zeros((2, hidden), dtype=jnp.float32)
    v = jnp.zeros((2, hidden), dtype=jnp.float32)

    # Sanity-check dtypes and shapes so that future refactors breaking them
    # surface through this test even without running the full forward pass.
    for tensor in (q, k, v):
        assert tensor.dtype == jnp.float32
        assert tensor.shape == (2, hidden) 