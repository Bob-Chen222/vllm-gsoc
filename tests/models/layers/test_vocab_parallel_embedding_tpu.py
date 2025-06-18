import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from vllm.model_executor.layers.vocab_parallel_embedding_tpu import (
    VocabParallelEmbedding as VocabParallelEmbeddingJAX, UnquantizedEmbeddingMethod)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbeddingPT)
from vllm.distributed import (get_tensor_model_parallel_rank,
                            get_tensor_model_parallel_world_size)

# Mock the distributed functions
def mock_get_tensor_model_parallel_rank():
    return 0

def mock_get_tensor_model_parallel_world_size():
    return 1

# Fixture to patch the distributed functions
@pytest.fixture(autouse=True)
def patch_distributed():
    import vllm.distributed
    vllm.distributed.get_tensor_model_parallel_rank = mock_get_tensor_model_parallel_rank
    vllm.distributed.get_tensor_model_parallel_world_size = mock_get_tensor_model_parallel_world_size
    yield
    # Restore original functions after test
    vllm.distributed.get_tensor_model_parallel_rank = get_tensor_model_parallel_rank
    vllm.distributed.get_tensor_model_parallel_world_size = get_tensor_model_parallel_world_size

# === TESTS FOR BOTH IMPLEMENTATIONS ===

@pytest.mark.parametrize("input_ids", [
    jnp.array([0, 1, 2, 3], dtype=jnp.int32),
    jnp.array([0, 999, 1000, 1001], dtype=jnp.int32),
    jnp.array([999], dtype=jnp.int32),
    jnp.array([-1, 0], dtype=jnp.int32),
    jnp.array([], dtype=jnp.int32),
    jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
    jnp.array([5, 5, 5, 5], dtype=jnp.int32),
])
def test_jax_vs_pt_embedding_outputs_match(input_ids):
    num_embeddings = 1000
    embedding_dim = 512

    # Initialize the same weights for both
    weight = jnp.arange(num_embeddings * embedding_dim, dtype=jnp.float32).reshape((num_embeddings, embedding_dim))

    # PT version
    embedding_pt = VocabParallelEmbeddingPT(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        params_dtype=jnp.float32
    )
    embedding_pt.weight_loader(embedding_pt.weight, weight)

    # JAX version
    embedding_jax = VocabParallelEmbeddingJAX(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        params_dtype=jnp.float32
    )
    embedding_jax.weight_loader(embedding_jax.weight, weight)

    pt_out = embedding_pt(input_ids)
    jax_out = embedding_jax(input_ids)

    assert pt_out.shape == jax_out.shape, "Oujaxt shapes differ"
    assert jnp.allclose(pt_out, jax_out, atol=1e-5), "Outputs mismatch between PT and JAX embedding"

# === YOUR ORIGINAL TESTS BELOW (UNCHANGED) ===

def test_vocab_parallel_embedding_init():
    """Test initialization of VocabParallelEmbedding with different configurations."""
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    assert embedding.num_embeddings == 1000
    assert embedding.embedding_dim == 512
    assert embedding.tp_size == 1
    assert embedding.org_vocab_size == 1000
    assert embedding.num_embeddings_padded == 1024  # Padded to multiple of 64
    assert embedding.num_embeddings_per_partition == 1024

    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1050,
        embedding_dim=512,
        org_num_embeddings=1000,
        params_dtype=jnp.float32
    )
    assert embedding.num_embeddings == 1050
    assert embedding.org_vocab_size == 1000
    assert embedding.num_added_embeddings == 50
    assert embedding.num_embeddings_padded == 1088  # Padded to multiple of 64

    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        padding_size=128,
        params_dtype=jnp.float32
    )
    assert embedding.num_embeddings_padded == 1024  # Padded to multiple of 128

def test_vocab_parallel_embedding_weight_loader():
    """Test weight loading functionality."""
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    assert embedding.weight.value.shape == (1024, 512)
    assert jnp.all(embedding.weight.value[:1000] == 1.0)
    assert jnp.all(embedding.weight.value[1000:] == 0.0)

def test_vocab_parallel_embedding_forward():
    """Test forward pass of VocabParallelEmbedding."""
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (4, 512)
    assert jnp.all(output == 1.0)
    input_ids = jnp.array([0, 999, 1000, 1001], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (4, 512)
    assert jnp.all(output[0] == 1.0)
    assert jnp.all(output[1] == 1.0)
    assert jnp.all(output[2] == 0.0)
    assert jnp.all(output[3] == 0.0)

def test_vocab_parallel_embedding_shard_indices():
    """Test shard indices calculation."""
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    indices = embedding.shard_indices
    assert indices.org_vocab_start_index == 0
    assert indices.org_vocab_end_index == 1000
    assert indices.padded_org_vocab_start_index == 0
    assert indices.padded_org_vocab_end_index == 1024
    assert indices.num_org_elements == 1000
    assert indices.num_org_elements_padded == 1024
    assert indices.num_org_vocab_padding == 24  # 1024 - 1000 

def test_forward_empty_input():
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (0, 512)

def test_forward_batched_input():
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (2, 2, 512)
    assert jnp.all(output == 1.0)

def test_forward_max_index():
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([999], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (1, 512)
    assert jnp.all(output == 1.0)

def test_forward_negative_indices():
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([-1, 0], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (2, 512)
    assert jnp.all(output[0] == 0.0)
    assert jnp.all(output[1] == 1.0)

def test_forward_invalid_dtype():
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([0, 1, 2], dtype=jnp.float32)
    with pytest.raises(TypeError):
        _ = embedding(input_ids)

def test_forward_repeated_ids():
    embedding = VocabParallelEmbeddingJAX(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    input_ids = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
    output = embedding(input_ids)
    assert output.shape == (4, 512)
    assert jnp.all(output == 1.0)
