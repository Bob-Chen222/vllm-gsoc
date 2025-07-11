import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding as VocabParallelEmbeddingTorch
from vllm.model_executor.layers.vocab_parallel_embedding_tpu import VocabParallelEmbedding as VocabParallelEmbeddingJAX

@pytest.mark.parametrize("input_ids_np", [
    np.array([0, 1, 2, 3], dtype=np.int32),
    np.array([0, 999, 1000, 1001], dtype=np.int32),
    np.array([999], dtype=np.int32),
    np.array([-1, 0], dtype=np.int32),
    np.array([], dtype=np.int32),
    np.array([[0, 1], [2, 3]], dtype=np.int32),
    np.array([5, 5, 5, 5], dtype=np.int32),
])
def test_torch_vs_jax_embedding_outputs_match(input_ids_np):
    num_embeddings = 1000
    embedding_dim = 128

    # Initialize the same weights for both
    weight_np = np.arange(num_embeddings * embedding_dim, dtype=np.float32).reshape((num_embeddings, embedding_dim))

    # Torch (PyTorch) version
    torch_embedding = VocabParallelEmbeddingTorch(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        params_dtype=torch.float32
    )
    torch_weight = torch.from_numpy(weight_np)
    torch_embedding.weight_loader(torch_embedding.weight, torch_weight)

    # JAX version
    jax_embedding = VocabParallelEmbeddingJAX(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        params_dtype=jnp.float32
    )
    jax_weight = jnp.array(weight_np)
    jax_embedding.weight_loader(jax_embedding.weight, jax_weight)

    # Prepare inputs for each framework
    torch_input = torch.from_numpy(input_ids_np)
    jax_input = jnp.array(input_ids_np)

    torch_out = torch_embedding(torch_input).detach().cpu().numpy()
    jax_out = np.array(jax_embedding(jax_input))

    assert torch_out.shape == jax_out.shape, "Shape mismatch"
    assert np.allclose(torch_out, jax_out, atol=1e-5), "Mismatch between PyTorch and JAX embedding outputs"


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
