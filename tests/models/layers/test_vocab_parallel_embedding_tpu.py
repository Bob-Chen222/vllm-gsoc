import pytest
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from vllm.model_executor.layers.vocab_parallel_embedding_tpu import (
    VocabParallelEmbedding, UnquantizedEmbeddingMethod)
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

    

def test_vocab_parallel_embedding_init():
    """Test initialization of VocabParallelEmbedding with different configurations."""
    # Test case 1: Basic initialization
    embedding = VocabParallelEmbedding(
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

    # Test case 2: With LoRA (org_num_embeddings < num_embeddings)
    embedding = VocabParallelEmbedding(
        num_embeddings=1050,  # Original + 50 LoRA tokens
        embedding_dim=512,
        org_num_embeddings=1000,
        params_dtype=jnp.float32
    )
    assert embedding.num_embeddings == 1050
    assert embedding.org_vocab_size == 1000
    assert embedding.num_added_embeddings == 50
    assert embedding.num_embeddings_padded == 1088  # Padded to multiple of 64

    # Test case 3: With custom padding size
    embedding = VocabParallelEmbedding(
        num_embeddings=1000,
        embedding_dim=512,
        padding_size=128,
        params_dtype=jnp.float32
    )
    assert embedding.num_embeddings_padded == 1024  # Padded to multiple of 128

def test_vocab_parallel_embedding_weight_loader():
    """Test weight loading functionality."""
    embedding = VocabParallelEmbedding(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    
    # Create a dummy weight tensor
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    
    # Test weight loading
    embedding.weight_loader(embedding.weight, weight)
    
    # Verify the loaded weights
    assert embedding.weight.value.shape == (1024, 512)  # Padded shape
    assert jnp.all(embedding.weight.value[:1000] == 1.0)  # Original weights
    assert jnp.all(embedding.weight.value[1000:] == 0.0)  # Padding should be zeros

def test_vocab_parallel_embedding_forward():
    """Test forward pass of VocabParallelEmbedding."""
    embedding = VocabParallelEmbedding(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    
    # Initialize weights with known values
    weight = jnp.ones((1000, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    
    # Test case 1: Basic forward pass
    input_ids = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    output = embedding.forward(input_ids)
    assert output.shape == (4, 512)
    assert jnp.all(output == 1.0)  # Since weights are all ones
    
    # Test case 2: With out-of-vocab indices
    input_ids = jnp.array([0, 999, 1000, 1001], dtype=jnp.int32)
    output = embedding.forward(input_ids)
    assert output.shape == (4, 512)
    # First two indices should have ones, last two should be zeros (out of vocab)
    assert jnp.all(output[0] == 1.0)
    assert jnp.all(output[1] == 1.0)
    assert jnp.all(output[2] == 0.0)
    assert jnp.all(output[3] == 0.0)

def test_vocab_parallel_embedding_with_lora():
    """Test VocabParallelEmbedding with LoRA tokens."""
    embedding = VocabParallelEmbedding(
        num_embeddings=1050,  # 1000 original + 50 LoRA tokens
        embedding_dim=512,
        org_num_embeddings=1000,
        params_dtype=jnp.float32
    )
    
    # Initialize weights
    weight = jnp.ones((1050, 512), dtype=jnp.float32)
    embedding.weight_loader(embedding.weight, weight)
    
    # Test forward pass with both original and LoRA tokens
    input_ids = jnp.array([0, 999, 1000, 1049], dtype=jnp.int32)
    output = embedding.forward(input_ids)
    assert output.shape == (4, 512)
    # All indices should have ones since they're within vocab range
    assert jnp.all(output == 1.0)

def test_vocab_parallel_embedding_shard_indices():
    """Test shard indices calculation."""
    embedding = VocabParallelEmbedding(
        num_embeddings=1000,
        embedding_dim=512,
        params_dtype=jnp.float32
    )
    
    # Verify shard indices
    indices = embedding.shard_indices
    assert indices.org_vocab_start_index == 0
    assert indices.org_vocab_end_index == 1000
    assert indices.padded_org_vocab_start_index == 0
    assert indices.padded_org_vocab_end_index == 1024
    assert indices.num_org_elements == 1000
    assert indices.num_org_elements_padded == 1024
    assert indices.num_org_vocab_padding == 24  # 1024 - 1000 