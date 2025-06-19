import pytest
import numpy as np
import jax
import jax.numpy as jnp

from vllm.model_executor.layers.layernorm import RMSNorm


def test_rmsnorm_basic_functionality():
    """Test basic RMS normalization functionality."""
    hidden_size = 8
    num_tokens = 4
    eps = 1e-6
    
    # Create test data
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                   [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    
    # Create RMSNorm layer with weight=1
    layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
    layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
    
    # Apply RMS normalization
    output = layer(x)
    
    # Verify output shape
    assert output.shape == x.shape
    
    # Verify that the output is properly normalized
    # For each row, the RMS should be approximately 1/sqrt(hidden_size)
    expected_rms = 1.0 / jnp.sqrt(hidden_size)
    
    for i in range(num_tokens):
        row_rms = jnp.sqrt(jnp.mean(output[i] ** 2))
        assert jnp.abs(row_rms - expected_rms) < 1e-5


def test_rmsnorm_with_weight():
    """Test RMS normalization with learned weights."""
    hidden_size = 4
    eps = 1e-6
    
    x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                   [2.0, 4.0, 6.0, 8.0]], dtype=jnp.float32)
    
    # Create weights
    weights = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    
    layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
    layer.weight.value = weights
    
    output = layer(x)
    
    # Verify output shape
    assert output.shape == x.shape
    
    # Verify that weights are applied correctly
    # The output should be: normalized_input * weights
    normalized_input = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    expected_output = normalized_input * weights
    
    np.testing.assert_allclose(output, expected_output, atol=1e-6)


def test_rmsnorm_with_residual():
    """Test RMS normalization with residual connection."""
    hidden_size = 4
    eps = 1e-6
    
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    residual = jnp.array([[0.5, 1.0, 1.5, 2.0]], dtype=jnp.float32)
    
    layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
    layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
    
    output, updated_residual = layer(x, residual)
    
    # Verify output shapes
    assert output.shape == x.shape
    assert updated_residual.shape == residual.shape
    
    # Verify that residual is updated correctly (x + residual)
    expected_residual = x + residual
    np.testing.assert_allclose(updated_residual, expected_residual, atol=1e-6)
    
    # Verify that output is normalized version of (x + residual)
    combined_input = x + residual
    normalized = combined_input / jnp.sqrt(jnp.mean(combined_input ** 2, axis=-1, keepdims=True) + eps)
    np.testing.assert_allclose(output, normalized, atol=1e-6)


def test_rmsnorm_edge_cases():
    """Test edge cases for RMS normalization."""
    eps = 1e-6
    
    # Test with hidden_size = 1
    layer = RMSNorm(1, eps=eps, dtype=jnp.float32)
    layer.weight.value = jnp.array([2.0], dtype=jnp.float32)
    
    x = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    output = layer(x)
    
    assert output.shape == x.shape
    # With hidden_size=1, normalization should just apply the weight
    expected = x * 2.0
    np.testing.assert_allclose(output, expected, atol=1e-6)
    
    # Test with all zeros
    x_zeros = jnp.zeros((2, 4), dtype=jnp.float32)
    layer_zeros = RMSNorm(4, eps=eps, dtype=jnp.float32)
    layer_zeros.weight.value = jnp.ones(4, dtype=jnp.float32)
    
    output_zeros = layer_zeros(x_zeros)
    # Should handle zeros gracefully (output will be zeros due to division by sqrt(eps))
    assert output_zeros.shape == x_zeros.shape


def test_rmsnorm_shape_validation():
    """Test shape validation in RMS normalization."""
    layer = RMSNorm(4, dtype=jnp.float32)
    
    # Test with correct shape
    x_correct = jnp.ones((2, 4), dtype=jnp.float32)
    output = layer(x_correct)
    assert output.shape == (2, 4)
    
    # Test with incorrect hidden size
    x_wrong = jnp.ones((2, 6), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Expected hidden_size to be 4, but found: 6"):
        layer(x_wrong)


def test_rmsnorm_3d_input():
    """Test RMS normalization with 3D input (batch, seq, hidden)."""
    batch_size, seq_len, hidden_size = 2, 3, 4
    eps = 1e-6
    
    x = jnp.random.randn(batch_size, seq_len, hidden_size).astype(jnp.float32)
    
    layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
    layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
    
    output = layer(x)
    
    assert output.shape == x.shape
    
    # Verify normalization is applied correctly across the last dimension
    for b in range(batch_size):
        for s in range(seq_len):
            row = output[b, s]
            row_rms = jnp.sqrt(jnp.mean(row ** 2))
            expected_rms = 1.0 / jnp.sqrt(hidden_size)
            assert jnp.abs(row_rms - expected_rms) < 1e-5


def test_rmsnorm_different_eps():
    """Test RMS normalization with different epsilon values."""
    hidden_size = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    
    # Test with different epsilon values
    for eps in [1e-6, 1e-5, 1e-4]:
        layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
        layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
        
        output = layer(x)
        
        # Verify the normalization formula
        expected = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
        np.testing.assert_allclose(output, expected, atol=1e-6)


def test_rmsnorm_no_weight():
    """Test RMS normalization without weight parameter."""
    hidden_size = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    
    layer = RMSNorm(hidden_size, has_weight=False, dtype=jnp.float32)
    
    output = layer(x)
    
    # Without weight, should just normalize
    expected = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + layer.variance_epsilon)
    np.testing.assert_allclose(output, expected, atol=1e-6)


def test_rmsnorm_variance_size_override():
    """Test RMS normalization with variance size override."""
    hidden_size = 8
    var_hidden_size = 4
    eps = 1e-6
    
    x = jnp.random.randn(2, hidden_size).astype(jnp.float32)
    
    layer = RMSNorm(hidden_size, eps=eps, var_hidden_size=var_hidden_size, dtype=jnp.float32)
    layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
    
    output = layer(x)
    
    # Variance should be computed only over the first var_hidden_size dimensions
    expected = x / jnp.sqrt(jnp.mean(x[:, :var_hidden_size] ** 2, axis=-1, keepdims=True) + eps)
    np.testing.assert_allclose(output, expected, atol=1e-6)
    
    # Test error case: hidden_size < var_hidden_size
    layer_error = RMSNorm(2, eps=eps, var_hidden_size=4, dtype=jnp.float32)
    x_error = jnp.random.randn(1, 2).astype(jnp.float32)
    
    with pytest.raises(ValueError, match="Expected hidden_size to be at least 4, but found: 2"):
        layer_error(x_error) 