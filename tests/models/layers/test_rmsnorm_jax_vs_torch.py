import numpy as np
import jax.numpy as jnp
import pytest
from vllm.model_executor.layers.layernorm import RMSNorm

# Assume RMSNorm is correctly imported and implemented

def test_rmsnorm_basic_functionality():
    """Test basic RMS normalization functionality."""
    hidden_size = 8
    num_tokens = 4
    eps = 1e-6
    
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                   [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    
    layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
    layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
    
    output = layer(x)
    assert output.shape == x.shape
    
    # The RMS of each row after RMSNorm should be approximately 1.0
    for i in range(num_tokens):
        row_rms = jnp.sqrt(jnp.mean(output[i] ** 2))
        assert jnp.abs(row_rms - 1.0) < 1e-5

def test_rmsnorm_with_weight():
    """Test RMS normalization with learned weights."""
    hidden_size = 4
    eps = 1e-6
    x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                   [2.0, 4.0, 6.0, 8.0]], dtype=jnp.float32)
    weights = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
    layer.weight.value = weights
    output = layer(x)
    assert output.shape == x.shape
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
    assert output.shape == x.shape
    assert updated_residual.shape == residual.shape
    expected_residual = x + residual
    np.testing.assert_allclose(updated_residual, expected_residual, atol=1e-6)
    normalized = expected_residual / jnp.sqrt(jnp.mean(expected_residual ** 2, axis=-1, keepdims=True) + eps)
    np.testing.assert_allclose(output, normalized, atol=1e-6)

def test_rmsnorm_edge_cases():
    """Test edge cases for RMS normalization."""
    eps = 1e-6
    # hidden_size = 1, weight = 2
    layer = RMSNorm(1, eps=eps, dtype=jnp.float32)
    layer.weight.value = jnp.array([2.0], dtype=jnp.float32)
    x = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    output = layer(x)
    assert output.shape == x.shape
    expected = jnp.array([[2.0], [2.0], [2.0]], dtype=jnp.float32)
    np.testing.assert_allclose(output, expected, atol=1e-6)

    # all zeros
    x_zeros = jnp.zeros((2, 4), dtype=jnp.float32)
    layer_zeros = RMSNorm(4, eps=eps, dtype=jnp.float32)
    layer_zeros.weight.value = jnp.ones(4, dtype=jnp.float32)
    output_zeros = layer_zeros(x_zeros)
    # Should handle zeros gracefully (output will be zeros)
    assert output_zeros.shape == x_zeros.shape
    np.testing.assert_allclose(output_zeros, jnp.zeros_like(x_zeros), atol=1e-6)

def test_rmsnorm_shape_validation():
    """Test shape validation in RMS normalization."""
    layer = RMSNorm(4, dtype=jnp.float32)
    x_correct = jnp.ones((2, 4), dtype=jnp.float32)
    output = layer(x_correct)
    assert output.shape == (2, 4)
    x_wrong = jnp.ones((2, 6), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Expected hidden_size to be 4, but found: 6"):
        layer(x_wrong)

def test_rmsnorm_different_eps():
    """Test RMS normalization with different epsilon values."""
    hidden_size = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    for eps in [1e-6, 1e-5, 1e-4]:
        layer = RMSNorm(hidden_size, eps=eps, dtype=jnp.float32)
        layer.weight.value = jnp.ones(hidden_size, dtype=jnp.float32)
        output = layer(x)
        expected = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
        np.testing.assert_allclose(output, expected, atol=1e-6)

def test_rmsnorm_no_weight():
    """Test RMS normalization without weight parameter."""
    hidden_size = 4
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    layer = RMSNorm(hidden_size, has_weight=False, dtype=jnp.float32)
    output = layer(x)
    expected = x / jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + layer.variance_epsilon)
    np.testing.assert_allclose(output, expected, atol=1e-6)
