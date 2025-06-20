import pytest
import jax
import jax.numpy as jnp
import numpy as np
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
)

# Mock tensor_model_parallel_all_gather for gather_output test
import vllm.model_executor.layers.linear as linear_mod

def test_forward_basic():
    input_size = 8
    output_size = 4
    layer = ColumnParallelLinear(input_size, output_size, bias=True, params_dtype=jnp.float32)
    # Set weights and bias to known values
    key = jax.random.PRNGKey(0)
    w_key, b_key, x_key = jax.random.split(key, 3)
    weight = jax.random.normal(w_key, (output_size, input_size), dtype=jnp.float32)
    bias = jax.random.normal(b_key, (output_size,), dtype=jnp.float32)
    x = jax.random.normal(x_key, (2, input_size), dtype=jnp.float32)
    layer.weight.value = weight
    layer.bias.value = bias
    # Reference output
    ref = x @ weight.T + bias
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)

def test_forward_no_bias():
    input_size = 8
    output_size = 4
    layer = ColumnParallelLinear(input_size, output_size, bias=False, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(1)
    w_key, x_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (output_size, input_size), dtype=jnp.float32)
    x = jax.random.normal(x_key, (2, input_size), dtype=jnp.float32)
    layer.weight.value = weight
    ref = x @ weight.T
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)
    assert getattr(layer, 'bias', None) is None

def test_skip_bias_add():
    input_size = 8
    output_size = 4
    layer = ColumnParallelLinear(input_size, output_size, bias=True, skip_bias_add=True, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(2)
    w_key, b_key, x_key = jax.random.split(key, 3)
    weight = jax.random.normal(w_key, (output_size, input_size), dtype=jnp.float32)
    bias = jax.random.normal(b_key, (output_size,), dtype=jnp.float32)
    x = jax.random.normal(x_key, (2, input_size), dtype=jnp.float32)
    layer.weight.value = weight
    layer.bias.value = bias
    out, bias_out = layer(x)
    ref = x @ weight.T
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(bias_out), np.array(bias), rtol=1e-5, atol=1e-5)

def test_parameter_shapes():
    input_size = 7
    output_size = 5
    layer = ColumnParallelLinear(input_size, output_size, bias=True, params_dtype=jnp.float32)
    assert layer.weight.value.shape == (output_size, input_size)
    assert layer.bias.value.shape == (output_size,)
    layer_nobias = ColumnParallelLinear(input_size, output_size, bias=False, params_dtype=jnp.float32)
    assert getattr(layer_nobias, 'bias', None) is None

@pytest.mark.parametrize("x", [
    jax.random.normal(jax.random.PRNGKey(101), (2, 8), dtype=jnp.float32),
    jax.random.normal(jax.random.PRNGKey(102), (4, 8), dtype=jnp.float32),
    jnp.zeros((3, 8), dtype=jnp.float32),
])
def test_merged_column_parallel_linear_forward(x):
    input_size = 8
    output_sizes = [4, 3, 2]
    layer = MergedColumnParallelLinear(input_size, output_sizes, bias=True, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(10)
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (sum(output_sizes), input_size), dtype=jnp.float32)
    bias = jax.random.normal(b_key, (sum(output_sizes),), dtype=jnp.float32)
    layer.weight.value = weight
    layer.bias.value = bias
    ref = x @ weight.T + bias
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("x", [
    jax.random.normal(jax.random.PRNGKey(111), (2, 8), dtype=jnp.float32),
    jax.random.normal(jax.random.PRNGKey(112), (5, 8), dtype=jnp.float32),
    jnp.ones((1, 8), dtype=jnp.float32),
])
def test_merged_column_parallel_linear_no_bias(x):
    input_size = 8
    output_sizes = [4, 3, 2]
    layer = MergedColumnParallelLinear(input_size, output_sizes, bias=False, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(11)
    w_key, _ = jax.random.split(key)
    weight = jax.random.normal(w_key, (sum(output_sizes), input_size), dtype=jnp.float32)
    layer.weight.value = weight
    ref = x @ weight.T
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)
    assert getattr(layer, 'bias', None) is None

@pytest.mark.parametrize("x", [
    jax.random.normal(jax.random.PRNGKey(121), (2, 8), dtype=jnp.float32),
    jax.random.normal(jax.random.PRNGKey(122), (3, 8), dtype=jnp.float32),
    jnp.full((1, 8), 2.0, dtype=jnp.float32),
])
def test_qkv_parallel_linear_forward(x):
    hidden_size = 8
    head_size = 2
    total_num_heads = 3
    total_num_kv_heads = 2
    layer = QKVParallelLinear(hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=True, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(12)
    w_key, b_key = jax.random.split(key)
    output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
    weight = jax.random.normal(w_key, (output_size, hidden_size), dtype=jnp.float32)
    bias = jax.random.normal(b_key, (output_size,), dtype=jnp.float32)
    layer.weight.value = weight
    layer.bias.value = bias
    ref = x @ weight.T + bias
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("x", [
    jax.random.normal(jax.random.PRNGKey(131), (2, 8), dtype=jnp.float32),
    jax.random.normal(jax.random.PRNGKey(132), (6, 8), dtype=jnp.float32),
    jnp.arange(10, 50, dtype=jnp.float32).reshape(4, 10)[:, :8],
])
def test_row_parallel_linear_forward(x):
    input_size = 8
    output_size = 5
    layer = RowParallelLinear(input_size, output_size, bias=True, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(13)
    w_key, b_key = jax.random.split(key)
    weight = jax.random.normal(w_key, (output_size, input_size), dtype=jnp.float32)
    bias = jax.random.normal(b_key, (output_size,), dtype=jnp.float32)
    layer.weight.value = weight
    layer.bias.value = bias
    ref = x @ weight.T + bias
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("x", [
    jax.random.normal(jax.random.PRNGKey(141), (2, 8), dtype=jnp.float32),
    jax.random.normal(jax.random.PRNGKey(142), (1, 8), dtype=jnp.float32),
    jnp.zeros((5, 8), dtype=jnp.float32),
])
def test_row_parallel_linear_no_bias(x):
    input_size = 8
    output_size = 5
    layer = RowParallelLinear(input_size, output_size, bias=False, params_dtype=jnp.float32)
    key = jax.random.PRNGKey(14)
    w_key, _ = jax.random.split(key)
    weight = jax.random.normal(w_key, (output_size, input_size), dtype=jnp.float32)
    layer.weight.value = weight
    ref = x @ weight.T
    out, _ = layer(x)
    np.testing.assert_allclose(np.array(out), np.array(ref), rtol=1e-5, atol=1e-5)
    assert getattr(layer, 'bias', None) is None 