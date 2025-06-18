import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp

from vllm.model_executor.layers.layernorm import RMSNorm as TorchRMSNorm
from vllm.model_executor.layers.layernorm_jax import RMSNorm as JaxRMSNorm

def _init_weights_np(hidden_size, dtype=np.float32):
    return np.random.normal(loc=1.0, scale=0.1, size=(hidden_size,)).astype(dtype)

def _make_inputs(num_tokens, hidden_size, dtype):
    np_dtype = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # jax.bfloat16 is not always available, so use float32 for comparison
    }[dtype]
    x = np.random.randn(num_tokens, hidden_size).astype(np_dtype) * (1 / (2 * hidden_size))
    return x

@pytest.mark.parametrize("num_tokens", [4, 16])
@pytest.mark.parametrize("hidden_size", [8, 32])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_jax_vs_torch(num_tokens, hidden_size, add_residual, dtype):
    np.random.seed(42)
    torch.manual_seed(42)
    key = jax.random.PRNGKey(42)

    # Prepare input and weight
    x_np = _make_inputs(num_tokens, hidden_size, dtype)
    weight_np = _init_weights_np(hidden_size, dtype=np.float32)
    residual_np = _make_inputs(num_tokens, hidden_size, dtype) if add_residual else None

    # PyTorch
    torch_x = torch.tensor(x_np, dtype=dtype)
    torch_weight = torch.tensor(weight_np, dtype=dtype)
    torch_layer = TorchRMSNorm(hidden_size, dtype=dtype)
    with torch.no_grad():
        torch_layer.weight.copy_(torch_weight)
    torch_residual = torch.tensor(residual_np, dtype=dtype) if add_residual else None
    torch_out = torch_layer.forward_native(torch_x, torch_residual)
    if isinstance(torch_out, tuple):
        torch_out = tuple(t.detach().cpu().numpy() for t in torch_out)
    else:
        torch_out = torch_out.detach().cpu().numpy()

    # JAX
    jax_x = jnp.array(x_np, dtype=jnp.float32)
    jax_weight = jnp.array(weight_np, dtype=jnp.float32)
    jax_layer = JaxRMSNorm(hidden_size, dtype=jnp.float32)
    # Set the weight parameter
    jax_layer.weight.value = jax_weight
    jax_residual = jnp.array(residual_np, dtype=jnp.float32) if add_residual else None
    jax_out = jax_layer(jax_x, jax_residual)
    if isinstance(jax_out, tuple):
        jax_out = tuple(np.array(t) for t in jax_out)
    else:
        jax_out = np.array(jax_out)

    # Compare outputs
    if isinstance(torch_out, tuple):
        assert len(torch_out) == len(jax_out)
        for t, j in zip(torch_out, jax_out):
            assert t.shape == j.shape
            np.testing.assert_allclose(t, j, atol=1e-4, rtol=1e-4)
    else:
        assert torch_out.shape == jax_out.shape
        np.testing.assert_allclose(torch_out, jax_out, atol=1e-4, rtol=1e-4)


def test_rmsnorm_batched_3d():
    # Test 3D input (batch, seq, hidden)
    np.random.seed(0)
    batch, seq, hidden = 2, 3, 5
    x_np = np.random.randn(batch, seq, hidden).astype(np.float32)
    weight_np = np.random.normal(1.0, 0.1, size=(hidden,)).astype(np.float32)
    torch_x = torch.tensor(x_np, dtype=torch.float32)
    torch_weight = torch.tensor(weight_np, dtype=torch.float32)
    torch_layer = TorchRMSNorm(hidden, dtype=torch.float32)
    with torch.no_grad():
        torch_layer.weight.copy_(torch_weight)
    torch_out = torch_layer.forward_native(torch_x)
    torch_out = torch_out.detach().cpu().numpy()
    jax_x = jnp.array(x_np, dtype=jnp.float32)
    jax_weight = jnp.array(weight_np, dtype=jnp.float32)
    jax_layer = JaxRMSNorm(hidden, dtype=jnp.float32)
    jax_layer.weight.value = jax_weight
    jax_out = jax_layer(jax_x)
    jax_out = np.array(jax_out)
    assert torch_out.shape == jax_out.shape
    np.testing.assert_allclose(torch_out, jax_out, atol=1e-4, rtol=1e-4)


def test_rmsnorm_hidden_size_1():
    # Edge case: hidden_size=1
    x_np = np.random.randn(7, 1).astype(np.float32)
    weight_np = np.random.normal(1.0, 0.1, size=(1,)).astype(np.float32)
    torch_x = torch.tensor(x_np, dtype=torch.float32)
    torch_weight = torch.tensor(weight_np, dtype=torch.float32)
    torch_layer = TorchRMSNorm(1, dtype=torch.float32)
    with torch.no_grad():
        torch_layer.weight.copy_(torch_weight)
    torch_out = torch_layer.forward_native(torch_x)
    torch_out = torch_out.detach().cpu().numpy()
    jax_x = jnp.array(x_np, dtype=jnp.float32)
    jax_weight = jnp.array(weight_np, dtype=jnp.float32)
    jax_layer = JaxRMSNorm(1, dtype=jnp.float32)
    jax_layer.weight.value = jax_weight
    jax_out = jax_layer(jax_x)
    jax_out = np.array(jax_out)
    assert torch_out.shape == jax_out.shape
    np.testing.assert_allclose(torch_out, jax_out, atol=1e-4, rtol=1e-4)


def test_rmsnorm_large_hidden_size():
    # Large hidden size
    np.random.seed(1)
    num_tokens, hidden = 2, 1024
    x_np = np.random.randn(num_tokens, hidden).astype(np.float32)
    weight_np = np.random.normal(1.0, 0.1, size=(hidden,)).astype(np.float32)
    torch_x = torch.tensor(x_np, dtype=torch.float32)
    torch_weight = torch.tensor(weight_np, dtype=torch.float32)
    torch_layer = TorchRMSNorm(hidden, dtype=torch.float32)
    with torch.no_grad():
        torch_layer.weight.copy_(torch_weight)
    torch_out = torch_layer.forward_native(torch_x)
    torch_out = torch_out.detach().cpu().numpy()
    jax_x = jnp.array(x_np, dtype=jnp.float32)
    jax_weight = jnp.array(weight_np, dtype=jnp.float32)
    jax_layer = JaxRMSNorm(hidden, dtype=jnp.float32)
    jax_layer.weight.value = jax_weight
    jax_out = jax_layer(jax_x)
    jax_out = np.array(jax_out)
    assert torch_out.shape == jax_out.shape
    np.testing.assert_allclose(torch_out, jax_out, atol=1e-4, rtol=1e-4)


def test_rmsnorm_no_weight():
    # Test has_weight=False
    np.random.seed(2)
    num_tokens, hidden = 3, 7
    x_np = np.random.randn(num_tokens, hidden).astype(np.float32)
    torch_x = torch.tensor(x_np, dtype=torch.float32)
    torch_layer = TorchRMSNorm(hidden, has_weight=False, dtype=torch.float32)
    torch_out = torch_layer.forward_native(torch_x)
    torch_out = torch_out.detach().cpu().numpy()
    jax_x = jnp.array(x_np, dtype=jnp.float32)
    jax_layer = JaxRMSNorm(hidden, has_weight=True, dtype=jnp.float32)  # JAX always has weight param, set to 1
    jax_layer.weight.value = jnp.ones(hidden, dtype=jnp.float32)
    jax_out = jax_layer(jax_x)
    jax_out = np.array(jax_out)
    assert torch_out.shape == jax_out.shape
    np.testing.assert_allclose(torch_out, jax_out, atol=1e-4, rtol=1e-4)


def test_rmsnorm_shape_mismatch():
    # Test error on shape mismatch
    x_np = np.random.randn(2, 5).astype(np.float32)
    torch_layer = TorchRMSNorm(4, dtype=torch.float32)
    jax_layer = JaxRMSNorm(4, dtype=jnp.float32)
    torch_x = torch.tensor(x_np, dtype=torch.float32)
    jax_x = jnp.array(x_np, dtype=jnp.float32)
    with pytest.raises(ValueError):
        torch_layer.forward_native(torch_x)
    with pytest.raises(ValueError):
        jax_layer(jax_x) 