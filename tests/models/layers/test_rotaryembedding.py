import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


# Helper function for reference implementation
def reference_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    base: float,
    is_neox_style: bool,
    rotary_dim: int,
    positions: torch.Tensor,
):
    inv_freq = 1.0 / (base**(
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    t = positions.float()
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    def apply_rotary(x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        cos_exp = cos.unsqueeze(-2).to(dtype)
        sin_exp = sin.unsqueeze(-2).to(dtype)

        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]

        if is_neox_style:
            x1, x2 = torch.chunk(x_rot, 2, dim=-1)
            o1 = x1 * cos_exp - x2 * sin_exp
            o2 = x2 * cos_exp + x1 * sin_exp
            x_rot = torch.cat((o1, o2), dim=-1)
        else:
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            o1 = x1 * cos_exp - x2 * sin_exp
            o2 = x2 * cos_exp + x1 * sin_exp
            x_rot = torch.stack((o1, o2), dim=-1).flatten(-2)

        if x_pass.shape[-1] > 0:
            return torch.cat((x_rot, x_pass), dim=-1)
        else:
            return x_rot

    rotated_query = apply_rotary(query)
    rotated_key = apply_rotary(key)
    return rotated_query, rotated_key


@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim_fraction", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("seed", [0])
def test_rotary_embedding(
    is_neox_style: bool,
    rotary_dim_fraction: float,
    dtype: torch.dtype,
    seed: int,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    head_size = 64
    rotary_dim = int(head_size * rotary_dim_fraction)
    max_position_embeddings = 4096
    base = 10000.0
    num_tokens = 128
    num_heads = 4

    rotary_emb = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    ).to(dtype)

    query_shape = (num_tokens, num_heads, head_size)
    key_shape = (num_tokens, num_heads, head_size)
    query = torch.randn(query_shape, dtype=dtype)
    key = torch.randn(key_shape, dtype=dtype)
    positions = torch.randint(0, max_position_embeddings, (num_tokens, ))

    # vLLM implementation
    # The forward path for RotaryEmbedding doesn't use CUDA ops unless on GPU
    # and we want to test the torch implementation here
    # so we will call forward_native directly.
    # The query and key are reshaped inside forward_native,
    # so we pass them with the correct shape.
    vllm_query, vllm_key = rotary_emb.forward_native(
        positions=positions,
        query=query.view(num_tokens, num_heads * head_size),
        key=key.view(num_tokens, num_heads * head_size))
    vllm_query = vllm_query.view(query_shape)
    vllm_key = vllm_key.view(key_shape)

    # Reference implementation
    ref_query, ref_key = reference_rotary_embedding(
        query=query,
        key=key,
        head_size=head_size,
        base=base,
        is_neox_style=is_neox_style,
        rotary_dim=rotary_dim,
        positions=positions,
    )

    # Compare
    assert torch.allclose(vllm_query, ref_query, atol=1e-3, rtol=1e-5)
    assert torch.allclose(vllm_key, ref_key, atol=1e-3, rtol=1e-5) 