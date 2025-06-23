import torch
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context


def test_attention_forward_identity():
    """Tests that the Attention layer returns the input unchanged when
    called with ``attn_metadata=None`` (warm-up pass).

    This exercises the complete eager path on CPU, including the custom
    operator registration and the forward context plumbing, without
    depending on GPU-specific kernels.
    """

    # 1. Create a minimal vLLM configuration and register it as the current
    #    config so that the Attention layer can add itself to the static
    #    forward context during construction.
    vllm_config = VllmConfig()

    with set_current_vllm_config(vllm_config):
        # Construct an Attention layer with a unique prefix so it does not
        # collide with other layers that might be built in the same test run.
        attn = Attention(
            num_heads=2,
            head_size=4,
            scale=1.0,
            prefix="test_attention_layer",
        )
        # The layer expects a per-virtual-engine KV-cache list.  For this unit
        # test we do not exercise KV-cache functionality, so a single empty
        # tensor placeholder is sufficient.
        attn.kv_cache = [torch.tensor([])]

    # 2. Prepare dummy input data.  The hidden size must equal
    #    ``num_heads * head_size``.
    num_tokens = 5
    hidden_size = attn.num_heads * attn.head_size
    query = torch.randn(num_tokens, hidden_size)

    # 3. Run the forward pass inside a forward context with ``attn_metadata``
    #    set to ``None`` – this triggers the warm-up path in the backend
    #    implementation, causing it to return the query tensor unchanged.
    with set_forward_context(attn_metadata=None, vllm_config=vllm_config):
        output = attn.forward(query, None, None)

    # 4. Validate basic properties of the output.
    assert output.shape == query.shape
    torch.testing.assert_close(output, query) 