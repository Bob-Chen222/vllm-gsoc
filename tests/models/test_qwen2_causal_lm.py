import pytest
import torch

from transformers import Qwen2Config, Qwen2ForCausalLM as HFQwen2ForCausalLM

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import initialize_model_parallel
from vllm.platforms import current_platform
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as VllmQwen2ForCausalLM


@pytest.mark.skipif(current_platform.is_cpu(), reason="This test requires a CUDA device for NCCL backend.")
@pytest.mark.usefixtures("dist_init")
@torch.inference_mode()
def test_qwen2_hidden_states_match_hf():
    """Ensure vLLM's Qwen2ForCausalLM matches HuggingFace implementation.

    We build a tiny Qwen-2 model (2 layers, hidden size 16) so the test runs
    quickly.  After copying the HF weights into the vLLM model, the final
    hidden states for the same token ids must be identical (modulo numerical
    noise).
    """

    # 1. Define a minimal HF Qwen-2 configuration.
    hf_cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=32,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )

    hf_model = HFQwen2ForCausalLM(hf_cfg).eval()

    # 2. Build a matching vLLM config that re-uses the same HF config.
    #    We start from the default VllmConfig (which provides sensible
    #    defaults) and then override the HF config using the provided helper.
    vllm_cfg = VllmConfig()
    vllm_cfg = vllm_cfg.with_hf_config(hf_cfg, architectures=["Qwen2ForCausalLM"])

    # 3. Construct the vLLM model inside the global config context.
    with set_current_vllm_config(vllm_cfg):
        vllm_model = VllmQwen2ForCausalLM(vllm_config=vllm_cfg).eval()

    # 4. Load the HF weights into the vLLM model.
    vllm_model.load_weights(hf_model.state_dict().items())

    # 5. Prepare a tiny batch of random token ids.
    batch, seq_len = 1, 5
    input_ids = torch.randint(0, hf_cfg.vocab_size, (batch, seq_len))
    positions = torch.arange(seq_len)

    # 6. Forward pass through both models.
    hf_out = hf_model(input_ids, output_hidden_states=True, return_dict=True)
    hf_hidden = hf_out.hidden_states[-1].squeeze(0)  # (seq_len, hidden)

    vllm_hidden = vllm_model(input_ids.squeeze(0), positions)

    # 7. Compare.
    assert torch.allclose(vllm_hidden, hf_hidden, atol=1e-4, rtol=1e-4) 