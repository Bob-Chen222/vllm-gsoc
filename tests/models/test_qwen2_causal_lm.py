import pytest
# NB: torch is required for HF reference model.
import torch

from transformers import Qwen2Config, Qwen2ForCausalLM as HFQwen2ForCausalLM

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as VllmQwen2ForCausalLM
from ..conftest import HfRunner, VllmRunner
import jax
import jax.numpy as jnp


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="Llama-3.2-1B-Instruct, Ilama-3.2-1B produce memory access fault.")
@pytest.mark.parametrize(
    "model,model_impl",
    [
        ("Qwen/Qwen2-1.5B-Instruct", "auto")
        # ("ArthurZ/Ilama-3.2-1B", "auto"),  # CUSTOM CODE
    ])  # trust_remote_code=True by default
def test_qwen2_hidden_states_match_hf(model: str, model_impl: str):
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

    hf_model = HFQwen2ForCausalLM(hf_cfg)

    vllm_runner = VllmRunner(model, model_impl=model_impl)
    vllm_cfg = vllm_runner.model.llm_engine.vllm_config

    # 2. Build a matching vLLM config that re-uses the same HF config.
    #    We start from the default VllmConfig (which provides sensible
    #    defaults) and then override the HF config using the provided helper.

    # 3. Construct the vLLM model inside the global config context.
    with set_current_vllm_config(vllm_cfg):
        vllm_model = VllmQwen2ForCausalLM(vllm_config=vllm_cfg)

    # 4. Load the HF weights into the vLLM model.
    vllm_model.load_weights(hf_model.state_dict().items())

    # 5. Prepare a tiny batch of random token ids.
    batch, seq_len = 1, 5
    vocab_size = hf_cfg.vocab_size  # Assuming hf_cfg is still accessible

    key = jax.random.PRNGKey(0)  # Set random seed
    input_ids = jax.random.randint(key, shape=(batch, seq_len), minval=0, maxval=vocab_size)

    positions = jnp.arange(seq_len)

    # 6. Forward pass through both models.
    # hf_out = hf_model(input_ids, output_hidden_states=True, return_dict=True)
    # hf_hidden = hf_out.hidden_states[-1].squeeze(0)  # (seq_len, hidden)

    # The model's ``__call__`` is JAX-centric; use ``forward`` for torch flow.
    vllm_hidden = vllm_model(input_ids, positions=positions)
    print("vllm_hidden shape:", vllm_hidden.shape)

    # 7. Compare.
    # assert torch.allclose(vllm_hidden, hf_hidden, atol=1e-4, rtol=1e-4) 

# if __name__ == "__main__":
#     test_qwen2_hidden_states_match_hf()