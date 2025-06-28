# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A basic correctness check for TPUs

Run `python test_basic.py`.
"""
from __future__ import annotations

import os

from torch_xla._internal import tpu
from vllm.platforms import current_platform

# Replace with the actual VllmRunner import
from tests.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen2-1.5B-Instruct",
    # "Qwen/Qwen2-7B-Instruct",  # Enable this with v6e
    # "meta-llama/Llama-3.1-8B",
]

TENSOR_PARALLEL_SIZES = [1]
MAX_NUM_REQS = [4]
MAX_TOKENS = 5

def test_basic_runner(
    vllm_runner: type[VllmRunner],
    model: str,
    max_tokens: int,
    tensor_parallel_size: int,
    max_num_seqs: int,
) -> None:
    seq_len = 32
    prompt = (
        "The next numbers of the sequence "
        + ", ".join(str(i) for i in range(seq_len))
        + " are:"
    )
    example_prompts = [prompt]

    os.environ["VLLM_USE_V1"] = "1"

    with vllm_runner(
        model,
        max_num_batched_tokens=64,
        max_model_len=256,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        output = vllm_outputs[0][1]
        assert "32" in output or "0, 1" in output, f"Unexpected output: {output}"
        print(f"Output: {output!r}")

def main():
    if not current_platform.is_tpu():
        print("This test is designed to run on a TPU.")
        return

    for model in MODELS:
        for tp_size in TENSOR_PARALLEL_SIZES:
            for max_num_seqs in MAX_NUM_REQS:
                test_basic_runner(VllmRunner, model, MAX_TOKENS, tp_size, max_num_seqs)

if __name__ == "__main__":
    main()
