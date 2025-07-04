# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the functionality of the Transformers backend."""
from typing import Any, Optional, Union

import pytest

from vllm.platforms import current_platform

from ..conftest import HfRunner, VllmRunner
from ..core.block.e2e.test_correctness_sliding_window import prep_prompts
from ..utils import multi_gpu_test
from .utils import check_logprobs_close


def check_implementation(
    runner_ref: type[Union[HfRunner, VllmRunner]],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: Optional[dict[str, Any]] = None,
    kwargs_test: Optional[dict[str, Any]] = None,
    **kwargs,
):
    if kwargs_ref is None:
        kwargs_ref = {}
    if kwargs_test is None:
        kwargs_test = {}

    max_tokens = 32
    num_logprobs = 5

    args = (example_prompts, max_tokens, num_logprobs)

    with runner_ref(model, **kwargs_ref) as model_ref:
        if isinstance(model_ref, VllmRunner):
            outputs_ref = model_ref.generate_greedy_logprobs(*args)
        else:
            outputs_ref = model_ref.generate_greedy_logprobs_limit(*args)
            
    with runner_test(model, **kwargs_test, **kwargs) as model_test:
        outputs_test = model_test.generate_greedy_logprobs(*args)


    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="ref",
        name_1="test",
    )


@pytest.mark.skipif(
    current_platform.is_rocm(),
    reason="Llama-3.2-1B-Instruct, Ilama-3.2-1B produce memory access fault.")
@pytest.mark.parametrize(
    "model,model_impl",
    [
        ("Qwen/Qwen2-1.5B-Instruct", "auto")
        # ("ArthurZ/Ilama-3.2-1B", "auto"),  # CUSTOM CODE
    ])  # trust_remote_code=True by default
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
) -> None:
    check_implementation(hf_runner,
                         vllm_runner,
                         example_prompts,
                         model,
                         model_impl=model_impl)