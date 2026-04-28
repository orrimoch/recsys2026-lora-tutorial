"""vLLM-backed LM module for high-throughput batch inference.

Drop-in replacement for LLAMA_MODEL with the same .batch_response_generation
and .batch_response_generation_multi interface. Internally uses vLLM's
PagedAttention + continuous batching for ~5-10x throughput on the
8000-row dev-set workload vs vanilla HF generate.

Lazy-imported by the lm_modules factory only when `use_vllm: true` is
set in the yaml config — keeps non-vLLM runs from paying vLLM's
~30-second cold start and dependency footprint.
"""
import os
from typing import Optional, List, Dict
import torch


_MAX_INPUT_TOKENS = 2048


def _truncate_to_budget(tokenizer, messages: List[Dict], max_tokens: int) -> List[Dict]:
    """Drop oldest history turns until the chat-template-rendered prompt
    fits in `max_tokens`. Mirrors the logic in LLAMA_MODEL._format_chat_history
    so both backends apply the same truncation policy.

    `messages` is expected to be [system, ...history, last_msg]. We never
    drop the first (system) or the last (recommend_item / final assistant
    turn) — both are critical to response correctness. Only the middle
    history range is shrunk.
    """
    if len(messages) <= 2:
        return messages
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    if n_tokens <= max_tokens:
        return messages

    system_msg = messages[0]
    last_msg = messages[-1]
    history = list(messages[1:-1])
    while history and n_tokens > max_tokens:
        history.pop(0)
        truncated = [system_msg] + history + [last_msg]
        rendered = tokenizer.apply_chat_template(truncated, tokenize=False, add_generation_prompt=True)
        n_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    return [system_msg] + history + [last_msg]


class VLLM_MODEL:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        dtype: torch.dtype = torch.bfloat16,
        max_model_len: int = 2304,
        gpu_memory_utilization: float = 0.7,
    ):
        # vLLM is CUDA-only in practice (CPU mode exists but is too slow
        # for our workload). Fail fast on MPS/CPU rather than silently
        # crash inside vLLM's loader.
        if device != "cuda":
            raise ValueError(
                f"VLLM_MODEL requires device='cuda' (got {device!r}). "
                "Use LLAMA_MODEL for MPS/CPU."
            )
        # `attn_implementation` is accepted but ignored — vLLM picks its
        # own attention backend (Flash Attention 2 by default on Ampere+).
        self.model_name = model_name
        self.dtype = dtype

        from vllm import LLM  # lazy import — vllm is ~1 GB install

        dtype_str = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }.get(dtype, "auto")

        # gpu_memory_utilization can be overridden via env var, useful when
        # Colab has leaked GPU memory from prior failed runs. Lower it (e.g.
        # 0.5) to fit in less free memory; or restart the Colab runtime.
        env_util = os.environ.get("VLLM_GPU_MEM_UTIL")
        if env_util:
            gpu_memory_utilization = float(env_util)
            print(f"[VLLM_MODEL] gpu_memory_utilization overridden via env to {gpu_memory_utilization}")

        # max_model_len = max input + max output. Default 2304 = our 2048
        # input cap + 256 generation budget (covers max_new_tokens=192 in
        # 029/030/031 plus headroom).
        self.lm = LLM(
            model=model_name,
            dtype=dtype_str,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
            trust_remote_code=False,
        )
        self.tokenizer = self.lm.get_tokenizer()
        self._max_input_tokens = max_model_len - 256  # leave room for gen

    def _build_messages(
        self,
        sys_prompt: str,
        chat_history: List[Dict],
        recommend_item: str,
    ) -> List[Dict]:
        msgs = [{"role": "system", "content": sys_prompt}]
        msgs += list(chat_history)
        msgs += [{"role": "assistant", "content": recommend_item}]
        return _truncate_to_budget(self.tokenizer, msgs, self._max_input_tokens)

    def batch_response_generation(
        self,
        sys_prompts: List[str],
        chat_histories: List[List[Dict]],
        recommend_items: List[str],
        max_new_tokens: int = 64,
    ) -> List[str]:
        """Greedy batched generation. Drop-in replacement for the
        LLAMA_MODEL method of the same name.
        """
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        message_batches = [
            self._build_messages(sp, ch, rec)
            for sp, ch, rec in zip(sys_prompts, chat_histories, recommend_items)
        ]
        outputs = self.lm.chat(message_batches, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def batch_response_generation_multi(
        self,
        sys_prompts: List[str],
        chat_histories: List[List[Dict]],
        recommend_items: List[str],
        max_new_tokens: int = 192,
        temperatures: Optional[List[float]] = None,
    ) -> List[List[str]]:
        """Sample K responses per query via temperature sweep. Used by
        the response-reranker pipeline (exp 026 + later).
        """
        from vllm import SamplingParams

        if temperatures is None:
            temperatures = [0.3, 0.7, 1.0]
        message_batches = [
            self._build_messages(sp, ch, rec)
            for sp, ch, rec in zip(sys_prompts, chat_histories, recommend_items)
        ]
        n = len(message_batches)
        results: List[List[str]] = [[] for _ in range(n)]
        for temp in temperatures:
            sampling = SamplingParams(
                temperature=max(temp, 0.0),
                top_p=0.9 if temp > 0 else 1.0,
                max_tokens=max_new_tokens,
            )
            outs = self.lm.chat(message_batches, sampling, use_tqdm=False)
            for i, o in enumerate(outs):
                results[i].append(o.outputs[0].text)
        return results
