from .llama import LLAMA_MODEL


def load_lm_module(lm_type, device, attn_implementation, dtype, use_vllm: bool = False):
    """Load the LM backend.

    use_vllm=True routes to VLLM_MODEL — ~5-10x throughput on batched
    inference via PagedAttention + continuous batching, but CUDA-only
    and adds ~30 sec cold start. Default LLAMA_MODEL is the vanilla HF
    generate path that works on MPS/CPU/CUDA.

    The `attn_implementation` argument is honoured by LLAMA_MODEL only;
    VLLM_MODEL picks its own backend (Flash Attention 2 on Ampere+).
    """
    if use_vllm:
        if device != "cuda":
            raise ValueError(
                f"use_vllm=True requires device='cuda'; got {device!r}"
            )
        # Lazy import — vllm is a ~1 GB install we shouldn't pull in for
        # the M4-local LLAMA_MODEL path.
        from .vllm_model import VLLM_MODEL
        return VLLM_MODEL(
            model_name=lm_type, device=device,
            attn_implementation=attn_implementation, dtype=dtype,
        )
    # LLAMA_MODEL uses AutoTokenizer/AutoModelForCausalLM — works with any causal LM
    return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
