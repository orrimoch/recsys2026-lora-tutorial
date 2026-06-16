from typing import Optional

from .llama import LLAMA_MODEL


def load_lm_module(
    lm_type,
    device,
    attn_implementation,
    dtype,
    use_vllm: bool = False,
    # W4 review P0 #3: PEFT/LoRA adapter on top of the base model. Set this
    # to a HF repo id or local dir from a KTO/DPO/GRPO run to load the
    # adapter at inference. Both backends support it:
    #   LLAMA_MODEL — wraps with PeftModel.from_pretrained after base load.
    #   VLLM_MODEL  — vLLM's enable_lora=True + per-request LoRARequest.
    lora_path: Optional[str] = None,
    lora_max_rank: int = 32,
):
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
            lora_path=lora_path, lora_max_rank=lora_max_rank,
        )
    # LLAMA_MODEL uses AutoTokenizer/AutoModelForCausalLM — works with any causal LM.
    # LoRA support: if lora_path is set, wrap the loaded model with PeftModel.
    lm = LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    if lora_path:
        from peft import PeftModel
        lm.lm = PeftModel.from_pretrained(lm.lm, lora_path).eval()
        print(f"[LLAMA_MODEL] loaded LoRA adapter from {lora_path}")
    return lm
