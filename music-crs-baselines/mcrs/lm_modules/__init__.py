from .llama import LLAMA_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    # LLAMA_MODEL uses AutoTokenizer/AutoModelForCausalLM — works with any causal LM
    return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
