"""EXP-008 efficiency: encoder load dtype selection for the 47k-catalog pass.

fp16 on CUDA ~halves weight memory and ~doubles throughput — the difference
between a 4B encoder fitting a 16GB T4/G4 vs OOM-ing in fp32. fp16 (not bf16) is
deliberate: Turing/T4 has no bf16. CPU/MPS stay fp32. An explicit override forces
a dtype on any device. Pure function — no model load, no GPU needed.
"""
import torch

from mcrs.retrieval_modules.dense_local import resolve_st_dtype


def test_auto_picks_fp16_on_cuda():
    assert resolve_st_dtype("cuda") == torch.float16


def test_auto_stays_fp32_on_cpu_and_mps():
    assert resolve_st_dtype("cpu") == torch.float32
    assert resolve_st_dtype("mps") == torch.float32


def test_explicit_override_wins_on_any_device():
    # forcing fp32 on cuda (e.g. to debug a precision issue) must be honored
    assert resolve_st_dtype("cuda", "float32") == torch.float32
    # forcing fp16 on cpu must be honored too (override beats the auto default)
    assert resolve_st_dtype("cpu", "float16") == torch.float16
    assert resolve_st_dtype("cuda", "bfloat16") == torch.bfloat16


def test_auto_string_is_treated_as_auto():
    assert resolve_st_dtype("cuda", "auto") == torch.float16
