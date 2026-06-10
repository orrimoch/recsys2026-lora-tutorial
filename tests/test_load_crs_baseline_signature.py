"""Guard: load_crs_baseline() must accept every kwarg run_inference_blindset.py
passes it. The wrapper signature has drifted behind the driver twice (extra_config,
then sasrec_context_use_goal) -> a TypeError that only surfaces at Blind-inference
time. This statically parses the driver's call and asserts the wrapper covers it.
"""
import ast
import inspect
from pathlib import Path

from mcrs import load_crs_baseline

DRIVER = Path(__file__).resolve().parents[1] / "music-crs-baselines" / "run_inference_blindset.py"


def _driver_call_kwargs() -> set[str]:
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "load_crs_baseline"):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError("no load_crs_baseline(...) call found in run_inference_blindset.py")


def test_wrapper_accepts_all_driver_kwargs():
    sig = set(inspect.signature(load_crs_baseline).parameters)
    missing = _driver_call_kwargs() - sig
    assert not missing, f"load_crs_baseline missing kwargs the driver passes: {sorted(missing)}"


def test_sasrec_context_use_goal_is_accepted():
    assert "sasrec_context_use_goal" in inspect.signature(load_crs_baseline).parameters
