"""F2 — experiment config: schema-validated loader with defaults + dotted access.

Loads a YAML file (or dict), applies defaults, validates types, and rejects unknown keys
so train and serve read one validated config. The schema grows as modules land; today it
covers the keys the foundation + early retrieval need. See feature doc §9.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Union

import yaml


class _Leaf:
    __slots__ = ("type", "default")

    def __init__(self, type_: type, default: Any) -> None:
        self.type = type_
        self.default = default


# Nested schema: dict => subsection, _Leaf => typed value with default.
SCHEMA: dict[str, Any] = {
    "seed": _Leaf(int, 42),
    "paths": {
        "data_root": _Leaf(str, "./data"),
        "cache_dir": _Leaf(str, "./cache"),
        "reports_dir": _Leaf(str, "./reports"),
    },
    "segment": {
        "cold_threshold": _Leaf(int, 1),
    },
    "retrieval": {
        "topk": _Leaf(int, 300),          # fusion-K (P0-sized)
        "fusion_k": _Leaf(int, 60),        # RRF k constant
    },
    "eval": {
        "parity_atol": _Leaf(float, 1e-9),
        "distinct_n": _Leaf(int, 2),
    },
}


def _validate_type(key: str, value: Any, leaf: _Leaf) -> Any:
    # bool is a subclass of int — reject it where an int/float is expected.
    if leaf.type in (int, float) and isinstance(value, bool):
        raise ValueError(f"config key '{key}': expected {leaf.type.__name__}, got bool")
    if leaf.type is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)  # accept ints for float fields
    if not isinstance(value, leaf.type):
        raise ValueError(
            f"config key '{key}': expected {leaf.type.__name__}, got {type(value).__name__}"
        )
    return value


def _merge(schema: dict, provided: dict, prefix: str = "") -> dict:
    if not isinstance(provided, dict):
        raise ValueError(f"config section '{prefix.rstrip('.')}' must be a mapping")
    for k in provided:
        if k not in schema:
            raise ValueError(f"unknown config key: '{prefix}{k}'")
    out: dict[str, Any] = {}
    for k, spec in schema.items():
        path = f"{prefix}{k}"
        if isinstance(spec, dict):
            out[k] = _merge(spec, provided.get(k, {}), prefix=path + ".")
        else:
            out[k] = _validate_type(path, provided[k], spec) if k in provided else spec.default
    return out


def _namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{
        k: (_namespace(v) if isinstance(v, dict) else v) for k, v in d.items()
    })


def load_config(src: Union[str, dict, None]) -> SimpleNamespace:
    """Load from a YAML path or a dict; return a validated dotted-access config."""
    if src is None:
        provided: dict = {}
    elif isinstance(src, str):
        with open(src) as f:
            provided = yaml.safe_load(f) or {}
    elif isinstance(src, dict):
        provided = src
    else:
        raise TypeError(f"load_config expects str|dict|None, got {type(src).__name__}")
    return _namespace(_merge(SCHEMA, provided))
