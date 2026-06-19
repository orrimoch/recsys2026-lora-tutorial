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
        # R8 ColBERT late-interaction channel (47_R8 §9). Brute-force MaxSim at 47k by default.
        "colbert": {
            "model":           _Leaf(str,           "colbert-ir/colbertv2.0"),
            "model_revision":  _Leaf(type(None),    None),   # pinned commit (D1 records it)
            "dim":             _Leaf(int,           128),
            "query_maxlen":    _Leaf(int,           32),     # incl. [MASK] query augmentation
            "doc_maxlen":      _Leaf(int,           300),    # doc cap; trim raw tags, keep doc2query
            "expansion_first": _Leaf(bool,          True),   # 'expansion | base' so doc2query survives truncation (train==serve)
            "mask_punctuation": _Leaf(bool,         True),
            "bsize":           _Leaf(int,           32),
            "dtype":           _Leaf(str,           "auto"),
            "normalize":       _Leaf(bool,          True),   # cosine MaxSim (L2-normalize tokens)
            "chunk_docs":      _Leaf(int,           4096),   # brute-force memory-safe doc chunking
            "query_key":       _Leaf(str,           "colbert"),  # Query.per_channel focused query
            "topk_internal":   _Leaf(int,           500),    # >= fusion_K (R7)
            "weight":          _Leaf(float,         1.0),    # R7 RRF weight
        },
    },
    "eval": {
        "parity_atol": _Leaf(float, 1e-9),
        "distinct_n": _Leaf(int, 2),
    },
    "rerank": {
        "neural": {
            # LoRA adapter config (K3b §9)
            "lora": {
                "enabled":        _Leaf(bool,  False),
                "r":              _Leaf(int,   16),
                "alpha":          _Leaf(int,   32),
                "dropout":        _Leaf(float, 0.05),
                "target_modules": _Leaf(list,  ["query", "value"]),
            },
            # Model / tokenisation
            "max_length":       _Leaf(int,        2048),
            "max_doc_tokens":   _Leaf(int,        1100),
            "dtype":            _Leaf(str,        "auto"),
            "adapter_revision": _Leaf(type(None), None),
            # Negative sampling
            "negatives": {
                "n":               _Leaf(int,   15),
                "k_min":           _Leaf(int,   4),
                "sampling":        _Leaf(str,   "rank_strat"),
                "same_artist":     _Leaf(str,   "soft_downweight"),
                "denoise_near_dup": _Leaf(bool, True),
                "skip_top_rank":   _Leaf(bool,  False),
            },
            # Goal-progress positive weighting
            "goal_progress": {
                "enabled": _Leaf(bool,  False),
                "w_low":   _Leaf(float, 0.3),
            },
            # Out-of-fold stacking
            "oof": {
                "folds":           _Leaf(int,  3),
                "dedup_cross_fold": _Leaf(bool, True),
                "score_norm":      _Leaf(str,  "within_pool"),
            },
            # Training hyper-parameters
            "train": {
                "epochs":                   _Leaf(int,   3),
                "lr":                       _Leaf(float, 1e-4),
                "weight_decay":             _Leaf(float, 0.0),
                "batch_groups":             _Leaf(int,   2),
                "grad_accum":               _Leaf(int,   16),
                "warmup":                   _Leaf(float, 0.05),
                "early_stop_patience":      _Leaf(int,   1),
                "group_by_length":          _Leaf(bool,  True),
                "log_every":                _Leaf(int,   50),
                "seed":                     _Leaf(int,   0),
                "gradient_checkpointing":   _Leaf(bool,  True),
            },
            # Train/eval split config
            "split": {
                "key":           _Leaf(str,  "session"),
                "dedup_near_dup": _Leaf(bool, True),
            },
        },
    },
    "query": {
        "markers":     _Leaf(bool, True),
        "taste_items": _Leaf(int,  5),
    },
    "logging": {
        "trackio": {
            "enabled": _Leaf(bool, False),
            "project": _Leaf(str,  "recsys2026"),
        },
    },
}


def _validate_type(key: str, value: Any, leaf: _Leaf) -> Any:
    # Optional[str]: leaf.type is type(None) means "str or None".
    if leaf.type is type(None):
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"config key '{key}': expected str or None, got {type(value).__name__}"
            )
        return value
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
