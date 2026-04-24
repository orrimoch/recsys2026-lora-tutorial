#!/usr/bin/env python
"""End-to-end experiment orchestrator: infer -> validate -> eval|package -> log (plan §4.2, W0-2/W0-3)."""

# NOTE ON SMOKE MODE: the existing `run_inference_{devset,blindset}.py` scripts do not
# expose a --subset / --session-limit flag. We implement `--smoke` by invoking the
# inference module via a short `python -c` subprocess that monkey-patches
# `datasets.load_dataset` to return only the first 5 sessions before calling the
# module's `main()`. This gives a real speed benefit without editing the baseline
# scripts. If that patching path ever breaks, fall back by running full inference
# and truncating the resulting JSON to the first 40 rows.

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
BASELINES_DIR = REPO_ROOT / "music-crs-baselines"
INFERENCE_DIR = BASELINES_DIR / "exp" / "inference"
LOGS_DIR = BASELINES_DIR / "exp" / "logs"
EXPERIMENTS_LOG = REPO_ROOT / "documents" / "experiments_log.md"
SUBMISSIONS_LOG = REPO_ROOT / "documents" / "submissions_log.md"

# Map the orchestrator's --split to the inference script + output subdir layout.
SPLIT_CONFIG = {
    "dev": {
        "script": "run_inference_devset.py",
        "out_subdir": "devset",
        "eval_dataset_flag": None,  # dev script has no --eval_dataset
    },
    "blindA": {
        "script": "run_inference_blindset.py",
        "out_subdir": "blindset_A",
        "eval_dataset_flag": "blindset_A",
    },
    "blindB": {
        "script": "run_inference_blindset.py",
        "out_subdir": "blindset_B",
        "eval_dataset_flag": "blindset_B",
    },
}


# ---------------------------------------------------------------------------
# Dynamic imports of sibling scripts
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_local_eval():
    return _load_module("_local_eval", SCRIPTS_DIR / "local_eval.py")


def _import_validate_prediction():
    return _load_module("_validate_prediction", SCRIPTS_DIR / "validate_prediction.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git_sha() -> str:
    """Return short git HEAD sha, or em dash on failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or "—"
    except Exception:
        return "—"


def tid_from_config(config_path: Path) -> str:
    """Experiment/task id = the config filename without the `.yaml` suffix."""
    return config_path.stem


def validate_config(config_path: Path) -> dict:
    """Load YAML via OmegaConf; hard-fail if `track_split_types != ['all_tracks']`."""
    if not config_path.exists():
        raise SystemExit(f"ERROR: config not found: {config_path}")
    cfg = OmegaConf.load(str(config_path))
    tst = OmegaConf.to_container(cfg.get("track_split_types", []))
    if tst != ["all_tracks"]:
        raise SystemExit(
            f"ERROR: config {config_path} has track_split_types={tst!r}; "
            "must be ['all_tracks'] per plan §2.5 catalog-integrity rule"
        )
    return OmegaConf.to_container(cfg, resolve=True)


def find_last_blind_config(submissions_log_path: Path) -> Optional[Path]:
    """Walk submissions_log.md (newest last) for the most recent `[blindA]`/`[blindB]` exp_id."""
    if not submissions_log_path.exists():
        return None
    try:
        text = submissions_log_path.read_text(encoding="utf-8")
    except OSError:
        return None
    last = None
    for line in text.splitlines():
        if "[blindA]" not in line and "[blindB]" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        last = cells[0]
    if not last:
        return None
    candidate = BASELINES_DIR / "config" / f"{last}.yaml"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Inference invocation
# ---------------------------------------------------------------------------

_SMOKE_WRAPPER = r"""
import sys
import importlib
import argparse

# Monkey-patch datasets.load_dataset to truncate to SMOKE_N sessions.
import datasets as _datasets
_orig_load = _datasets.load_dataset
SMOKE_N = {smoke_n}

def _patched(*a, **kw):
    ds = _orig_load(*a, **kw)
    try:
        return ds.select(range(min(SMOKE_N, len(ds))))
    except Exception:
        return ds

_datasets.load_dataset = _patched

# Also patch the already-imported symbol inside the target module if reloaded.
mod = importlib.import_module({module_name!r})
mod.load_dataset = _patched

ns = argparse.Namespace(**{kwargs!r})
mod.main(ns)
"""


def build_inference_cmd(
    split: str,
    tid: str,
    batch_size: int,
    smoke: bool,
    smoke_n: int = 5,
) -> list[str]:
    """Return the subprocess argv for the inference invocation (smoke or full)."""
    spec = SPLIT_CONFIG[split]
    script = spec["script"]
    module_name = script.replace(".py", "")
    script_kwargs: dict = {"tid": tid, "batch_size": batch_size, "save_path": "./exp/inference"}
    if spec["eval_dataset_flag"] is not None:
        script_kwargs["eval_dataset"] = spec["eval_dataset_flag"]

    if smoke:
        src = _SMOKE_WRAPPER.format(
            smoke_n=smoke_n,
            module_name=module_name,
            kwargs=script_kwargs,
        )
        return [sys.executable, "-c", src]

    # Full: invoke the script directly.
    argv = [sys.executable, script, "--tid", tid, "--batch_size", str(batch_size)]
    if spec["eval_dataset_flag"] is not None:
        argv.extend(["--eval_dataset", spec["eval_dataset_flag"]])
    return argv


def run_inference(
    cmd: list[str],
    log_path: Path,
    detach: bool,
) -> tuple[int, Optional[int]]:
    """Run the inference subprocess; return (return_code, pid). rc=-1 when detached."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if detach:
        log_f = open(log_path, "wb")
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASELINES_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return -1, proc.pid

    with open(log_path, "wb") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASELINES_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
    return rc, proc.pid


# ---------------------------------------------------------------------------
# Experiments log skeleton
# ---------------------------------------------------------------------------

_ENTRY_HEADER_RE = re.compile(r"^##\s+Entry template", re.MULTILINE)


def append_experiment_skeleton(
    log_path: Path,
    tid: str,
    axis: str,
    config_path: Path,
    sha: str,
    smoke: bool,
    split: str,
    metrics: Optional[dict],
) -> None:
    """Append a narrative-skeleton entry (template per plan §4.3) to experiments_log.md."""
    today = datetime.now().strftime("%Y-%m-%d")
    mode = "smoke" if smoke else "full"
    metrics_block = "  - pending" if not metrics else "\n".join(
        f"  - {k}: {v}" for k, v in metrics.items()
    )

    try:
        cfg_display = config_path.relative_to(REPO_ROOT)
    except ValueError:
        cfg_display = config_path

    entry = (
        f"\n### Exp {tid} — (headline TBD) — {today}\n\n"
        f"- **Hypothesis**: (fill in)\n"
        f"- **Axis**: {axis}\n"
        f"- **Config**: {cfg_display}\n"
        f"- **Code**: git sha {sha}\n"
        f"- **Code origin**: (fill in)\n"
        f"- **Smoke result** (5 rows): (pending)\n"
        f"- **Full result** (split: {split}, mode: {mode}):\n"
        f"{metrics_block}\n"
        f"- **Lessons**: \n"
        f"- **Verdict**: \n"
        f"- **Suggests next**: \n"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(
            "# Experiments log — narrative\n\nAppend-only. New entries at top.\n"
        )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Post-inference: validate + (dev: evaluate | blind: package)
# ---------------------------------------------------------------------------

def prediction_output_path(split: str, tid: str) -> Path:
    return INFERENCE_DIR / SPLIT_CONFIG[split]["out_subdir"] / f"{tid}.json"


def truncate_smoke_prediction(pred_path: Path, n_sessions: int = 5) -> Path:
    """Rewrite `pred_path` in place so it contains only the first `n_sessions` sessions."""
    with pred_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    seen: list[str] = []
    kept = []
    for r in rows:
        sid = r.get("session_id")
        if sid not in seen:
            if len(seen) >= n_sessions:
                continue
            seen.append(sid)
        if sid in seen[:n_sessions]:
            kept.append(r)
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    return pred_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end experiment orchestrator (plan §4.2): inference -> validate -> eval|package -> log.",
    )
    p.add_argument("--config", required=True, help="Path to config YAML under music-crs-baselines/config/")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="5-session subset run (plan W0-3)")
    mode.add_argument("--full", action="store_true", help="Full inference over the whole split")
    p.add_argument("--split", required=True, choices=("dev", "blindA", "blindB"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--detach", action="store_true",
                   help="Launch inference detached (Popen, start_new_session=True) so it survives shell close; prints PID and exits")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without running inference, writing files, or evaluating")
    p.add_argument("--axis", default=None,
                   help="Override the 'axis' label for the experiments_log skeleton (else use config.axis if set, else 'unknown')")
    p.add_argument("--sigma", type=float, default=0.005, help="Promotion margin for local dev eval")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    cfg_dict = validate_config(config_path)
    tid = tid_from_config(config_path)
    axis = args.axis or cfg_dict.get("axis", "unknown")
    sha = git_sha()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"{tid}_{args.split}_{timestamp}.log"
    split = args.split
    smoke = bool(args.smoke)

    inf_cmd = build_inference_cmd(
        split=split,
        tid=tid,
        batch_size=args.batch_size,
        smoke=smoke,
    )
    pred_path = prediction_output_path(split, tid)

    if smoke:
        cmd_preview = (
            f"{inf_cmd[0]} -c '<smoke wrapper: patches datasets.load_dataset -> "
            f"first 5 sessions, then calls {SPLIT_CONFIG[split]['script'][:-3]}.main>'"
        )
    else:
        cmd_preview = " ".join(inf_cmd)

    plan_lines = [
        f"tid            : {tid}",
        f"axis           : {axis}",
        f"config         : {config_path}",
        f"split          : {split}",
        f"mode           : {'smoke' if smoke else 'full'}",
        f"git sha        : {sha}",
        f"inference cmd  : {cmd_preview}",
        f"inference cwd  : {BASELINES_DIR}",
        f"inference log  : {log_path}",
        f"prediction out : {pred_path}",
        f"detach         : {args.detach}",
    ]
    print("\n".join(plan_lines))

    if args.dry_run:
        print("\n[dry-run] would run inference and post-processing; exiting without side effects.")
        return 0

    # ------------------------------------------------------------------
    # 1. Inference
    # ------------------------------------------------------------------
    rc, pid = run_inference(inf_cmd, log_path, detach=args.detach)

    if args.detach:
        print(f"\ninference launched detached: pid={pid}, log={log_path}")
        print("re-run (without --detach) or inspect the log when the run finishes.")
        return 0

    if rc != 0:
        print(f"ERROR: inference exited with code {rc}. See log: {log_path}", file=sys.stderr)
        return rc

    if not pred_path.exists():
        print(f"ERROR: inference returned 0 but no prediction file at {pred_path}", file=sys.stderr)
        return 1

    # Smoke: ensure we're only validating the 5-session slice (monkey-patched path
    # should already yield exactly 40 rows, but truncate defensively in case the
    # patch got bypassed).
    if smoke:
        truncate_smoke_prediction(pred_path, n_sessions=5)

    # ------------------------------------------------------------------
    # 2. Schema validation
    # ------------------------------------------------------------------
    vp = _import_validate_prediction()
    with pred_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    # For smoke the EXPECTED_ROWS check in validate_schema will fail the dev count
    # (expects 8000). Run a minimal sanity check instead for smoke.
    if smoke:
        errors = [e for e in vp.validate_schema(predictions, split) if "row count mismatch" not in e]
    else:
        errors = vp.validate_schema(predictions, split)

    if errors:
        print(f"ERROR: schema validation failed ({len(errors)} issue(s)):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"schema OK ({len(predictions)} rows, split={split}, smoke={smoke})")

    metrics: Optional[dict] = None

    # ------------------------------------------------------------------
    # 3a. Dev: run local_eval + append submissions_log row
    # ------------------------------------------------------------------
    if split == "dev" and not smoke:
        le = _import_local_eval()
        scores = le.run_evaluator(tid, split="devset")
        composite_retrieval = le.compute_composite_retrieval(scores)
        benchmarks = le.load_benchmarks()
        deltas = le.compute_deltas(scores, benchmarks)
        decision = le.decide(scores, deltas, sigma=args.sigma)
        summary = le._format_summary(
            tid=tid,
            scores=scores,
            composite_retrieval=composite_retrieval,
            composite_projected=None,
            llm_last_known=None,
            deltas=deltas,
            decision=decision,
            sigma=args.sigma,
        )
        print("\n" + summary)
        le.append_submissions_log_row(
            exp_id=tid,
            scores=scores,
            composite_retrieval=composite_retrieval,
            composite_projected=None,
            llm_last_known=None,
            status=decision,
        )
        metrics = {
            "nDCG@20": scores.get("ndcg@20"),
            "catalog_diversity": scores.get("catalog_diversity"),
            "lexical_diversity": scores.get("lexical_diversity"),
            "composite_retrieval": composite_retrieval,
            "decision": decision,
        }

    # ------------------------------------------------------------------
    # 3b. Blind: budget + attribution + package
    # ------------------------------------------------------------------
    if split in ("blindA", "blindB") and not smoke:
        ok, msg = vp.check_budget(SUBMISSIONS_LOG)
        if not ok:
            print(f"ERROR: budget check failed: {msg}", file=sys.stderr)
            return 1
        print(f"budget OK: {msg}")

        last_blind = find_last_blind_config(SUBMISSIONS_LOG)
        if last_blind is not None:
            warn = vp.check_attribution_warning(config_path, last_blind)
            if warn:
                print(f"WARNING: {warn}", file=sys.stderr)

        zip_path = pred_path.with_name("prediction.zip")
        vp.package_zip(pred_path, zip_path)
        print(f"packaged: {zip_path}")
        print("next step: upload this zip to CodaBench (no auto-upload).")
        metrics = {"zip": str(zip_path)}

    # ------------------------------------------------------------------
    # 4. Narrative skeleton append
    # ------------------------------------------------------------------
    append_experiment_skeleton(
        log_path=EXPERIMENTS_LOG,
        tid=tid,
        axis=axis,
        config_path=config_path,
        sha=sha,
        smoke=smoke,
        split=split,
        metrics=metrics,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
