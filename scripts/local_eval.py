"""Retrieval-side local evaluation harness — wraps music-crs-evaluator and computes composite metrics per plan 2.5."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_DIR = REPO_ROOT / "music-crs-evaluator"
EVALUATOR_SCRIPT = EVALUATOR_DIR / "evaluate_devset.py"
BENCHMARKS_PATH = REPO_ROOT / "documents" / "benchmarks.md"
SUBMISSIONS_LOG_PATH = REPO_ROOT / "documents" / "submissions_log.md"


COMPOSITE_WEIGHTS = {
    "ndcg@20": 0.50,
    "catalog_diversity": 0.10,
    "lexical_diversity": 0.10,
    "llm_judge_normalized": 0.30,
}


def _scores_path(tid: str, split: str = "devset") -> Path:
    return EVALUATOR_DIR / "exp" / "scores" / split / f"{tid}.json"


def compute_composite_retrieval(scores: dict) -> float:
    """Return the retrieval-side composite: 0.50*nDCG@20 + 0.10*CatDiv + 0.10*LexDiv."""
    return (
        COMPOSITE_WEIGHTS["ndcg@20"] * float(scores.get("ndcg@20", 0.0))
        + COMPOSITE_WEIGHTS["catalog_diversity"] * float(scores.get("catalog_diversity", 0.0))
        + COMPOSITE_WEIGHTS["lexical_diversity"] * float(scores.get("lexical_diversity", 0.0))
    )


def compute_composite_projected(scores: dict, llm_last_known: Optional[float]) -> float:
    """Return composite_retrieval plus the projected LLM term if llm_last_known is provided."""
    base = compute_composite_retrieval(scores)
    if llm_last_known is None:
        return base
    normalized = (float(llm_last_known) - 1.0) / 4.0
    return base + COMPOSITE_WEIGHTS["llm_judge_normalized"] * normalized


def run_evaluator(tid: str, split: str = "devset", force: bool = False) -> dict:
    """Invoke music-crs-evaluator/evaluate_devset.py for `tid` and return parsed scores JSON.

    Cached: if the scores file already exists and force=False, re-read it instead of rerunning.
    """
    scores_file = _scores_path(tid, split=split)
    if scores_file.exists() and not force:
        with scores_file.open("r") as f:
            return json.load(f)

    if not EVALUATOR_SCRIPT.exists():
        raise FileNotFoundError(f"Evaluator script not found at {EVALUATOR_SCRIPT}")

    cmd = [sys.executable, str(EVALUATOR_SCRIPT), "--tid", tid, "--eval_dataset", split]
    try:
        subprocess.run(
            cmd,
            cwd=str(EVALUATOR_DIR),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write(e.stdout or "")
        sys.exit(e.returncode or 1)

    if not scores_file.exists():
        raise FileNotFoundError(
            f"Evaluator ran but no scores file at {scores_file}; check evaluator output."
        )
    with scores_file.open("r") as f:
        return json.load(f)


def _parse_benchmark_table(text: str, header_regex: str) -> Optional[dict]:
    """Parse a 2-column `| Metric | Value |` markdown table starting after header_regex.

    Returns None if the section has no numeric rows.
    """
    header_match = re.search(header_regex, text)
    if not header_match:
        return None
    # consume until next `## ` heading
    tail = text[header_match.end():]
    end = tail.find("\n## ")
    section = tail if end == -1 else tail[:end]

    result: dict = {}
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, val = cells
        key = key.strip("* ")
        if key.lower() in {"metric", "---", ""} or set(key) <= {"-"}:
            continue
        val_clean = val.strip("* ")
        # strip markdown emphasis markers
        num_match = re.search(r"[-+]?\d*\.?\d+", val_clean)
        if not num_match:
            continue
        try:
            result[key] = float(num_match.group())
        except ValueError:
            continue
    return result if result else None


def _parse_bchamp_table(text: str) -> Optional[dict]:
    """Parse the B-champ 🥇 row; return None if it's the placeholder."""
    header_match = re.search(r"##\s*B-champ[^\n]*\n", text)
    if not header_match:
        return None
    tail = text[header_match.end():]
    end = tail.find("\n## ")
    section = tail if end == -1 else tail[:end]

    for line in section.splitlines():
        if "🥇" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # 🥇 row: rank | exp_id | composite | composite_retrieval | nDCG@20 | CatDiv | LexDiv | LLM | mechanism | config | date
        if len(cells) < 7:
            continue
        exp_id = cells[1]
        if exp_id in {"—", "-", ""}:
            return None
        def _num(c):
            m = re.search(r"[-+]?\d*\.?\d+", c)
            return float(m.group()) if m else None
        return {
            "exp_id": exp_id,
            "composite": _num(cells[2]),
            "composite_retrieval": _num(cells[3]),
            "nDCG@20": _num(cells[4]),
            "catalog_diversity": _num(cells[5]),
            "lexical_diversity": _num(cells[6]),
        }
    return None


def load_benchmarks() -> dict:
    """Parse documents/benchmarks.md → {b_floor, b_champ, b_target}.

    Resilient to the B-champ section being in its "None yet" placeholder state.
    """
    if not BENCHMARKS_PATH.exists():
        return {"b_floor": {}, "b_champ": None, "b_target": {}}
    text = BENCHMARKS_PATH.read_text()
    b_floor = _parse_benchmark_table(text, r"##\s*B-floor[^\n]*\n") or {}
    b_target = _parse_benchmark_table(text, r"##\s*B-target[^\n]*\n") or {}
    b_champ = _parse_bchamp_table(text)
    return {"b_floor": b_floor, "b_champ": b_champ, "b_target": b_target}


_METRIC_SYNONYMS = {
    "ndcg@1": ["ndcg@1", "nDCG@1"],
    "ndcg@10": ["ndcg@10", "nDCG@10"],
    "ndcg@20": ["ndcg@20", "nDCG@20"],
    "catalog_diversity": ["catalog_diversity", "CatDiv"],
    "lexical_diversity": ["lexical_diversity", "LexDiv"],
    "composite_retrieval": ["composite_retrieval"],
}


def _lookup(d: Optional[dict], key: str) -> Optional[float]:
    if not d:
        return None
    for name in _METRIC_SYNONYMS.get(key, [key]):
        if name in d and d[name] is not None:
            try:
                return float(d[name])
            except (TypeError, ValueError):
                return None
    return None


def compute_deltas(scores: dict, benchmarks: dict) -> dict:
    """Return {vs_b_floor, vs_b_champ, vs_b_target} for each comparable metric + composite_retrieval."""
    metrics = ["ndcg@1", "ndcg@10", "ndcg@20", "catalog_diversity", "lexical_diversity", "composite_retrieval"]

    # enrich the scores dict with composite_retrieval for the delta computation
    enriched = dict(scores)
    enriched["composite_retrieval"] = compute_composite_retrieval(scores)

    out = {}
    for label, key in [("vs_b_floor", "b_floor"), ("vs_b_champ", "b_champ"), ("vs_b_target", "b_target")]:
        bench = benchmarks.get(key)
        if not bench:
            out[label] = None
            continue
        # compute composite_retrieval on bench if derivable
        bench_enriched = dict(bench)
        if "composite_retrieval" not in bench_enriched and {"nDCG@20", "catalog_diversity", "lexical_diversity"} & set(bench_enriched.keys()):
            # try lowercased key
            nd = bench_enriched.get("nDCG@20", bench_enriched.get("ndcg@20"))
            cd = bench_enriched.get("catalog_diversity", bench_enriched.get("CatDiv"))
            ld = bench_enriched.get("lexical_diversity", bench_enriched.get("LexDiv"))
            if nd is not None and cd is not None and ld is not None:
                bench_enriched["composite_retrieval"] = 0.5 * nd + 0.1 * cd + 0.1 * ld

        deltas = {}
        for m in metrics:
            cur = _lookup(enriched, m)
            ref = _lookup(bench_enriched, m)
            if cur is None or ref is None:
                continue
            deltas[m] = cur - ref
        out[label] = deltas
    return out


def decide(scores: dict, deltas: dict, sigma: float) -> str:
    """Return 'PROMOTE to blind' or 'ITERATE'.

    Promote if composite_retrieval exceeds b_champ.composite_retrieval + sigma when b_champ is
    known; else if it exceeds b_floor.composite_retrieval + sigma.
    """
    cur = compute_composite_retrieval(scores)
    vs_champ = deltas.get("vs_b_champ")
    if vs_champ and "composite_retrieval" in vs_champ:
        if vs_champ["composite_retrieval"] > sigma:
            return "PROMOTE to blind"
        return "ITERATE"
    vs_floor = deltas.get("vs_b_floor")
    if vs_floor and "composite_retrieval" in vs_floor:
        if vs_floor["composite_retrieval"] > sigma:
            return "PROMOTE to blind"
    return "ITERATE"


def _git_sha() -> str:
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


def append_submissions_log_row(
    exp_id: str,
    scores: dict,
    composite_retrieval: float,
    composite_projected: Optional[float],
    llm_last_known: Optional[float],
    status: str,
    log_path: Path = SUBMISSIONS_LOG_PATH,
) -> None:
    """Append a `[dev-local]`-tagged markdown table row to submissions_log.md.

    Preserves existing content and uses append mode (`a`).
    """
    def fmt(v, places=4):
        if v is None:
            return "—"
        try:
            return f"{float(v):.{places}f}"
        except (TypeError, ValueError):
            return str(v)

    today = date.today().isoformat()
    sha = _git_sha()
    llm_cell = fmt(llm_last_known, places=2) if llm_last_known is not None else "—"
    composite_cell = fmt(composite_projected, places=4) if composite_projected is not None and llm_last_known is not None else "—"

    row = (
        f"| {exp_id} | [dev-local] "
        f"| {fmt(scores.get('ndcg@1'))} "
        f"| {fmt(scores.get('ndcg@10'))} "
        f"| {fmt(scores.get('ndcg@20'))} "
        f"| {fmt(scores.get('catalog_diversity'))} "
        f"| {fmt(scores.get('lexical_diversity'))} "
        f"| {llm_cell} "
        f"| {fmt(composite_retrieval)} "
        f"| {composite_cell} "
        f"| {today} "
        f"| {status} "
        f"| {sha} |\n"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(
            "| exp_id | tag | nDCG@1 | nDCG@10 | nDCG@20 | CatDiv | LexDiv | LLM | composite_retrieval | composite | date | status | git_sha |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        )
    with log_path.open("a") as f:
        f.write(row)


def _format_summary(
    tid: str,
    scores: dict,
    composite_retrieval: float,
    composite_projected: Optional[float],
    llm_last_known: Optional[float],
    deltas: dict,
    decision: str,
    sigma: float,
) -> str:
    lines = []
    lines.append(f"Local eval — {tid}")
    lines.append("-" * 60)
    lines.append(f"  nDCG@1             : {scores.get('ndcg@1', float('nan')):.4f}")
    lines.append(f"  nDCG@10            : {scores.get('ndcg@10', float('nan')):.4f}")
    lines.append(f"  nDCG@20            : {scores.get('ndcg@20', float('nan')):.4f}")
    lines.append(f"  catalog_diversity  : {scores.get('catalog_diversity', float('nan')):.4f}")
    lines.append(f"  lexical_diversity  : {scores.get('lexical_diversity', float('nan')):.4f}")
    lines.append(f"  composite_retrieval: {composite_retrieval:.4f}")
    if composite_projected is not None and llm_last_known is not None:
        lines.append(f"  composite_projected: {composite_projected:.4f}  (LLM_last_known={llm_last_known})")
    lines.append("")
    for label in ("vs_b_floor", "vs_b_champ", "vs_b_target"):
        d = deltas.get(label)
        if d is None:
            lines.append(f"  {label}: (no benchmark)")
            continue
        comp = d.get("composite_retrieval")
        comp_str = f"{comp:+.4f}" if comp is not None else "—"
        nd = d.get("ndcg@20")
        nd_str = f"{nd:+.4f}" if nd is not None else "—"
        lines.append(f"  {label}: Δcomposite_retrieval={comp_str}  ΔnDCG@20={nd_str}")
    lines.append("")
    lines.append(f"  sigma    : {sigma}")
    lines.append(f"  decision : {decision}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieval-side local evaluation harness (plan §2.5).",
    )
    parser.add_argument("--tid", required=True, help="Experiment id (matches evaluator's --tid).")
    parser.add_argument("--split", default="dev", choices=["dev", "devset"],
                        help="Split to evaluate on. 'dev' is aliased to 'devset'.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the cached scores file and re-run the evaluator.")
    parser.add_argument("--sigma", type=float, default=0.005,
                        help="Promotion margin for composite_retrieval over B-champ/B-floor.")
    parser.add_argument("--llm-last-known", type=float, default=None,
                        help="Last-known Gemini blind LLM-judge (1-5) for the response branch; used for composite_projected.")
    args = parser.parse_args()

    split = "devset" if args.split in ("dev", "devset") else args.split

    scores = run_evaluator(args.tid, split=split, force=args.force)
    composite_retrieval = compute_composite_retrieval(scores)
    composite_projected = compute_composite_projected(scores, args.llm_last_known)

    benchmarks = load_benchmarks()
    deltas = compute_deltas(scores, benchmarks)
    decision = decide(scores, deltas, sigma=args.sigma)

    print(_format_summary(
        tid=args.tid,
        scores=scores,
        composite_retrieval=composite_retrieval,
        composite_projected=composite_projected,
        llm_last_known=args.llm_last_known,
        deltas=deltas,
        decision=decision,
        sigma=args.sigma,
    ))

    append_submissions_log_row(
        exp_id=args.tid,
        scores=scores,
        composite_retrieval=composite_retrieval,
        composite_projected=composite_projected,
        llm_last_known=args.llm_last_known,
        status=decision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
