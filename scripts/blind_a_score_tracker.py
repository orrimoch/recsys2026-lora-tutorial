"""Append-only Blind-A submission score tracker.

Maintains a markdown table in a memory file so all CodaBench Blind-A
submissions are visible in one place. Run after each CodaBench score
becomes visible:

    python scripts/blind_a_score_tracker.py append \\
        --tracker ~/.claude/projects/.../memory/project_blind_a_submissions.md \\
        --config_id 170-wrrf-sid-v5kto-blindsetA \\
        --composite 0.245 --ndcg 0.082 --llm 2.50 --lex_div 0.77 \\
        --url https://codabench.org/competitions/.../submission/...

    python scripts/blind_a_score_tracker.py read \\
        --tracker ~/.claude/projects/.../memory/project_blind_a_submissions.md
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Optional


HEADER = (
    "| config_id | composite | ndcg@20 | llm | lex_div | submitted_at | url | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def append_score(
    tracker_path: Path,
    config_id: str,
    composite: float,
    ndcg: float,
    llm: float,
    lex_div: float,
    submission_url: str,
    notes: str = "",
    submitted_at: Optional[str] = None,
) -> None:
    """Append one CodaBench submission result to the markdown table."""
    if submitted_at is None:
        submitted_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if not tracker_path.exists():
        # First write: include the table header.
        preamble = (
            "---\n"
            "name: Blind-A submissions tracker\n"
            "description: Append-only log of all CodaBench Blind-A submissions for the SID retriever sprint. Maintained by scripts/blind_a_score_tracker.py.\n"
            "type: project\n"
            "---\n\n"
            "# Blind-A submissions\n\n"
            "Each row = one CodaBench submission. composite = 0.5*nDCG@20 + 0.1*CatDiv + 0.1*LexDiv + 0.3*LLM (CatDiv is saturated at ~0.03 across the leaderboard; only nDCG/LLM/LexDiv are interesting).\n\n"
        )
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text(preamble + HEADER)

    row = (
        f"| {config_id} "
        f"| {composite:.4f} "
        f"| {ndcg:.4f} "
        f"| {llm:.4f} "
        f"| {lex_div:.4f} "
        f"| {submitted_at} "
        f"| {submission_url} "
        f"| {notes} |\n"
    )
    with tracker_path.open("a") as f:
        f.write(row)


def read_tracker(tracker_path: Path) -> list[dict]:
    """Parse the markdown table back into a list of dicts."""
    if not tracker_path.exists():
        return []
    text = tracker_path.read_text()
    rows: list[dict] = []
    in_table = False
    columns: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("| config_id "):
            columns = [c.strip() for c in line.strip("|").split("|")]
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != len(columns):
                continue
            rows.append(dict(zip(columns, cells)))
        elif in_table and not line.startswith("|"):
            # Table ended
            in_table = False
    return rows


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    append_p = sub.add_parser("append")
    append_p.add_argument("--tracker", type=Path, required=True)
    append_p.add_argument("--config_id", type=str, required=True)
    append_p.add_argument("--composite", type=float, required=True)
    append_p.add_argument("--ndcg", type=float, required=True)
    append_p.add_argument("--llm", type=float, required=True)
    append_p.add_argument("--lex_div", type=float, required=True)
    append_p.add_argument("--url", type=str, dest="submission_url", required=True)
    append_p.add_argument("--notes", type=str, default="")

    read_p = sub.add_parser("read")
    read_p.add_argument("--tracker", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "append":
        append_score(
            tracker_path=args.tracker,
            config_id=args.config_id,
            composite=args.composite, ndcg=args.ndcg,
            llm=args.llm, lex_div=args.lex_div,
            submission_url=args.submission_url,
            notes=args.notes,
        )
        print(f"appended row for {args.config_id} -> {args.tracker}")
    elif args.cmd == "read":
        rows = read_tracker(args.tracker)
        print(f"{len(rows)} rows:")
        for r in rows:
            print(f"  {r['config_id']:50s} composite={r['composite']} ndcg={r['ndcg@20']}")


if __name__ == "__main__":
    main()
