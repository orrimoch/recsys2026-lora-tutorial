"""Dump scalars from a W3 SID-generator TensorBoard run and print a diagnostic summary."""
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = Path(sys.argv[1])
ea = EventAccumulator(str(RUN_DIR), size_guidance={"scalars": 0})
ea.Reload()

tags = ea.Tags()["scalars"]
print(f"=== tags in {RUN_DIR.name} ===")
for t in tags:
    print(f"  {t}")
print()

def series(tag):
    return [(s.step, s.value) for s in ea.Scalars(tag)]

def summarize(tag, every=None):
    pts = series(tag)
    if not pts:
        return
    n = len(pts)
    first_step, first_val = pts[0]
    last_step, last_val = pts[-1]
    vmin = min(v for _, v in pts)
    vmax = max(v for _, v in pts)
    print(f"--- {tag}  (n={n}, step {first_step}->{last_step}) ---")
    print(f"    first: {first_val:.6g}   last: {last_val:.6g}   min: {vmin:.6g}   max: {vmax:.6g}")
    # Print ~10 evenly-spaced snapshots so we can eyeball the trajectory.
    every = max(1, n // 10) if every is None else every
    for i in range(0, n, every):
        s, v = pts[i]
        print(f"    step {s:>6}:  {v:.6g}")
    s, v = pts[-1]
    print(f"    step {s:>6}:  {v:.6g}  (final)")
    print()

for tag in tags:
    summarize(tag)
