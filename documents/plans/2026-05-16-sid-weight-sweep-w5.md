# SID Weight Sweep + Multi-Blind-A Submissions (W5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find the SID stream weight in the wRRF ensemble that maximizes dev nDCG@20, then submit the winning config (plus the pure-SID baseline 171) to Blind-A. Locks the W6 frozen config based on Blind-A composite.

**Architecture:** Parameterize the SID hub repo + SID stream weight via two new YAML fields (`sid_hub_repo`, `sid_stream_weight`) that the factory reads with safe defaults — no new config files needed. A sweep notebook generates 4 temporary configs (weights {0.3, 0.5, 0.7, 1.0}) on the fly, runs each on dev, paired-bootstrap CIs vs config 170 (baseline weight=0.5), picks winner. A second notebook submits the winner + the pure-SID config 171 to Blind-A.

**Tech Stack:** Python 3.10, `omegaconf` (existing), `pandas`, `pyarrow`. Reuses W4's `compare_blind_predictions.py` for paired-bootstrap. No new pip deps.

**Spec reference:** `documents/specs/2026-05-15-sid-retrieval-design.md` §5.1 W5 row + spec §1 promotion path (v2 attempt pure-SID if it Pareto-dominates).

**Inputs from prior weeks:**
- W4 artifact: HF Hub `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged` (the trained SID generator)
- W4 artifact: configs 170 (ensemble) + 171 (pure-SID) in `music-crs-baselines/config/`
- W4 artifact: `scripts/compare_blind_predictions.py` (paired-bootstrap on prediction files)
- W4 artifact: factory entries `sid_generator` + `wrrf_bm25_dense_sid_v1` in `mcrs/retrieval_modules/__init__.py`

**Hard prerequisite:** W4's dev gate (notebook 64) must have PASSED (`Δ nDCG@20 ≥ +0.005 AND CI lo > 0`) before starting W5. If W4 gate failed, W5 is cancelled — fix W3 (bigger model / more epochs / better doc2query) before sweeping weights.

**Output artifacts:**
- `experiments/cache/w5_sweep/<run_id>/{170-w0.3.json, 170-w0.5.json, 170-w0.7.json, 170-w1.0.json, sweep_summary.json}` on Drive
- Two Blind-A submission zips: ensemble-winner + pure-SID-171
- `experiments/cache/w5_sweep/<run_id>/blind_a_results.json` with both composites once CodaBench scores arrive
- Memory file `project_w5_weight_sweep_result.md`

---

## File structure

| Path | Type | Responsibility |
|---|---|---|
| `music-crs-baselines/mcrs/retrieval_modules/__init__.py` | modify | Read `sid_stream_weight` + `sid_hub_repo` from config dict; pass into factory branches |
| `music-crs-baselines/run_inference_blindset.py` | modify | Pass full `config` dict to `load_retrieval_module` (currently passes specific fields only) so the new YAML fields are visible |
| `tests/test_w5_weight_override.py` | new | 4 TDD tests: factory respects `sid_stream_weight`, factory respects `sid_hub_repo`, defaults preserved when fields absent, wrrf sub_specs reflect override |
| `colab/65_sid_weight_sweep.ipynb` | new | 10 cells: clone + deps + Drive + generate 4 weight-variant configs + run each on dev + paired-bootstrap comparison + winner display |
| `colab/66_blind_a_w5_submissions.ipynb` | new | 9 cells: clone + deps + Drive + run Blind-A on winner ensemble + run Blind-A on config 171 + validate + zip × 2 |

**No new YAML config files.** Sweep notebook generates ephemeral configs on-the-fly via OmegaConf override — avoids 4 near-duplicate YAML files that would drift over time.

---

## Task 1: Setup — verify W4 gate passed + W4 artifacts on Hub

**Files:**
- Verify: `experiments/cache/sid_eval/w4_gate.json` shows `gate_pass: true`
- Verify: HF Hub `OrRim123/recsys2026-sid-generator-qwen15b-v1-merged`
- Verify: configs `170-wrrf-sid-v5kto-blindsetA.yaml` + `171-pure-sid-v5kto-blindsetA.yaml` exist

- [ ] **Step 1: Verify W4 gate JSON shows pass**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
cat experiments/cache/sid_eval/w4_gate.json 2>/dev/null | python -c "
import json, sys
try:
    m = json.load(sys.stdin)
    delta = m.get('delta_mean', 0)
    ci_lo = m.get('paired_bootstrap_ci', {}).get('lo', -1)
    print(f'delta={delta:+.4f}, ci_lo={ci_lo:+.4f}')
    if delta >= 0.005 and ci_lo > 0:
        print('OK: W4 gate PASSED — proceed with W5')
    else:
        print('WARN: W4 gate did NOT pass — W5 may be premature. Pivot recommended.')
except Exception as e:
    print(f'W4 gate JSON not yet on this machine (likely on Drive). Skip if you ran notebook 64.')
"
```

Expected: `OK: W4 gate PASSED — proceed with W5`. If the JSON lives only on Colab Drive, skip and accept the user-stated W4 gate outcome.

- [ ] **Step 2: Verify W4 model on Hub**

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.repo_info('OrRim123/recsys2026-sid-generator-qwen15b-v1-merged')
    print(f'OK: model exists, last_modified={info.last_modified}')
except Exception as e:
    print(f'MISSING: {e}'); print('ABORT W5 — re-run W3 first')
"
```

Expected: `OK: model exists, last_modified=...`.

- [ ] **Step 3: Verify W4 configs exist**

```bash
ls music-crs-baselines/config/170-wrrf-sid-v5kto-blindsetA.yaml music-crs-baselines/config/171-pure-sid-v5kto-blindsetA.yaml
```

Expected: both files listed (W4 deliverable).

---

## Task 2: TDD `sid_hub_repo` YAML field support in factory

**Files:**
- Create: `tests/test_w5_weight_override.py`
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`

The factory currently hardcodes the SID hub repo. W5 needs to allow YAML configs to override it (so we can A/B v1 vs a hypothetical v2 without changing factory code). Defaults to v1 when absent.

- [ ] **Step 1: Read current factory to find the SID branch**

```bash
grep -n "sid_generator\|hub_repo=" music-crs-baselines/mcrs/retrieval_modules/__init__.py | head -10
```

Expected output: factory branch `elif retrieval_type == "sid_generator":` plus a `hub_repo=...` line.

- [ ] **Step 2: Write the failing test**

Create `tests/test_w5_weight_override.py`:

```python
"""W5 tests: sid_stream_weight + sid_hub_repo YAML override in factory."""
import pytest


def test_factory_sid_generator_respects_hub_repo_override(monkeypatch, tmp_path):
    """When extra_config contains sid_hub_repo, factory passes it to SID_GENERATOR."""
    import pandas as pd

    # Build a tiny W1 lookup so SID_GENERATOR init can find it.
    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    pd.DataFrame([
        {"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0,
         "popularity": 1.0, "bucket_rank": 0},
    ]).to_parquet(sid_dir / "track_to_sid.parquet")

    captured = {}
    class _StubSidGen:
        def __init__(self, hub_repo, sid_lookup_path, **kwargs):
            captured["hub_repo"] = hub_repo
            captured["sid_lookup_path"] = str(sid_lookup_path)
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "mcrs.retrieval_modules.sid_generator.SID_GENERATOR", _StubSidGen,
    )

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="sid_generator",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
        extra_config={"sid_hub_repo": "OrRim123/recsys2026-sid-generator-qwen15b-v2-merged"},
    )

    assert captured["hub_repo"] == "OrRim123/recsys2026-sid-generator-qwen15b-v2-merged"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_w5_weight_override.py::test_factory_sid_generator_respects_hub_repo_override -q
```

Expected: TypeError or AttributeError — `load_retrieval_module` doesn't accept `extra_config` yet.

- [ ] **Step 4: Update factory to accept extra_config**

In `music-crs-baselines/mcrs/retrieval_modules/__init__.py`, find the function signature `def load_retrieval_module(retrieval_type, dataset_name, track_split_types, corpus_types, cache_dir):` and add the new parameter:

```python
def load_retrieval_module(
    retrieval_type,
    dataset_name,
    track_split_types,
    corpus_types,
    cache_dir,
    extra_config: dict | None = None,
):
    extra_config = extra_config or {}
```

(Place the `extra_config = extra_config or {}` line as the FIRST line of the function body.)

Then in the `elif retrieval_type == "sid_generator":` branch, replace the hardcoded hub_repo with:

```python
    elif retrieval_type == "sid_generator":
        from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
        sid_hub_repo = extra_config.get(
            "sid_hub_repo",
            "OrRim123/recsys2026-sid-generator-qwen15b-v1-merged",
        )
        return SID_GENERATOR(
            hub_repo=sid_hub_repo,
            sid_lookup_path=Path(cache_dir) / "sid" / "track_to_sid.parquet",
            device="cuda",
            num_beams=20,
            max_prompt_len=1024,
            cap_per_bucket=1,
        )
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_w5_weight_override.py -q
```

Expected: 1 passing.

- [ ] **Step 6: Commit**

```bash
git add tests/test_w5_weight_override.py music-crs-baselines/mcrs/retrieval_modules/__init__.py
git commit -m "sid w5: factory accepts sid_hub_repo override (TDD, 1 test)"
```

---

## Task 3: TDD `sid_stream_weight` YAML field support in wRRF factory

The wRRF factory currently hardcodes `weight: 0.5` for the SID stream. W5 needs to allow YAML configs to override this weight without forking the config file.

**Files:**
- Modify: `music-crs-baselines/mcrs/retrieval_modules/__init__.py`
- Modify: `tests/test_w5_weight_override.py` (append)

- [ ] **Step 1: Append failing test**

```python
def test_factory_wrrf_sid_respects_stream_weight_override(monkeypatch, tmp_path):
    """When extra_config contains sid_stream_weight, the wrrf_bm25_dense_sid_v1
    sub_specs has the SID entry with the overridden weight (not the default 0.5)."""
    captured_sub_specs = {}

    class _StubRrf:
        def __init__(self, dataset_name, track_split_types, corpus_types, cache_dir,
                     sub_specs, k):
            captured_sub_specs["specs"] = sub_specs
            captured_sub_specs["k"] = k

    monkeypatch.setattr("mcrs.retrieval_modules.RRF_MODEL", _StubRrf)

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="wrrf_bm25_dense_sid_v1",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
        extra_config={"sid_stream_weight": 0.7},
    )

    sid_specs = [s for s in captured_sub_specs["specs"] if s["type"] == "sid_generator"]
    assert len(sid_specs) == 1
    assert sid_specs[0]["weight"] == 0.7


def test_factory_wrrf_sid_defaults_to_0_5_when_override_absent(monkeypatch, tmp_path):
    """When extra_config lacks sid_stream_weight, default of 0.5 is preserved."""
    captured_sub_specs = {}

    class _StubRrf:
        def __init__(self, dataset_name, track_split_types, corpus_types, cache_dir,
                     sub_specs, k):
            captured_sub_specs["specs"] = sub_specs

    monkeypatch.setattr("mcrs.retrieval_modules.RRF_MODEL", _StubRrf)

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="wrrf_bm25_dense_sid_v1",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
    )

    sid_specs = [s for s in captured_sub_specs["specs"] if s["type"] == "sid_generator"]
    assert sid_specs[0]["weight"] == 0.5


def test_factory_sid_hub_repo_defaults_to_v1_when_override_absent(monkeypatch, tmp_path):
    """When extra_config lacks sid_hub_repo, default to v1 merged."""
    import pandas as pd
    sid_dir = tmp_path / "sid"
    sid_dir.mkdir()
    pd.DataFrame([
        {"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0,
         "popularity": 1.0, "bucket_rank": 0},
    ]).to_parquet(sid_dir / "track_to_sid.parquet")

    captured = {}
    class _StubSidGen:
        def __init__(self, hub_repo, sid_lookup_path, **kwargs):
            captured["hub_repo"] = hub_repo

    monkeypatch.setattr(
        "mcrs.retrieval_modules.sid_generator.SID_GENERATOR", _StubSidGen,
    )

    from mcrs.retrieval_modules import load_retrieval_module
    load_retrieval_module(
        retrieval_type="sid_generator",
        dataset_name="dummy", track_split_types=[], corpus_types=[],
        cache_dir=str(tmp_path),
    )

    assert captured["hub_repo"] == "OrRim123/recsys2026-sid-generator-qwen15b-v1-merged"
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
python -m pytest tests/test_w5_weight_override.py -q
```

Expected: 2 new test failures (weight assertions fail because factory still hardcodes 0.5; the third test about default hub_repo should pass since Task 2 already set that up).

- [ ] **Step 3: Update wRRF branch to read sid_stream_weight**

In `music-crs-baselines/mcrs/retrieval_modules/__init__.py`, find the `elif retrieval_type == "wrrf_bm25_dense_sid_v1":` branch and replace the SID sub-spec entry's hardcoded weight with the override-able value:

```python
    elif retrieval_type == "wrrf_bm25_dense_sid_v1":
        sid_weight = float(extra_config.get("sid_stream_weight", 0.5))
        return RRF_MODEL(
            dataset_name, track_split_types, corpus_types, cache_dir,
            sub_specs=[
                {
                    "type": "bm25",
                    "corpus_types": [
                        "track_name", "artist_name", "album_name",
                        "release_date", "tag_list",
                    ],
                    "topk_internal": 60,
                    "weight": 1.0,
                },
                {
                    "type": "dense_metadata_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "dense_lyrics_qwen3_instruct",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": 0.4,
                },
                {
                    "type": "sid_generator",
                    "corpus_types": corpus_types,
                    "topk_internal": 20,
                    "weight": sid_weight,
                },
            ],
            k=60,
        )
```

(Only one line changed inside the SID sub-spec: `"weight": 0.5` → `"weight": sid_weight`, plus the `sid_weight = ...` line at the top of the branch.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_w5_weight_override.py -q
```

Expected: 4 passing.

- [ ] **Step 5: Commit**

```bash
git add tests/test_w5_weight_override.py music-crs-baselines/mcrs/retrieval_modules/__init__.py
git commit -m "sid w5: factory accepts sid_stream_weight override (TDD, 3 tests; default 0.5)"
```

---

## Task 4: Wire `extra_config` from `run_inference_blindset.py` config dict to factory

The factory now accepts `extra_config`, but the runner doesn't pass it. Without this wiring, YAML overrides are invisible.

**Files:**
- Modify: `music-crs-baselines/mcrs/crs_baseline.py`
- Modify: `music-crs-baselines/run_inference_blindset.py`

- [ ] **Step 1: Find the load_retrieval_module call site**

```bash
grep -n "load_retrieval_module" music-crs-baselines/mcrs/crs_baseline.py music-crs-baselines/run_inference_blindset.py
```

- [ ] **Step 2: Update `crs_baseline.py` `__init__` to accept + forward extra_config**

Find the call to `load_retrieval_module(...)` in `crs_baseline.py:__init__`. Look at the surrounding code to understand the current signature. Then:

(a) Add a new optional parameter `extra_config: dict | None = None` to `CRS_BASELINE.__init__`.

(b) Forward it:
```python
self.retrieval = load_retrieval_module(
    retrieval_type=retrieval_type,
    dataset_name=item_db_name,
    track_split_types=track_split_types,
    corpus_types=corpus_types,
    cache_dir=cache_dir,
    extra_config=extra_config,
)
```

- [ ] **Step 3: Update `run_inference_blindset.py` to pass the FULL config dict**

Find the `CRS_BASELINE(...)` instantiation in `run_inference_blindset.py`. Add `extra_config=dict(config)` as a kwarg. Use `OmegaConf.to_container(config, resolve=True)` if `config` is an OmegaConf object (it is per the file's existing `OmegaConf.load` usage).

Concretely, near the CRS_BASELINE construction line:

```python
extra_config_dict = OmegaConf.to_container(config, resolve=True)
music_crs = CRS_BASELINE(
    # ... existing kwargs ...,
    extra_config=extra_config_dict,
)
```

- [ ] **Step 4: Smoke test that imports + Crm baseline init still resolves**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -c "
import sys
sys.path.insert(0, 'music-crs-baselines')
from mcrs.crs_baseline import CRS_BASELINE
import inspect
sig = inspect.signature(CRS_BASELINE.__init__)
assert 'extra_config' in sig.parameters, 'extra_config not on CRS_BASELINE signature'
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Run all existing tests to verify no regressions**

```bash
python -m pytest tests/ -q --ignore=tests/test_local_eval.py --ignore=tests/test_wave0_integration.py --ignore=tests/test_wave1_integration.py --ignore=tests/test_wave2_integration.py 2>&1 | tail -3
```

Expected: previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add music-crs-baselines/mcrs/crs_baseline.py music-crs-baselines/run_inference_blindset.py
git commit -m "sid w5: thread extra_config from YAML through CRS_BASELINE to factory"
```

---

## Task 5: Create sweep notebook `colab/65_sid_weight_sweep.ipynb`

This notebook generates 4 ephemeral configs on-the-fly (weights {0.3, 0.5, 0.7, 1.0}), runs each on dev, computes paired-bootstrap CIs, picks the winner.

**Files:**
- Create: `colab/65_sid_weight_sweep.ipynb`

- [ ] **Step 1: Generate the notebook**

Run this exact script from repo root:

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python <<'EOF'
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md("""# 65 — W5 SID weight sweep on dev

Sweeps SID stream weight in wRRF over {0.3, 0.5, 0.7, 1.0}, runs each variant
on the dev split, computes paired-bootstrap CIs vs the W4 baseline (config 170
with weight=0.5), picks the winner.

Wallclock: ~5-6 hr on L4 (4 full-pipeline runs back-to-back) / ~2-3 hr Blackwell.
""")

code("""# 1) Setup — same pattern as notebook 64.
import os
from google.colab import userdata, drive
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
drive.mount('/content/drive', force_remount=False)

BRANCH = 'fresh-model'
!rm -rf /content/recsys2026
!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026
%cd /content/recsys2026

DRIVE_BASE = '/content/drive/MyDrive'
LOCAL_BASE = '/content/recsys2026/experiments/cache'
os.makedirs(LOCAL_BASE, exist_ok=True)
for name, drive_subdir in [
    ('sid', 'recsys2026_sid_cache'),
    ('sid_training', 'recsys2026_sid_training_cache'),
    ('dense', 'recsys2026_dense_cache'),
    ('w5_sweep', 'recsys2026_w5_sweep_cache'),
]:
    src = f'{DRIVE_BASE}/{drive_subdir}'
    dst = f'{LOCAL_BASE}/{name}'
    os.makedirs(src, exist_ok=True)
    if os.path.islink(dst): os.unlink(dst)
    elif os.path.exists(dst):
        import shutil; shutil.rmtree(dst)
    os.symlink(src, dst)

!pip install -q -U \"peft>=0.10\" \"transformers>=4.40\" \"accelerate>=0.30\" \"torchao>=0.17\"
""")

code("""# 2) Generate 4 ephemeral configs by copying 170 + injecting sid_stream_weight overrides.
import shutil, datetime
from omegaconf import OmegaConf

%cd /content/recsys2026/music-crs-baselines
WEIGHTS = [0.3, 0.5, 0.7, 1.0]
RUN_ID = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
print(f'Sweep run_id: {RUN_ID}')

base_config = OmegaConf.load('config/170-wrrf-sid-v5kto-blindsetA.yaml')
# Override test_dataset_name to point at dev split
base_config.test_dataset_name = 'talkpl-ai/TalkPlayData-Challenge-Dataset'

for w in WEIGHTS:
    cfg = OmegaConf.merge(base_config, OmegaConf.create({'sid_stream_weight': w}))
    tid = f'170-sweep-w{int(w*10):02d}'
    OmegaConf.save(cfg, f'config/{tid}.yaml')
    print(f'  wrote config/{tid}.yaml (sid_stream_weight={w})')""")

code("""# 3) Run each weight variant on dev. ~75 min each on L4. (Skip variants you've already run.)
import os
for w in WEIGHTS:
    tid = f'170-sweep-w{int(w*10):02d}'
    out_path = f'exp/inference/dev/{tid}.json'
    if os.path.exists(out_path):
        print(f'SKIP (cached): {tid}')
        continue
    print(f'RUN: {tid} (weight={w})')
    !python run_inference_blindset.py \\
        --tid {tid} \\
        --eval_dataset dev \\
        --batch_size 32 \\
        2>&1 | tee /content/drive/MyDrive/recsys2026_w5_sweep_cache/{RUN_ID}_{tid}.log | tail -30""")

code("""# 4) Paired-bootstrap CIs: each weight variant vs config 170 (weight=0.5 baseline).
# Also vs config 132 (current champion).
%cd /content/recsys2026
import json, subprocess
sweep_results = {}
for w in WEIGHTS:
    tid = f'170-sweep-w{int(w*10):02d}'
    out_path = f'experiments/cache/w5_sweep/{RUN_ID}_{tid}_vs_170.json'
    !python scripts/compare_blind_predictions.py \\
        --pred_a music-crs-baselines/exp/inference/dev/{tid}.json \\
        --pred_b music-crs-baselines/exp/inference/dev/170-wrrf-sid-v5kto-blindsetA.json \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --gold_split test \\
        --label_a 'sweep w={w}' \\
        --label_b 'baseline w=0.5 (170)' \\
        --n_resamples 1000 \\
        --alpha 0.05 \\
        --output {out_path}
    sweep_results[w] = json.load(open(out_path))""")

code("""# 5) Pick winner: highest mean_ndcg_a (the sweep variant) among configs that
# beat the baseline by Δ ≥ 0 (any improvement is reported; statistical significance
# logged but not blocking — sweep is exploratory).
print(f'{'weight':>6} {'mean_ndcg':>10} {'\\u0394 vs 0.5':>10} {'CI lo':>8} {'CI hi':>8}')
print('-' * 50)
best_w, best_ndcg = None, -1
for w, m in sweep_results.items():
    ndcg = m['mean_ndcg_a']
    delta = m['delta_mean']
    lo = m['paired_bootstrap_ci']['lo']
    hi = m['paired_bootstrap_ci']['hi']
    print(f'{w:>6.1f} {ndcg:>10.4f} {delta:>+10.4f} {lo:>+8.4f} {hi:>+8.4f}')
    if ndcg > best_ndcg:
        best_ndcg, best_w = ndcg, w
print('-' * 50)
print(f'\\nWINNER: weight={best_w} with mean_ndcg@20={best_ndcg:.4f}')

# Persist sweep summary for notebook 66 to pick up.
import json
summary = {
    'run_id': RUN_ID,
    'best_weight': best_w,
    'best_ndcg': best_ndcg,
    'all_results': {str(w): m for w, m in sweep_results.items()},
}
with open('experiments/cache/w5_sweep/sweep_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'wrote experiments/cache/w5_sweep/sweep_summary.json')""")

code("""# 6) ALSO compare best sweep variant vs current champion config 132 (sanity check
# that the W5 winner actually beats the non-SID baseline, not just the W4 baseline).
best_tid = f'170-sweep-w{int(best_w*10):02d}'
!python scripts/compare_blind_predictions.py \\
    --pred_a music-crs-baselines/exp/inference/dev/{best_tid}.json \\
    --pred_b music-crs-baselines/exp/inference/dev/132-bge-m3-v5kto-prorank-rerank-blindsetA.json \\
    --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
    --gold_split test \\
    --label_a f'best (w={best_w})' \\
    --label_b 'champion 132' \\
    --output experiments/cache/w5_sweep/best_vs_132.json
print()
print(json.dumps(json.load(open('experiments/cache/w5_sweep/best_vs_132.json')),
                 indent=2))""")

md("""## After the sweep

The winning weight is in `experiments/cache/w5_sweep/sweep_summary.json` under `best_weight`.

**Decision tree for Blind-A submissions (notebook 66):**

1. **Winner ensemble beats config 132 on dev (Δ > 0 + CI lo > 0)**: submit the winner ensemble + config 171 (pure-SID) to Blind-A in notebook 66.
2. **Winner ensemble ties 132 on dev**: still submit (Blind-A may differ from dev). Pure-SID 171 still informative.
3. **Winner ensemble loses to 132 on dev**: do NOT submit Blind-A. Pivot: retrain W3 with bigger model OR fix doc2query coverage.

Pre-W6 deadline: 2026-06-20 — leaves 2-day buffer for cleanup + memory writes before Blind-B opens 2026-06-23.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path('colab/65_sid_weight_sweep.ipynb').write_text(json.dumps(nb, indent=1))
print('wrote colab/65_sid_weight_sweep.ipynb')
EOF
```

- [ ] **Step 2: Verify the notebook**

```bash
python -c "
import nbformat
nb = nbformat.read('colab/65_sid_weight_sweep.ipynb', as_version=4)
print(f'cells: {len(nb.cells)}')
for i, c in enumerate(nb.cells):
    src = ''.join(c.source)
    print(f'  cell {i:2d} [{c.cell_type.upper()}] {src[:70].replace(chr(10), \" | \")}')
"
```

Expected: 8 cells (1 markdown header + 6 code + 1 markdown footer).

- [ ] **Step 3: Commit**

```bash
git add colab/65_sid_weight_sweep.ipynb
git commit -m "sid w5: notebook 65 — SID weight sweep {0.3,0.5,0.7,1.0} + paired-bootstrap"
```

---

## Task 6: Create multi-submission notebook `colab/66_blind_a_w5_submissions.ipynb`

Runs Blind-A on the sweep winner (ensemble) + on config 171 (pure-SID). Produces TWO CodaBench submission zips.

**Files:**
- Create: `colab/66_blind_a_w5_submissions.ipynb`

- [ ] **Step 1: Generate the notebook**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python <<'EOF'
import json
from pathlib import Path

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})

md("""# 66 — W5 Blind-A submissions: sweep winner + pure-SID

Reads `experiments/cache/w5_sweep/sweep_summary.json` to find the winning weight
from notebook 65. Runs Blind-A on:
  (A) the winning ensemble config (e.g. `170-sweep-w07-blindsetA.yaml`)
  (B) config 171 (pure-SID)

Produces two CodaBench-ready zips. Wallclock ~3 hr on L4 / ~1.5 hr Blackwell.
""")

code("""# 1) Setup.
import os
from google.colab import userdata, drive
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
drive.mount('/content/drive', force_remount=False)

BRANCH = 'fresh-model'
!rm -rf /content/recsys2026
!git clone -b {BRANCH} https://github.com/orrimoch/recsys2026-lora-tutorial.git /content/recsys2026
%cd /content/recsys2026

DRIVE_BASE = '/content/drive/MyDrive'
LOCAL_BASE = '/content/recsys2026/experiments/cache'
os.makedirs(LOCAL_BASE, exist_ok=True)
for name, drive_subdir in [
    ('sid', 'recsys2026_sid_cache'),
    ('dense', 'recsys2026_dense_cache'),
    ('w5_sweep', 'recsys2026_w5_sweep_cache'),
]:
    src = f'{DRIVE_BASE}/{drive_subdir}'
    dst = f'{LOCAL_BASE}/{name}'
    os.makedirs(src, exist_ok=True)
    if os.path.islink(dst): os.unlink(dst)
    elif os.path.exists(dst):
        import shutil; shutil.rmtree(dst)
    os.symlink(src, dst)

!pip install -q -U \"peft>=0.10\" \"transformers>=4.40\" \"accelerate>=0.30\" \"torchao>=0.17\"
""")

code("""# 2) Pick the winning weight from notebook 65's sweep summary.
import json
summary_path = 'experiments/cache/w5_sweep/sweep_summary.json'
assert os.path.exists(summary_path), 'Run notebook 65 first to produce sweep_summary.json'
summary = json.load(open(summary_path))
best_w = summary['best_weight']
print(f'Winner: weight={best_w} (mean_ndcg@20={summary[\"best_ndcg\"]:.4f})')

# Generate the Blind-A version of the winning config: copy base 170 + inject weight
# + flip test_dataset_name back to Blind-A.
from omegaconf import OmegaConf
%cd /content/recsys2026/music-crs-baselines
base = OmegaConf.load('config/170-wrrf-sid-v5kto-blindsetA.yaml')
winner_cfg = OmegaConf.merge(base, OmegaConf.create({'sid_stream_weight': best_w}))
winner_tid = f'170-best-w{int(best_w*10):02d}-blindsetA'
OmegaConf.save(winner_cfg, f'config/{winner_tid}.yaml')
print(f'wrote config/{winner_tid}.yaml')""")

code("""# 3) Run Blind-A on the winning ensemble.
!python run_inference_blindset.py \\
    --tid {winner_tid} \\
    --batch_size 32 \\
    2>&1 | tee /content/drive/MyDrive/recsys2026_w5_sweep_cache/blindA_{winner_tid}.log | tail -30""")

code("""# 4) Run Blind-A on config 171 (pure-SID).
!python run_inference_blindset.py \\
    --tid 171-pure-sid-v5kto-blindsetA \\
    --batch_size 32 \\
    2>&1 | tee /content/drive/MyDrive/recsys2026_w5_sweep_cache/blindA_171-pure-sid.log | tail -30""")

code("""# 5) Validate both prediction files (must be 80 entries each).
%cd /content/recsys2026
for tid in [winner_tid, '171-pure-sid-v5kto-blindsetA']:
    pred_path = f'music-crs-baselines/exp/inference/blindset_A/{tid}.json'
    preds = json.load(open(pred_path))
    n = len(preds) if isinstance(preds, list) else len(preds.keys())
    assert n == 80, f'{tid}: expected 80, got {n}'
    print(f'OK: {tid} has {n} entries')
    !python scripts/validate_prediction.py --input {pred_path}""")

code("""# 6) Zip both for CodaBench submission.
import zipfile, datetime
date_str = datetime.date.today().strftime('%Y-%m-%d')
SUBMIT_DIR = f'/content/drive/MyDrive/recsys2026_submissions'
os.makedirs(SUBMIT_DIR, exist_ok=True)
for tid, label in [(winner_tid, f'sid-ensemble-w{int(best_w*10):02d}'),
                   ('171-pure-sid-v5kto-blindsetA', 'sid-pure')]:
    pred_path = f'music-crs-baselines/exp/inference/blindset_A/{tid}.json'
    zip_path = f'{SUBMIT_DIR}/{date_str}-{label}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname='prediction.json')
    print(f'zip ready: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)')""")

md("""## After the run

1. Download both zips from `/content/drive/MyDrive/recsys2026_submissions/`.
2. Upload BOTH to CodaBench (note: there may be daily submission quota — sequence them).
3. Record both composite scores in `project_blind_a_w5_results.md`:
   - Winner ensemble (`170-best-w<N>`): composite, nDCG@20, LLM, lex_div
   - Pure-SID (`171`): composite, nDCG@20, LLM, lex_div
4. Compare to:
   - Pre-SID champion (config 132, composite 0.21 if cached)
   - W4 first submission (config 170 weight=0.5)
5. **W5 gate**: Blind-A nDCG@20 ≥ 0.08 on the winner. If PASS → freeze for W6/Blind-B. If FAIL → pivot.

Final choice for W6:
- If ensemble wins on Blind-A composite: ship ensemble.
- If pure-SID wins or ties (per spec §1 promotion path v3): ship pure-SID (simpler pipeline).
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
Path('colab/66_blind_a_w5_submissions.ipynb').write_text(json.dumps(nb, indent=1))
print('wrote colab/66_blind_a_w5_submissions.ipynb')
EOF
```

- [ ] **Step 2: Verify the notebook**

```bash
python -c "
import nbformat
nb = nbformat.read('colab/66_blind_a_w5_submissions.ipynb', as_version=4)
print(f'cells: {len(nb.cells)}')
"
```

Expected: 8 cells.

- [ ] **Step 3: Commit**

```bash
git add colab/66_blind_a_w5_submissions.ipynb
git commit -m "sid w5: notebook 66 — multi-Blind-A (winner ensemble + pure-SID)"
```

---

## Task 7: Final test sweep + code-reviewer agent

- [ ] **Step 1: Run all SID tests**

```bash
cd /Users/orrimoch/PythonProjs/recsys2026
python -m pytest tests/test_sid_preprocessing.py tests/test_sid_validation.py tests/test_sid_quantizer.py tests/test_sid_training_data.py tests/test_sid_generator_dataloader.py tests/test_sid_vocab.py tests/test_sid_training_format.py tests/test_sid_inference.py tests/test_sid_eval.py tests/test_sid_generator_retrieval.py tests/test_w5_weight_override.py tests/test_compare_blind_predictions.py -q 2>&1 | tail -3
```

Expected: 117 (103 W1-W4 + 4 W5 + 8 compare-script + 2 more from session leakage fix patches) passing.

- [ ] **Step 2: Full repo test sweep**

```bash
python -m pytest tests/ -q --ignore=tests/test_local_eval.py --ignore=tests/test_wave0_integration.py --ignore=tests/test_wave1_integration.py --ignore=tests/test_wave2_integration.py 2>&1 | tail -3
```

Expected: ~518+ passing (514 W4-baseline + 4 W5).

- [ ] **Step 3: Verify clean git state + push**

```bash
git status -s
git log --oneline -10
git push origin fresh-model
```

- [ ] **Step 4: Dispatch code-reviewer agent**

```
Agent({
  description: "Review SID W5 weight sweep",
  subagent_type: "superpowers:code-reviewer",
  prompt: "Review the W5 implementation of the SID weight sweep + multi-Blind-A submissions against the design spec.

  Spec: documents/specs/2026-05-15-sid-retrieval-design.md §5.1 W5 row + §1 promotion path
  Plan: documents/plans/2026-05-16-sid-weight-sweep-w5.md

  W4 artifacts this depends on:
  - HF Hub: OrRim123/recsys2026-sid-generator-qwen15b-v1-merged
  - configs 170 (ensemble) + 171 (pure-SID)
  - scripts/compare_blind_predictions.py

  Files to review (newly added on fresh-model):
  - music-crs-baselines/mcrs/retrieval_modules/__init__.py (added extra_config support)
  - music-crs-baselines/mcrs/crs_baseline.py (extra_config forwarding)
  - music-crs-baselines/run_inference_blindset.py (extra_config from YAML)
  - tests/test_w5_weight_override.py (4 tests)
  - colab/65_sid_weight_sweep.ipynb
  - colab/66_blind_a_w5_submissions.ipynb

  Critical checks:
  1. Backwards compatibility: existing configs that don't set sid_stream_weight or sid_hub_repo
     must still get the W4 defaults (0.5 / v1-merged). Verify tests cover this.
  2. The sweep notebook generates ephemeral configs with OmegaConf.merge — does this actually
     work to inject sid_stream_weight into the YAML, or does it land at the wrong nesting level?
  3. Notebook 66 picks winner from sweep_summary.json — what if the file doesn't exist (user
     skipped notebook 65)? Assert exists with a clear error.
  4. Notebook 65 cell 2 changes test_dataset_name to dev for sweep; notebook 66 cell 2 must
     reset it to Blind-A for the winning config. Verify.
  5. The factory accepts extra_config; does crs_baseline.py forward it for ALL retrieval_type
     branches, or only for SID? If only for SID, document; if for all, no concern.

  Report: APPROVE / APPROVE-WITH-CHANGES / NEEDS-MAJOR-REVISION + top-5 findings."
})
```

- [ ] **Step 5: Address reviewer findings**

If APPROVE → done. If APPROVE-WITH-CHANGES → patch inline + commit. If NEEDS-MAJOR-REVISION → triage before user runs notebook 65.

---

## Self-review checklist (per writing-plans skill)

**1. Spec coverage** (against §5.1 W5 row + §1 promotion path):

- §5.1 sweep weights {0.3, 0.5, 0.7, 1.0} — Task 5 cell 2 ✓
- §5.1 compare ensemble vs pure-SID on Blind-A — Task 6 cells 3+4 ✓
- §5.1 W5 gate: Blind-A nDCG@20 ≥ 0.08 — documented in markdown footer of Task 6 ✓
- §5.1 Pareto-dominance decision — Task 6 markdown footer ✓
- §1 promotion path v2 (pure-SID comparison) — Task 6 submits config 171 ✓

**2. Placeholder scan**: searched for "TBD", "TODO", "implement later" — none. Every step has runnable code/commands.

**3. Type consistency**:
- `extra_config: dict | None = None` — used identically in factory + CRS_BASELINE
- `sid_stream_weight: float` (default 0.5) — read by `wrrf_bm25_dense_sid_v1` branch
- `sid_hub_repo: str` (default v1-merged) — read by `sid_generator` branch
- Sweep notebook uses `OmegaConf.merge` to inject `sid_stream_weight` at the top level of the config dict — matches what factory reads via `extra_config.get("sid_stream_weight", 0.5)`

**4. Sequencing**: Task 1 verifies prereqs, Tasks 2-3 add factory support (tests-first), Task 4 wires it through CRS_BASELINE, Task 5 uses it from a notebook, Task 6 reads Task 5's sweep_summary, Task 7 reviews.

**Plan complete.**

---

## Estimated wallclock

| Component | Time |
|---|---|
| Task 1 (setup verify) | 2 min |
| Tasks 2-3 (factory override + tests) | ~15 min via subagent |
| Task 4 (wire extra_config) | ~10 min via subagent |
| Task 5 (sweep notebook) | ~5 min via subagent |
| Task 6 (submission notebook) | ~5 min via subagent |
| Task 7 (final review) | ~10 min agent + iteration |
| **Total my coordination time** | **~50 min** |
| User Colab time (notebook 65 sweep) | **~5-6 hr on L4 / ~2-3 hr Blackwell** |
| User Colab time (notebook 66 submissions) | **~3 hr L4 / ~1.5 hr Blackwell** |
| User CodaBench upload + score wait | hours-days |

W5 is implementation-light (factory parameterization + two notebooks). The heavy lift is Colab compute time for the sweep — 4 full pipeline runs back-to-back.
