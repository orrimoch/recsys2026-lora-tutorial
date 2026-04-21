# RecSys 2026 — LoRA Fine-tune for Music CRS (Blind-A)

Minimal, self-contained repo to fine-tune Qwen 2.5-3B with LoRA on the
TalkPlay challenge train data and produce a `prediction.zip` for the
Blind-A leaderboard.

Everything is glued together by a single bash script (`run_pipeline.sh`) that
runs five stages: venv → dataset → train → inference → package.

```
recsys2026-lora-tutorial/
├── README.md
├── requirements.txt          # Pinned versions known to work end-to-end
├── setup_venv.sh             # Bootstrap a Python 3.10 venv
├── run_pipeline.sh           # Full E2E with all hyperparams as env vars
├── prompts/                  # v10 champion prompt files
│   ├── roleplay.txt
│   └── response_generation_v4.txt
├── lora/                     # LoRA-specific scripts
│   ├── build_train_dataset.py
│   └── train_lora.py
├── inference/                # Self-contained inference + packaging
│   ├── run_inference.py
│   └── make_prediction_zip.py
└── colab/
    └── Train_LoRA_Colab.ipynb
```

After the pipeline finishes, `output/prediction.zip` is the file to upload
to CodaBench.

---

## Prerequisites

- **Python 3.10** (the venv is created from this exact version).
- **macOS / Linux** (the bash script uses POSIX syntax).
- **Disk:** ~15 GB (HF cache for Qwen 2.5-3B + the train/test datasets).
- **Memory:** 16 GB+ on M4 (32 GB+ comfortable). A100 / L4 GPU on Colab is
  drastically faster.

## Data

This repo does **NOT include the dataset** — it's pulled at runtime from
HuggingFace via `datasets.load_dataset()`. All four datasets are public
under the `talkpl-ai` org:

| Dataset | Used by | Size |
|---|---|---|
| `talkpl-ai/TalkPlayData-Challenge-Dataset` (train split) | `lora/build_train_dataset.py` | ~50 MB |
| `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` | dataset build + inference | ~30 MB |
| `talkpl-ai/TalkPlayData-Challenge-User-Metadata` | dataset build + inference | ~5 MB |
| `talkpl-ai/TalkPlayData-Challenge-Blind-A` (test split) | `inference/run_inference.py` | ~50 KB |

HF caches downloads to `~/.cache/huggingface/` after first use; subsequent
runs read from cache (no network).

**Pre-fetch all four datasets up-front** (recommended — confirms HF
connectivity before the long training run, and lands data in a repo-local
cache instead of `~/.cache/huggingface/`):

```bash
./download_data.sh                      # all four, cache at ./data/hf_cache
./download_data.sh --only blind         # smoke: just the 80-row test set
HF_TOKEN=hf_... ./download_data.sh      # auth (faster, no rate limit)
HF_HOME=/some/path ./download_data.sh   # custom cache location
```

The bash wrapper sets `HF_HOME` to a repo-local path (`./data/hf_cache`) so
datasets stay co-located with the code. `run_pipeline.sh` defaults to the
**same** `HF_HOME`, so the pipeline reads from the cache the download
populated — no double-download.

To suppress HF's rate-limit warning system-wide, get a token at
<https://huggingface.co/settings/tokens> and `export HF_TOKEN=hf_...`.

---

## Quick start (full pipeline, default hyperparams)

```bash
cd recsys2026-lora-tutorial
./run_pipeline.sh
```

That runs all five stages with sane defaults. On an A100 it takes ~1 hour.
On an M4 it takes 4–8 hours (training is the bottleneck — see "Time budget").

When done, upload `output/prediction.zip` to the CodaBench Blind-A leaderboard.

---

## What each stage does

| Stage | Script | Output | M4 time | A100 time |
|---|---|---|---|---|
| 1. venv | `setup_venv.sh` | `recsys26-lora/` | ~3 min | ~3 min |
| 2. dataset | `lora/build_train_dataset.py` | `data/train_sft/` | ~10 min | ~10 min |
| 3. train | `lora/train_lora.py` | `lora_adapters/.../final_adapter/` | 4–6 h | 25–35 min |
| 4. inference | `inference/run_inference.py` | `output/predictions.json` | ~15 min | ~5 min |
| 5. package | `inference/make_prediction_zip.py` | `output/prediction.zip` | ~1 sec | ~1 sec |

### Stage 2 — what gets filtered

The SFT dataset is built from the HF train split with these filters (each
maps to a known failure mode from prior leaderboard experiments):

- **Turn-1 only.** Blind-A is single-turn cold-start; turns 2–8 are
  conversational follow-ups.
- **Iterative warm-opener strip.** "Awesome! Glad you liked X." removed.
- **Min 40 words after strip.** Discards 1-sentence chat replies.
- **No banned words** (absolutely / fantastic / truly / amazing). The v10
  prompt explicitly bans these to lift the LLM-judge score.
- **No apologies / failure admissions.** v10's prompt rule 4 forbids "I'm
  sorry"; training data with these would re-introduce the bug.
- **No "Glad you liked / enjoyed".** Causes turn-1 hallucinations.
- **Citation pattern required** (year-in-parens OR quoted/asterisked title).
  Pushes the model toward v10's metadata-rich critic style.
- **Sentence-final close** (`. ! ? "` etc.).

Typical retention: ~1,200 rows out of 15,199 sessions. Small but high quality.

### Stage 3 — LoRA training details

- Base: `Qwen/Qwen2.5-3B-Instruct` (no gating; M4-feasible in fp32).
- LoRA rank 16, α=32, dropout 0.05.
- Target modules: Q/K/V/O projections (`attn`) by default. Set
  `TARGET_MODULES=attn_mlp` to add MLP layers (more capacity, ~2× params).
- **Completion-only loss** — only assistant tokens contribute to gradient.
  System prompt + user query are masked with `-100`. This is critical:
  without it, ~78% of gradient signal would be the model memorizing the
  system prompt instead of learning v10-style responses.
- **`enable_input_require_grads()`** — required for gradient_checkpointing
  + PEFT to combine without breaking embedding gradient flow.
- LR 1e-4 with cosine schedule, 0.03 warmup ratio.

### Stage 4 — inference

- Same BM25 retrieval as the v10 champion (4-field corpus:
  track_name, artist_name, album_name, release_date).
- Same prompt template (v10 persona, ban list, top-3 candidates).
- Loads base Qwen + applies the LoRA adapter via
  `PeftModel.from_pretrained(...).merge_and_unload()` for fast generation.
- Greedy decode with `[system, user]` chat template only — avoids the
  `Glad you enjoyed X` hallucination caused by the stock module's
  fake-assistant turn injection.

### Stage 5 — packaging

CodaBench requires:
- A single `.zip` with **exactly** `prediction.json` (singular) at root.
- 80 rows, all five required fields, 20 distinct `predicted_track_ids` per row.

`make_prediction_zip.py` validates all of this before writing the zip.
Validation failures abort with a clear error message.

---

## Tuning hyperparameters

Every hyperparam is an **env var read by `run_pipeline.sh`**. Override at
invocation time:

```bash
NUM_EPOCHS=2 LR=5e-5 LORA_R=32 ./run_pipeline.sh
```

| Variable | Default | Notes |
|---|---|---|
| `NUM_EPOCHS` | `1` | More epochs risk overfitting on ~1.2k rows. |
| `LR` | `1e-4` | Conservative — 2e-4+ risks catastrophically forgetting v10 style. |
| `PER_DEVICE_BATCH` | `1` | M4 default. A100 can do 2–4. |
| `GRAD_ACCUM` | `16` | Effective batch = `PER_DEVICE_BATCH × GRAD_ACCUM`. |
| `LORA_R` | `16` | Rank. 8–32 is reasonable; 16 is the SFT default. |
| `LORA_ALPHA` | `32` | Usually `2 × LORA_R`. |
| `LORA_DROPOUT` | `0.05` | Up to 0.1 if overfitting on small data. |
| `TRAIN_MAX_LENGTH` | `2048` | M4: 2048 / A100: 3072. Smaller drops rows. |
| `TARGET_MODULES` | `attn` | `attn` (Q/K/V/O) or `attn_mlp` (+ gate/up/down). |
| `BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Try 7B on Colab if VRAM allows. |
| `DEVICE` | auto (cuda > mps > cpu) | Force a specific device for inference. |
| `SUBSET` | empty | Set to e.g. `3` for an inference smoke (skips zip). |

### Skip stages with the `SKIP_*` switches

```bash
# Already trained an adapter; just re-run inference + package
SKIP_VENV=1 SKIP_DATASET=1 SKIP_TRAIN=1 ./run_pipeline.sh

# Smoke test: 3 rows, no zip output
SKIP_VENV=1 SUBSET=3 ./run_pipeline.sh
```

---

## Time budget

| Setup | Stage 3 (training) | Total E2E |
|---|---|---|
| **Colab A100** | 25–35 min | ~50 min |
| **Colab L4** | ~70–90 min | ~2 h |
| **M4 Mac (fp32)** | 4–6 h | 5–7 h |
| **CPU only** | days — don't | n/a |

For Colab, use `colab/Train_LoRA_Colab.ipynb`. The notebook clones the
repo to `/content/`, installs deps, runs Stage 2 + 3, then zips the
adapter to Drive. You then download the adapter and run Stages 4 + 5
locally on M4 (Stage 4 is fast even on M4; only training is slow).

---

## Smoke test (verify the pipeline works without committing to a full run)

```bash
# Build dataset, train on a tiny subset, do 3-row inference, skip zip.
./run_pipeline.sh
# (then) re-run with smoke flags:
SKIP_VENV=1 SKIP_DATASET=1 SUBSET=3 ./run_pipeline.sh
```

Expected output: 3 generated responses written to
`output/predictions.json`, each cites a real track from BM25 top-1.

---

## Troubleshooting

**`No module named pip` after venv creation:**
The bash script bootstraps pip via `python -m ensurepip --upgrade`.
If this fails, install Python 3.10 from python.org (not Homebrew's
`python@3.10`, which sometimes ships without ensurepip).

**`SFTConfig.__init__() got an unexpected keyword argument 'max_length'`:**
TRL version mismatch. Pin to the exact version in `requirements.txt`.

**Dataset build retains 0 rows:**
Likely the HF dataset name changed. Check
`talkpl-ai/TalkPlayData-Challenge-Dataset` on HF Hub.

**Trainer drops all rows ("dropped N rows with no completion tokens"):**
`TRAIN_MAX_LENGTH` is too small — system prompt + user query alone
exceeds it. Bump to 2048 (M4) or 3072 (A100).

**MPS thrashing (slow steps + page-faults):**
Quit other apps. The 3B model + activations + LoRA optimizer state
peaks around 18–20 GB. On 16 GB Macs use Colab.

**Adapter inference produces gibberish:**
You may have loaded the wrong adapter (training was incomplete). Verify
`final_adapter/adapter_config.json` exists and `final_adapter/adapter_model.safetensors` is non-empty.

---

## What this repo does NOT do

- Does **not** include the train/test data — pulled fresh from HuggingFace each run.
- Does **not** reproduce the v10 champion BM25 cache from the larger repo —
  built fresh on first inference run (~3 min one-time cost).
- Does **not** include the broader `mcrs/` retrieval module ecosystem
  (BERT, dense, RRF, sequential rerank). Inference uses BM25 only — that's
  what v10 uses.
- Does **not** ship with experiment logs or champion ledger; this is a
  standalone fine-tuning repo, not a competition tracker.
