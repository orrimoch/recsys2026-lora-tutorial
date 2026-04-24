# `karpathy/autoresearch` — Summary & Blueprint for RecSys 2026

Source: https://github.com/karpathy/autoresearch (cloned 2026-04-17)
License: MIT

## 1. What it is

A minimal scaffold that lets an AI coding agent (Claude Code / Codex / etc.) autonomously iterate on an LLM pretraining script overnight. The agent edits code, runs a fixed 5-minute training experiment, reads one scalar metric, keeps or discards the change, commits, and repeats — for as long as the human lets it run (~12 experiments/hour, ~100 per overnight session).

The repo is the human-visible "origin story" of the fictional autonomous-research-swarm future described in the README.

## 2. The entire repo — three files that matter

| File | Role | Who edits it |
|---|---|---|
| `prepare.py` | Fixed. Downloads data shards, trains an 8 192-vocab BPE tokenizer, provides the dataloader and the `evaluate_bpb` metric. | **Never** |
| `train.py` | The full GPT model + Muon/AdamW optimizer + training loop. Everything is fair game. | **Agent** |
| `program.md` | Natural-language instructions that define the agent's loop, allowed actions, logging format, and stopping rules. | **Human** |

Supporting: `pyproject.toml` (uv-managed deps: torch 2.9.1, kernels, rustbpe, tiktoken, pyarrow), `analysis.ipynb` (post-hoc plotting of results.tsv), `progress.png` (teaser).

## 3. The core design decisions (why it works)

1. **Single editable file.** The agent only touches `train.py`. Scope stays manageable; diffs stay reviewable.
2. **Fixed 5-minute time budget.** Every experiment runs for exactly 300 s wall-clock (excluding compile/startup). Consequence: changes are directly comparable regardless of model size, batch size, or architecture — the metric *is* "best model achievable in 5 minutes on this box".
3. **One scalar metric: `val_bpb`.** Bits-per-byte on a pinned validation shard. Vocab-size-independent, so even vocabulary/tokenizer changes remain comparable. Lower is better.
4. **Frozen evaluation harness.** `evaluate_bpb` in `prepare.py` is the ground truth — the agent cannot touch it. This removes Goodhart-style metric gaming.
5. **Git as the keep/discard mechanism.** Each experiment is a commit on a `autoresearch/<tag>` branch; if `val_bpb` improves you advance, otherwise you `git reset` back.
6. **Flat results.tsv log (untracked).** `commit \t val_bpb \t memory_gb \t status \t description`, with statuses `keep | discard | crash`.
7. **Simplicity criterion.** Explicit rule in `program.md`: a tiny gain that adds ugly complexity is not worth it; a simplification that holds `val_bpb` flat *is* a win. This prevents the agent from ratcheting up code entropy.
8. **"NEVER STOP".** `program.md` hard-codes that once the loop starts, the agent does not pause to ask the human "should I continue?". The human might be asleep.

## 4. The agent loop (from `program.md`)

```
LOOP FOREVER:
  1. Inspect current branch/commit.
  2. Modify train.py with one experimental idea.
  3. git commit.
  4. uv run train.py > run.log 2>&1      # never tee — keep context clean
  5. grep "^val_bpb:\|^peak_vram_mb:" run.log
  6. If empty → crashed. `tail -n 50 run.log`, try to fix once or twice, else skip.
  7. Append row to results.tsv.
  8. If val_bpb improved → advance branch.
     Else → git reset to previous commit.
```

Extra rules: kill runs >10 min (treat as failure), dumb crashes → fix and retry, fundamentally broken ideas → log `crash` and move on, cold starts of new experiments never ask the human for permission.

## 5. The baseline `train.py` (what the agent starts from)

Non-trivial modern GPT. The agent doesn't start from a toy — it starts from something already strong, and must search for *further* gains.

- Architecture: decoder-only Transformer, RoPE, RMSNorm (functional, no learned scale), GQA-ready attention, **ReLU²** MLP, attention softcap on logits, alternating sliding-window attention (`window_pattern="SSSL"`), **value residual / ResFormer** with per-head gated value embeddings, learned per-layer `resid_lambdas` + `x0_lambdas` shortcut scalars.
- Backbone: FlashAttention-3 kernel fetched at runtime (`varunneal/flash-attention-3` on Hopper, `kernels-community/flash-attn3` otherwise).
- Optimizer: **MuonAdamW** — Muon (polar-express Newton–Schulz orthogonalisation, NorMuon variance reduction, cautious weight decay) for 2-D matrix params; AdamW for embeddings, lm_head, scalars. Both paths are `torch.compile(fullgraph=True)`-fused.
- Schedules: trapezoidal LR (no warmup, 50 % warmdown), Muon momentum ramp 0.85 → 0.95 over 300 steps, linearly decaying weight decay.
- Defaults: `DEPTH=8`, `ASPECT_RATIO=64` (⇒ n_embd ≈ 512), `HEAD_DIM=128`, `TOTAL_BATCH_SIZE=2¹⁹ ≈ 524 K` tokens/step, `DEVICE_BATCH_SIZE=128`, `MAX_SEQ_LEN=2048`, `VOCAB_SIZE=8192`.
- Failsafe: aborts if `loss` is NaN or >100.
- Hardware target: single NVIDIA GPU, tested on H100. Peak ~45 GB VRAM, ~40 % MFU, ~950 steps, ~500 M tokens, val_bpb ≈ 0.998 on the baseline.

## 6. Platform notes (relevant to our Mac + Colab setup)

The official repo is H100-only (FlashAttention-3 requires CUDA). Karpathy lists notable forks:
- `miolini/autoresearch-macos` — MacOS
- `trevin-creator/autoresearch-mlx` — Apple Silicon MLX
- `jsegov/autoresearch-win-rtx` — Windows RTX
- `andyluo7/autoresearch` — AMD

For small compute the README recommends: switch to TinyStories, drop vocab to 1–4 K (or byte-level), `MAX_SEQ_LEN=256`, lower `EVAL_TOKENS`, `DEPTH=4`, `WINDOW_PATTERN="L"`, `TOTAL_BATCH_SIZE=2¹⁴`.

## 7. Blueprint: how to apply this to RecSys 2026 (Music CRS)

Our challenge is a retrieve-then-generate music conversational recommender, scored by nDCG@{1,10,20} on a dev set, with inference that must run on Colab GPU (not locally). The autoresearch pattern maps onto it cleanly — with a few adaptations.

### Direct mappings

| autoresearch | RecSys 2026 equivalent |
|---|---|
| `prepare.py` (frozen) | `music-crs-evaluator/evaluate_devset.py` + `make_ground_truth.py` + the `data/` HF datasets |
| `train.py` (agent edits) | A single experiment entry point — e.g. `music-crs-baselines/mcrs/crs_baseline.py` driven by **a new YAML under `config/`** (per CLAUDE.md: never modify existing YAMLs) |
| `val_bpb` | **nDCG@10** on the dev set (single headline scalar; nDCG@{1,20} logged alongside) |
| 5-minute time budget | One Colab inference + local eval cycle — probably ~20–40 min. Budget should be wall-clock-fixed per experiment so results stay comparable. |
| `results.tsv` | e.g. `experiments/results.tsv` with `commit \t ndcg@10 \t ndcg@1 \t ndcg@20 \t status \t description` |
| `autoresearch/<tag>` branch | `experiments/<tag>` branch off `baseline` |
| FlashAttention-3 on H100 | Qwen on Colab GPU (per memory: prefer Qwen over Llama — no gating) |

### What the agent is allowed to touch

- Create **new** config YAMLs under `music-crs-baselines/config/` (never edit existing ones).
- Edit retrieval logic, prompt templates, reranking, candidate pooling, query construction.
- Swap retriever (BM25 → dense / hybrid / learned sparse).
- Swap generator prompt or decoding config.
- Must keep `track_split_types=["all_tracks"]` (hard rule from CLAUDE.md).
- Must not touch `music-crs-evaluator/exp/ground_truth/`.

### What the agent is forbidden to touch

- `music-crs-evaluator/` evaluation code (the analogue of `evaluate_bpb`).
- Existing config YAMLs.
- The HF raw datasets in `data/`.
- Commit cached indices, model weights, or large files.

### Adaptations needed for our setting

1. **Two execution surfaces, one loop.** autoresearch runs everything locally in one process. We have a split: inference on Colab GPU, eval on laptop. The loop needs a "dispatch → wait → pull logs → score → decide" step. A pragmatic first cut: have the agent *prepare* an experiment locally (new config + code change + commit), then the human runs the Colab notebook, and the agent resumes from the returned log file. A fuller automation would SSH/API-trigger Colab or use a Modal/Runpod runner.
2. **Longer iteration → stronger prior over what to try.** With 12 × autoresearch/hour you can afford random ablations; with maybe 2–4 RecSys experiments/hour we need better ideas per experiment. The `program.md` for our version should point the agent at the leaderboard gap (LLaMA-1B + BM25: nDCG@10 = 0.0627 → everything above that is open field) and at the `.claude/agents/researcher` and `prompt-engineer` agents for idea generation.
3. **Multi-metric decision rule.** BPB is one scalar; nDCG@10 is our headline but @1 and @20 matter. Keep it single-headline — advance on nDCG@10, but log all three and only discard a nDCG@10 improvement if @1 or @20 regressed by more than, say, 10 %.
4. **Simplicity criterion carries over.** Same rule: tiny score gains that balloon the config graph or pipeline complexity are not worth it.
5. **"NEVER STOP" applies with caveats.** On the GPU budget of Colab, the loop should pause on quota exhaustion rather than thrash. The human-facing contract should be "run until I stop you, or until Colab refuses".
6. **Reproducibility.** autoresearch pins one val shard. We already have a fixed dev set — preserve that. Also pin seeds in any new config.

### Minimum viable port — what to build first

1. `program_recsys.md` — our adapted playbook (setup, allowed/forbidden, loop, log format, stopping rules, simplicity clause, NEVER STOP clause).
2. `experiments/results.tsv` header row.
3. A thin wrapper `run_experiment.py` that: takes a config name, runs inference (Colab or local toggle), runs eval, extracts nDCG@{1,10,20} + peak memory, prints the 5-line summary block that the agent greps for.
4. A branch-per-experiment convention: `experiments/<tag>` off `baseline`.
5. Point one of our existing agents (most likely `experimenter`) at `program_recsys.md` and let it rip.

### Open questions before we start

- Do we automate the Colab dispatch or keep a human in the loop for the GPU step? (Affects `program_recsys.md` wording and the value of per-hour throughput.)
- What is our fixed wall-clock-per-experiment budget? (Equivalent of the 5-min constant — needs to be a hard constant for fair comparison.)
- Single headline metric: confirm nDCG@10, and define the tie-breaking rule against @1/@20.

## 8. One-line takeaway

autoresearch is not a framework — it's a 3-file, 1-scalar, 1-branch *protocol* for letting a coding agent optimise a single objective overnight. The leverage we get by porting it is turning our CLAUDE.md rules, evaluator, and config directory into exactly that protocol — with nDCG@10 in the role of val_bpb and a new YAML + pipeline tweak in the role of a `train.py` commit.
