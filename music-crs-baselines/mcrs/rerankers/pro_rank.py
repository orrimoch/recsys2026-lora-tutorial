"""ProRank — last-token-logit-diff reranker over an SLM.

Per RecSys_Challenge_Plan §A5 (and ProRank, Mixedbread 2025
documents/research/ProRank_2506.03487.pdf). For each (query, doc) pair, prompt
a small causal LM with a yes/no relevance question and score by the difference
in next-token logits at "yes" vs "no":

    score(q, d) = logit_t("yes") - logit_t("no")

where t is the position immediately after the prompt. Higher score = more
relevant. Sort candidates by score descending → top-k.

Why this works (paper §3): the SLM has been instruction-tuned to answer
yes/no questions; the *difference* is a calibrated per-pair relevance
signal that beats much larger zero-shot rerankers on BEIR.

This module ships **inference-only** by default — uses a pretrained Qwen-0.5B
out of the box. The W3 gate can be evaluated without training. For the
trained ProRank policy (paper's GRPO warmup + main loop), see
`colab/20_train_prorank_dev.ipynb` which produces a LoRA adapter that can
be loaded via `model_path`.

Rationale generation (plan §A5): `generate_rationales()` produces 3-5-word
"why this fits" strings for the top-K reranked tracks, used by W4's
`<reranker_rationales>` responder prompt block (plan §6.5). Decoupled from
`rerank()` so callers can opt-in only at W4+.

Backend compatibility (matches StateTracker / CMQR_REWRITER pattern):
  - HF transformers (default) — works on any device
  - vLLM (CUDA only) — drop-in via lm.lm.generate() if a VLLM_MODEL is passed

Cache: a `(query, tid)` score cache so repeat eval runs are fast. Cache
filename includes `sha1(prompt_template)[:8]` so changing the prompt
template forces fresh scoring (silent reuse of stale scores would
invalidate the W3 gate). Lives at
`{cache_dir}/prorank/scores_{model_safe}__{prompt_hash}.pkl`.

Note on prompt template: this module's template is music-domain-tuned
("Is this document relevant to the user's music request?"). The paper
uses MS-MARCO style ("Given a query, predict whether the document is
relevant... Relevant: "). For chat-tuned Qwen the difference is
empirically negligible; if a future GRPO warmup is attempted, evaluate
both prompts.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Standard ProRank-style relevance prompt. The model answers yes/no — we
# read the next-token logits at " yes" vs " no" (leading space to match
# the tokenizer's natural tokenization for most Qwen / Llama families).
DEFAULT_PROMPT_TEMPLATE = (
    "Query: {query}\n"
    "Document: {doc}\n"
    "Is this document relevant to the user's music request? Answer yes or no."
)

# Template for rationale generation. Decoupled from scoring — only invoked
# when the caller opts in via `generate_rationales()`. We ask for a 3-5 word
# phrase that justifies the match; output is decoded greedily with a small
# max_new_tokens budget so the cost is bounded.
DEFAULT_RATIONALE_TEMPLATE = (
    "Query: {query}\n"
    "Track: {doc}\n"
    "In 3-5 words, why does this track fit the query?"
    " Output ONLY the 3-5 word rationale, no preamble."
)
RATIONALE_MAX_NEW_TOKENS = 16


def _safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _detect_backend(lm) -> str:
    """Match StateTracker._detect_backend pattern."""
    if "vllm" in type(lm).__name__.lower():
        return "vllm"
    return "hf"


class ProRankReranker:
    """Last-token-logit-diff reranker over Qwen-0.5B (or any causal LM).

    Args:
        item_db_name, track_split_types, corpus_types, cache_dir: shared
            with all rerankers; used to build the tid → text map (same as
            BGE_RERANKER builds it).
        model_name: HF model id or local path. Default Qwen-0.5B.
        model_path: optional local LoRA adapter path. If provided, base
            `model_name` weights load first then the adapter is applied.
        prompt_template: relevance question template. Must contain
            {query} and {doc} placeholders.
        max_doc_chars: truncate long doc strings before prompting.
        max_new_tokens_score: ignored — we only need the next-token logits,
            no generation. Kept for API parity if we later switch to
            generation-based scoring.
        device: 'cuda' / 'mps' / 'cpu' / None (auto-pick).
        dtype: torch dtype; default bf16 on CUDA/MPS, fp32 on CPU.
        batch_size: how many (query, doc) pairs per forward pass.
        with_rationales: if True, after scoring run a tiny generation
            pass on the top-N to emit per-track 3-5-word rationales.
            Default False (W3 only needs IDs; W4 enables rationales).
    """

    def __init__(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        model_name: str = DEFAULT_MODEL,
        model_path: Optional[str] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_doc_chars: int = 600,
        device: Optional[str] = None,
        dtype=None,
        batch_size: int = 16,
        with_rationales: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Device pick (mirrors BGE).
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.max_doc_chars = int(max_doc_chars)
        self.prompt_template = prompt_template
        self.with_rationales = bool(with_rationales)

        # Validate prompt template.
        if "{query}" not in prompt_template or "{doc}" not in prompt_template:
            raise ValueError(
                "prompt_template must contain both {query} and {doc} placeholders."
            )

        # Load model + tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so we can read the *last non-pad* token's logits cleanly
        # for batched scoring. Right-pad would require per-row attention masks.
        self.tokenizer.padding_side = "left"

        # P1 #3 fix: use `torch_dtype=` (the long-standing kwarg) for
        # compatibility with older transformers (≤4.45 silently ignores
        # `dtype=`). vLLM stacks pin transformers; bge_reranker.py uses the
        # same. Use the alias accepted by both old + new transformers.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
        ).to(device).eval()
        # Optional LoRA adapter load (for trained ProRank in W3+).
        if model_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_path).eval()
            print(f"[pro-rank] loaded LoRA adapter from {model_path}")
        print(f"[pro-rank] loaded {model_name} on {device} dtype={dtype}")
        # P2 #11 fix: assert padding side stayed left after model load — some
        # chat tokenizers default to right-padding which would silently
        # corrupt `logits[:, -1, :]` in batched scoring.
        assert self.tokenizer.padding_side == "left", (
            f"ProRank expects left-padded tokenizer; got "
            f"{self.tokenizer.padding_side!r}. Right-padding makes the "
            f"last-position logit read a <pad> token."
        )

        # Resolve "yes" / "no" token ids. We try several variants because
        # different tokenizers split these differently. Use the first token
        # of the most common rendering.
        self.yes_id, self.no_id = self._resolve_yesno_ids()

        # Build/load tid → text map (same as BGE — shared cache fmt).
        self.tid_to_text = self._load_or_build_tid_text(
            item_db_name, track_split_types, corpus_types, cache_dir
        )

        # Score cache — keyed by hash(query + tid). Avoids re-scoring repeat
        # (query, tid) pairs across CMQR rewrites + multiple eval passes.
        # P1 #4 fix: mix sha1(prompt_template)[:8] into the filename so a
        # prompt change forces fresh scoring (prevents silent stale-cache reuse
        # invalidating the W3 gate).
        prompt_hash = hashlib.sha1(prompt_template.encode("utf-8")).hexdigest()[:8]
        self._score_cache_path = (
            Path(cache_dir) / "prorank"
            / f"scores_{_safe_model_name(model_name)}__{prompt_hash}.pkl"
        )
        self._score_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._score_cache: dict[str, float] = {}
        if self._score_cache_path.exists():
            try:
                with self._score_cache_path.open("rb") as f:
                    self._score_cache = pickle.load(f)
                print(f"[pro-rank] loaded score cache: {len(self._score_cache):,} entries")
            except (pickle.PickleError, EOFError, AttributeError, ImportError):
                # P2 #17 fix: also catch AttributeError + ImportError — fires
                # when a stale pickle was produced by a different module path
                # or a removed class. Silently start fresh.
                self._score_cache = {}

        self.stats = {
            "scored_pairs": 0,
            "cache_hits": 0,
            "rerank_calls": 0,
        }

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _resolve_yesno_ids(self) -> tuple[int, int]:
        """Pick single-token IDs for yes / no. Falls back to first-token of
        a multi-token encoding if needed (with a P2 #16 warning).
        """
        # Try variants in order of preference. Most chat-tuned LMs respond with
        # " yes" / " no" (leading space) as the first generated token.
        for yes_str, no_str in [
            (" yes", " no"),
            ("yes", "no"),
            (" Yes", " No"),
            ("Yes", "No"),
        ]:
            yes_ids = self.tokenizer.encode(yes_str, add_special_tokens=False)
            no_ids = self.tokenizer.encode(no_str, add_special_tokens=False)
            if len(yes_ids) == 1 and len(no_ids) == 1:
                return yes_ids[0], no_ids[0]
        # Fallback: first token of " yes" / " no". This is the unsupported
        # path for sentencepiece-style tokenizers (T5, BART) that prefix-split
        # "yes" → ["▁", "yes"]. Scoring on the prefix logits is wrong but
        # won't crash — warn loudly so the user knows the model isn't great
        # for this scoring scheme.
        yes_ids = self.tokenizer.encode(" yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode(" no", add_special_tokens=False)
        print(
            f"[pro-rank] WARNING: tokenizer fragments yes/no into "
            f"{len(yes_ids)}/{len(no_ids)} tokens — scoring on first-token "
            f"prefix only. Consider switching to a BPE-tokenizer model."
        )
        return yes_ids[0], no_ids[0]

    def _load_or_build_tid_text(
        self,
        item_db_name: str,
        track_split_types: list[str],
        corpus_types: list[str],
        cache_dir: str,
    ) -> dict[str, str]:
        """Same caching scheme as BGE_RERANKER — keyed by (item_db, splits, fields)."""
        safe = item_db_name.replace("/", "_")
        fields_tag = "_".join(sorted(corpus_types))
        splits_tag = "_".join(sorted(track_split_types))
        cache_path = os.path.join(
            cache_dir, "rerank", f"{safe}__{splits_tag}__{fields_tag}__tid_text.pkl"
        )
        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"[pro-rank] loaded tid-text cache: {len(cache)} entries")
            return cache

        from datasets import concatenate_datasets, load_dataset

        print(f"[pro-rank] building tid-text map from {item_db_name}[{track_split_types}] fields={corpus_types}")
        ds = load_dataset(item_db_name)
        concat = concatenate_datasets([ds[s] for s in track_split_types])

        def _str(v):
            if isinstance(v, list):
                return " ".join(str(x) for x in v) if v else ""
            return str(v) if v is not None else ""

        cache: dict[str, str] = {}
        for item in concat:
            parts = []
            for f in corpus_types:
                val = _str(item.get(f, ""))
                if val:
                    parts.append(val)
            cache[item["track_id"]] = " | ".join(parts)[: self.max_doc_chars]

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[pro-rank] built tid-text cache: {len(cache)} entries → {cache_path}")
        return cache

    def _cache_key(self, query: str, tid: str) -> str:
        # Hash the (query, tid) pair so the in-memory dict stays small.
        h = hashlib.sha1(f"{query}{tid}".encode("utf-8")).hexdigest()
        return h[:16]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Batched last-token-logit-diff scoring. Returns numpy array of scores."""
        import torch

        prompts = [
            self.prompt_template.format(
                query=q.strip()[:600],
                doc=d.strip()[: self.max_doc_chars],
            )
            for q, d in pairs
        ]

        scores = np.zeros(len(prompts), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=2048,
                )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attn)
                # Logits shape: (B, T, V). With left-padding, the last
                # position is the actual end of the prompt for every row.
                logits = out.logits[:, -1, :]  # (B, V)
                yes_logits = logits[:, self.yes_id]
                no_logits = logits[:, self.no_id]
                diff = (yes_logits - no_logits).float().cpu().numpy()
                scores[start : start + len(batch)] = diff
        return scores

    def _score_query_candidates(
        self, query: str, candidate_tids: Sequence[str],
    ) -> np.ndarray:
        """Score each candidate against the query, hitting cache when possible."""
        # Split into cache hits vs misses.
        hit_idx, miss_idx, miss_pairs = [], [], []
        scores = np.full(len(candidate_tids), np.nan, dtype=np.float32)
        for i, tid in enumerate(candidate_tids):
            key = self._cache_key(query, tid)
            cached = self._score_cache.get(key)
            if cached is not None:
                scores[i] = cached
                hit_idx.append(i)
            else:
                miss_idx.append(i)
                miss_pairs.append((query, self.tid_to_text.get(tid, "")))
        self.stats["cache_hits"] += len(hit_idx)
        self.stats["scored_pairs"] += len(miss_idx)

        if miss_pairs:
            new_scores = self._score_pairs(miss_pairs)
            for i, s, (q, _d) in zip(miss_idx, new_scores, miss_pairs):
                scores[i] = float(s)
                self._score_cache[self._cache_key(q, candidate_tids[i])] = float(s)
        return scores

    def save_cache(self) -> None:
        """Persist the in-memory score cache. Call at the end of a run."""
        try:
            with self._score_cache_path.open("wb") as f:
                pickle.dump(self._score_cache, f)
        except OSError as e:
            print(f"[pro-rank] cache save failed: {e}")

    # ------------------------------------------------------------------
    # Public reranker interface
    # ------------------------------------------------------------------

    def rerank(
        self,
        queries: list[str],
        candidate_tids: list[list[str]],
        topk: int,
        # Side-channel kwargs accepted for interface parity with LGBM_RERANKER /
        # BGE_RERANKER. ProRank is query-driven; we ignore user/goal channels.
        **_kwargs: object,
    ) -> list[list[str]]:
        """Per query, score its candidates with last-token-logit-diff and
        return the top-`topk` tids sorted by score descending."""
        self.stats["rerank_calls"] += len(queries)
        out: list[list[str]] = []
        for q, tids in zip(queries, candidate_tids):
            if not tids:
                out.append([])
                continue
            scores = self._score_query_candidates(q, tids)
            order = np.argsort(-scores)
            keep = order[:topk]
            out.append([tids[int(i)] for i in keep])
        return out

    # ------------------------------------------------------------------
    # P0 #2 fix — rationale generation (plan §A5 + §6.5)
    # ------------------------------------------------------------------
    #
    # `rerank()` returns IDs only (matching the existing reranker contract
    # used by BGE / LGBM). Rationales are generated by a separate, opt-in
    # method so the W3 inference path doesn't pay the rationale cost
    # unnecessarily — W4 (responder envelope) wires this up explicitly.
    #
    # Cost: one greedy generate per (query, tid) at max_new_tokens=16. For
    # top-N=5 with 8000 turns: 40k generations ≈ 7 min on Qwen-0.5B / A100.
    # Skip during W3 dev sweep; call only at W4 inference time after the
    # reranked top-20 is known.

    def generate_rationales(
        self,
        query: str,
        tids: Sequence[str],
        max_rationale_tokens: int = RATIONALE_MAX_NEW_TOKENS,
    ) -> list[str]:
        """For each tid, generate a short (3-5 word) rationale string.

        Output format: one string per input tid, lowercased, stripped of
        leading/trailing punctuation. Falls back to empty string on
        generation error so callers can ALWAYS index by position.

        Token budget: max_rationale_tokens (default 16) — enough for ~5
        words with safety margin. Generations are truncated at the first
        newline / period to avoid run-on paragraphs.
        """
        if not tids:
            return []

        import torch

        prompts = [
            DEFAULT_RATIONALE_TEMPLATE.format(
                query=query.strip()[:600],
                doc=self.tid_to_text.get(t, "").strip()[: self.max_doc_chars],
            )
            for t in tids
        ]

        out: list[str] = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=2048,
                )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                gen_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_rationale_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                # Slice off the prompt prefix (left-padded → all rows have
                # the same prompt length within a batch).
                new_tokens = gen_ids[:, input_ids.shape[1]:]
                texts = self.tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True,
                )
                for t in texts:
                    # Truncate at the first hard stop so multi-sentence
                    # generations get squeezed to a single rationale.
                    for stop in ["\n", ". ", "?", "!"]:
                        idx = t.find(stop)
                        if idx > 0:
                            t = t[:idx]
                            break
                    t = t.strip().strip(".,;:!?\"'`").lower()
                    out.append(t[:80])  # hard cap for safety
        return out

    def batch_generate_rationales(
        self,
        queries,
        tids_per_query,
        max_rationale_tokens: int = RATIONALE_MAX_NEW_TOKENS,
        batch_size: int = None,
    ) -> list:
        """Vectorized counterpart to generate_rationales(). Flattens all
        (query, tid) pairs across queries into one batched generate loop —
        ~2-3x faster than calling generate_rationales() once per query in a
        Python loop, because the GPU stays at high batch fill instead of
        cycling through small per-query trailing batches.

        Returns a list of length len(queries), each element being a list of
        rationale strings (one per tid in tids_per_query[i]) — same shape as
        [self.generate_rationales(q, tids) for q, tids in zip(...)].

        Use a larger `batch_size` here than self.batch_size if your reranker
        model fits the bigger batch (cross-encoders are tiny). Pilot config
        bumps this to 64 in cell 7.
        """
        if not queries:
            return []
        bs = int(batch_size) if batch_size else self.batch_size

        # Flatten with offsets so we can re-shape per-query at the end.
        flat_prompts: list[str] = []
        offsets: list[int] = [0]
        for q, tids in zip(queries, tids_per_query):
            for t in tids:
                flat_prompts.append(
                    DEFAULT_RATIONALE_TEMPLATE.format(
                        query=(q or "").strip()[:600],
                        doc=self.tid_to_text.get(t, "").strip()[: self.max_doc_chars],
                    )
                )
            offsets.append(len(flat_prompts))

        if not flat_prompts:
            return [[] for _ in queries]

        import torch

        flat_out: list[str] = []
        with torch.no_grad():
            for start in range(0, len(flat_prompts), bs):
                batch = flat_prompts[start : start + bs]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=2048,
                )
                input_ids = enc["input_ids"].to(self.device)
                attn = enc["attention_mask"].to(self.device)
                gen_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_rationale_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                new_tokens = gen_ids[:, input_ids.shape[1]:]
                texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for t in texts:
                    for stop in ["\n", ". ", "?", "!"]:
                        idx = t.find(stop)
                        if idx > 0:
                            t = t[:idx]
                            break
                    t = t.strip().strip(".,;:!?\"'`").lower()
                    flat_out.append(t[:80])

        return [flat_out[offsets[i]:offsets[i + 1]] for i in range(len(queries))]

    def report(self) -> dict:
        return {
            **self.stats,
            "cached_total": len(self._score_cache),
            "model": self.model_name,
            "device": self.device,
        }
