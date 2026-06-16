"""CMQR — Multi-Query Rewriting wrapper around an inner retriever.

Per RecSys_Challenge_Plan §A2 (and CMQR paper, SIGIR 2024).

Pipeline per turn:
  1. Take the user query + the (StateTracker-extracted) user_state JSON.
  2. Single LM call emits N=4 numbered rewrites that inject 1–2 state fields each.
  3. Call the inner retriever once per rewrite at top-K_per_rewrite (default 50).
  4. Fuse the N ranked lists via Reciprocal Rank Fusion (RRF, k=60) — same fn
     `RRF_MODEL` uses internally so the math is consistent across the codebase.
  5. Dedupe (Gap A7 — `reward_fns.dedupe_keep_first` style) and trim to topk.

The implementation is **2-stage RRF**: the inner retriever (e.g. wRRF over BM25
+ dense_metadata + dense_lyrics) does its own per-rewrite RRF, and we layer
another RRF on top across the N rewrites. The plan's literal design calls for
1-stage 12-list RRF — pragmatically equivalent if the inner retriever's weights
are already tuned, and far easier to wire (we don't crack open the wRRF box).
If the W2 gate falls short by a measurable margin, swap to 1-stage by passing
the un-fused per-stream lists from the inner retriever via `inner.peek_subs()`.

Cache: rewrites cached by (session_id, turn_number) at
`{cache_dir}/cmqr/{session_id}__{turn_number}.json`. Idempotent; second call
hits cache.

Backend compatibility: reuses StateTracker's `_detect_backend` pattern so this
works with both LLAMA_MODEL (HF) and VLLM_MODEL (Colab CUDA).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence

# Same allowed keys as StateTracker / scripts/reward_fns.py — keep in sync.
ALLOWED_STATE_KEYS = ("mood", "intent", "energy", "sonic_pref", "era_pref", "familiarity")

# Numbered-rewrite parser — accepts "1. foo", "1) foo", "1: foo".
REWRITE_LINE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+?)\s*$", re.MULTILINE)
# Min/max length sanity — rewrites should be 3-20 words.
MIN_WORDS = 3
MAX_WORDS = 20


def parse_rewrites(text: str, n: int = 4) -> list[str]:
    """Extract numbered rewrites from model output.

    Returns up to `n` non-empty rewrites stripped of leading numbering.
    Tolerates extra surrounding text, missing/duplicate numbers, and noisy
    trailing content. Returns [] if no usable rewrites found.
    """
    if not text:
        return []
    seen_nums: set[int] = set()
    rewrites: list[str] = []
    for m in REWRITE_LINE.finditer(text):
        num = int(m.group(1))
        body = m.group(2).strip()
        if not body or num in seen_nums:
            continue
        # Filter obvious junk: too short / too long.
        wc = len(body.split())
        if wc < MIN_WORDS or wc > MAX_WORDS:
            continue
        seen_nums.add(num)
        rewrites.append(body)
        if len(rewrites) >= n:
            break
    return rewrites


def format_state_for_prompt(state: Optional[dict]) -> str:
    """Format user_state dict for inclusion in the rewriter prompt."""
    if not state:
        return "(none)"
    lines = []
    for k in ALLOWED_STATE_KEYS:
        v = state.get(k)
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str or v_str.lower() == "unknown":
            continue
        lines.append(f"  {k}: {v_str}")
    return "\n".join(lines) if lines else "(none)"


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    topk: int,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> list[str]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Same formula as `RRF_MODEL`: score(d) = Σ_r w_r / (k + rank_r(d)).
    Documents missing from a list contribute 0 from that list.
    Returns top-`topk` deduped document ids by fused score.
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length {len(weights)} != ranked_lists length {len(ranked_lists)}"
        )
    fused: dict[str, float] = {}
    for w, ranks in zip(weights, ranked_lists):
        for rank, tid in enumerate(ranks, start=1):
            fused[tid] = fused.get(tid, 0.0) + float(w) / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [tid for tid, _ in ordered[:topk]]


def dedupe_keep_first(ids: Sequence[str]) -> list[str]:
    """Mirror of scripts/reward_fns.dedupe_keep_first — kept local so this
    module doesn't import from /scripts/ (mcrs is a self-contained package)."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Backend detection (same pattern as StateTracker._detect_backend)
# ---------------------------------------------------------------------------

def _detect_backend(lm) -> str:
    """Return 'vllm' if the wrapper looks like VLLM_MODEL, else 'hf'."""
    cls_name = type(lm).__name__.lower()
    if "vllm" in cls_name:
        return "vllm"
    return "hf"


# ---------------------------------------------------------------------------
# CMQR_REWRITER — wraps an inner retriever, fuses N rewrite outputs.
# ---------------------------------------------------------------------------

class CMQR_REWRITER:
    """Multi-Query Rewriter that fuses N ranked lists from an inner retriever.

    Args:
        lm: LM wrapper (LLAMA_MODEL or VLLM_MODEL). Reused from CRS_BASELINE
            so we don't load the rewriter weights twice.
        inner_retriever: any retriever exposing
            `batch_text_to_item_retrieval(queries, topk, user_ids=None)`.
        prompt_path: path to cmqr_rewrites.txt.
        cache_dir: parent dir; rewrites cached at `{cache_dir}/cmqr/`.
        n_rewrites: how many rewrites to emit per query. Plan default 4.
        topk_per_rewrite: how many candidates each rewrite pulls from the
            inner retriever before fusion. Plan default 50.
        rrf_k: RRF k parameter. Plan default 60.
        max_new_tokens: cap for the rewriter generation. 4 rewrites × ~10
            tokens × ~1.3 token/word ≈ 60; we use 96 for safety.
        debug: capture raw rewrite outputs in `self.failures` for diagnosis.

    Usage in CRS_BASELINE:
        cmqr = CMQR_REWRITER(lm, inner_retriever, prompt_path, cache_dir)
        cmqr.set_batch_context(session_ids, turn_numbers, extracted_states)
        top100 = cmqr.batch_text_to_item_retrieval(queries, topk=100, user_ids=...)
    """

    def __init__(
        self,
        lm,
        inner_retriever,
        prompt_path: str | Path,
        cache_dir: str | Path = "./cache",
        n_rewrites: int = 4,
        topk_per_rewrite: int = 50,
        rrf_k: int = 60,
        max_new_tokens: int = 96,
        debug: bool = False,
    ):
        self.lm = lm
        self.inner = inner_retriever
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.cache_root = Path(cache_dir) / "cmqr"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.n_rewrites = int(n_rewrites)
        self.topk_per_rewrite = int(topk_per_rewrite)
        self.rrf_k = int(rrf_k)
        self.max_new_tokens = int(max_new_tokens)
        self.debug = bool(debug)
        self._backend = _detect_backend(lm)

        # Per-call context (set via set_batch_context).
        self._ctx_session_ids: list[Optional[str]] = []
        self._ctx_turn_numbers: list[Optional[int]] = []
        self._ctx_states: list[Optional[dict]] = []

        # Stats / diagnostics.
        self.failures: list[dict] = []
        self.stats = {
            "calls": 0,
            "cache_hits": 0,
            "rewrites_ok": 0,
            "rewrites_partial": 0,   # N_emitted < n_rewrites
            "rewrites_dropped": 0,    # 0 valid → fallback to original query only
        }

    # -------- Context plumbing ---------------------------------------------

    def set_batch_context(
        self,
        session_ids: Sequence[Optional[str]],
        turn_numbers: Sequence[Optional[int]],
        extracted_states: Sequence[Optional[dict]],
    ) -> None:
        """Provide per-row session_id / turn_number / state for the next batch call.

        Lengths must match the number of queries that follow. Caller (CRS_BASELINE)
        sets this once per `batch_chat`. CMQR uses session_id/turn_number for
        cache lookup and state for prompt injection.
        """
        n = len(session_ids)
        if len(turn_numbers) != n or len(extracted_states) != n:
            raise ValueError(
                f"context lengths must match: got "
                f"sessions={len(session_ids)} turns={len(turn_numbers)} "
                f"states={len(extracted_states)}"
            )
        self._ctx_session_ids = list(session_ids)
        self._ctx_turn_numbers = list(turn_numbers)
        self._ctx_states = list(extracted_states)

    def _get_ctx(self, idx: int) -> tuple[Optional[str], Optional[int], Optional[dict]]:
        if idx >= len(self._ctx_session_ids):
            return None, None, None
        return (
            self._ctx_session_ids[idx],
            self._ctx_turn_numbers[idx],
            self._ctx_states[idx],
        )

    # -------- Caching ------------------------------------------------------

    def _cache_path(self, session_id: str, turn_number: int) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", str(session_id))
        return self.cache_root / f"{safe}__{int(turn_number):d}.json"

    def _load_cached(self, session_id: str, turn_number: int) -> Optional[list[str]]:
        cp = self._cache_path(session_id, turn_number)
        if cp.exists():
            try:
                with cp.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    rewrites = payload.get("rewrites")
                    if isinstance(rewrites, list) and rewrites:
                        return list(rewrites)
            except (OSError, json.JSONDecodeError):
                pass
        return None

    def _save_cached(
        self,
        session_id: str,
        turn_number: int,
        rewrites: list[str],
        original_query: str,
    ) -> None:
        cp = self._cache_path(session_id, turn_number)
        with cp.open("w", encoding="utf-8") as f:
            json.dump({
                "rewrites": rewrites,
                "n": len(rewrites),
                "original_query": original_query,
            }, f, ensure_ascii=False, indent=2)

    # -------- LM generation ------------------------------------------------

    def _build_user_message(self, user_query: str, state: Optional[dict]) -> str:
        return (
            f'user query: "{user_query.strip()}"\n'
            f"user_state:\n{format_state_for_prompt(state)}"
        )

    def _generate(self, user_query: str, state: Optional[dict]) -> str:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": self._build_user_message(user_query, state)},
        ]
        if self._backend == "vllm":
            return self._generate_vllm(messages)
        return self._generate_hf(messages)

    def _generate_hf(self, messages: list[dict]) -> str:
        """HF transformers path — uses LLAMA_MODEL's exposed tokenizer + model."""
        tokenizer = self.lm.tokenizer
        model = self.lm.lm
        device = self.lm.device

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # Pre-prime the assistant turn with "1." so the model continues from
        # the numbered-rewrite shape — same trick StateTracker uses for the
        # `<user_state>` opener (Qwen-1.5B sometimes wanders into prose).
        prompt_text = prompt_text + "1."

        token_inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(device)
        attention_mask = token_inputs.attention_mask.to(device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = outputs[:, input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        # Re-prepend the prefix so parse_rewrites sees the full "1. ..." line.
        return "1." + decoded

    def _generate_vllm(self, messages: list[dict]) -> str:
        """vLLM path — uses lm.lm.generate() with SamplingParams.

        Untested on real CUDA in this venv; mirrors StateTracker's vLLM path.
        Verified by inspection only — production use should run a Colab smoke.
        """
        tokenizer = self.lm.tokenizer
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_text = prompt_text + "1."
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        outputs = self.lm.lm.generate([prompt_text], sampling_params=params)
        decoded = outputs[0].outputs[0].text
        return "1." + decoded

    # -------- Per-query: emit + fuse ---------------------------------------

    def _emit_rewrites(self, idx: int, query: str) -> list[str]:
        """Return the N rewrites for this query, hitting cache when available.

        On parse failure: log to self.failures and fall back to [query] (so
        downstream still gets a valid query list — degraded to 1-rewrite mode).
        """
        sid, tn, state = self._get_ctx(idx)
        cached = self._load_cached(sid, tn) if sid and tn else None
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        text = self._generate(query, state)
        rewrites = parse_rewrites(text, n=self.n_rewrites)

        if not rewrites:
            self.stats["rewrites_dropped"] += 1
            if self.debug:
                self.failures.append({
                    "session_id": sid, "turn_number": tn,
                    "user_query": query[:160], "raw_output": text[:300],
                })
            # Degrade to original query so retrieval still runs.
            rewrites = [query]
        elif len(rewrites) < self.n_rewrites:
            self.stats["rewrites_partial"] += 1
        else:
            self.stats["rewrites_ok"] += 1

        # Persist (even partial / fallback) so we don't re-pay the LM cost.
        if sid and tn:
            self._save_cached(sid, tn, rewrites, query)
        return rewrites

    # -------- Public retriever interface ----------------------------------

    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
        user_ids: Optional[list] = None,
        # Accept (but currently ignore) batch_context so CRS_BASELINE.batch_chat's
        # primary call path doesn't TypeError out and force a fallback that drops
        # the context entirely. Forwarding to the inner retriever happens in the
        # try/except block below — when the inner's batch is built from
        # flat_rewrites, each rewrite uses the original query's batch_context entry.
        # Bug surfaced 2026-05-17 in first end-to-end run of notebook 63.
        batch_context: Optional[list[dict]] = None,
    ) -> list[list[str]]:
        """Retriever-shape interface: for each query, return top-`topk` track ids.

        Per query:
          1. emit N rewrites (cached by session_id/turn_number)
          2. inner retriever → top-`topk_per_rewrite` per rewrite (one batched call)
          3. RRF-fuse the N lists → trim to topk → dedupe
        """
        if not queries:
            return []
        self.stats["calls"] += len(queries)

        # Step 1 — emit rewrites per query (this is the LM-bound step).
        per_query_rewrites: list[list[str]] = []
        flat_rewrites: list[str] = []
        flat_user_ids: list[Any] = []
        rewrite_owner_idx: list[int] = []  # which query each flat rewrite belongs to
        for q_idx, q in enumerate(queries):
            rs = self._emit_rewrites(q_idx, q)
            per_query_rewrites.append(rs)
            for r in rs:
                flat_rewrites.append(r)
                flat_user_ids.append(user_ids[q_idx] if user_ids else None)
                rewrite_owner_idx.append(q_idx)

        # Step 2 — single batched call to the inner retriever for ALL flat rewrites.
        # This minimises the number of round-trips through dense encoders / BM25
        # bookkeeping. user_ids threaded through for cf-bpr-style retrievers.
        # batch_context is duplicated across each rewrite of the same query so
        # SID_GENERATOR (downstream of inner) sees its training-format context.
        flat_batch_context: Optional[list[dict]] = None
        if batch_context is not None:
            flat_batch_context = []
            for q_idx in range(len(queries)):
                ctx = batch_context[q_idx]
                for _ in per_query_rewrites[q_idx]:
                    flat_batch_context.append(ctx)
        try:
            flat_results = self.inner.batch_text_to_item_retrieval(
                flat_rewrites,
                topk=self.topk_per_rewrite,
                user_ids=flat_user_ids if user_ids else None,
                batch_context=flat_batch_context,
            )
        except TypeError:
            # Back-compat: inner predates user_ids/batch_context kwargs.
            try:
                flat_results = self.inner.batch_text_to_item_retrieval(
                    flat_rewrites,
                    topk=self.topk_per_rewrite,
                    user_ids=flat_user_ids if user_ids else None,
                )
            except TypeError:
                flat_results = self.inner.batch_text_to_item_retrieval(
                    flat_rewrites, topk=self.topk_per_rewrite,
                )

        # Step 3 — partition flat_results back per query and RRF-fuse.
        partitioned: list[list[list[str]]] = [[] for _ in queries]
        for owner, ranks in zip(rewrite_owner_idx, flat_results):
            partitioned[owner].append(ranks)

        out: list[list[str]] = []
        for q_idx, lists in enumerate(partitioned):
            if not lists:
                out.append([])
                continue
            fused = rrf_fuse(lists, topk=topk, k=self.rrf_k)
            out.append(dedupe_keep_first(fused))
        return out

    def text_to_item_retrieval(
        self, query: str, topk: int, user_id=None,
    ) -> list[str]:
        """Single-query convenience wrapper — same shape as the inner retriever."""
        return self.batch_text_to_item_retrieval(
            [query], topk=topk,
            user_ids=[user_id] if user_id is not None else None,
        )[0]

    # -------- Stats helpers -----------------------------------------------

    def report(self) -> dict:
        total = self.stats["calls"] - self.stats["cache_hits"]
        if total > 0:
            full_rate = self.stats["rewrites_ok"] / total
        else:
            full_rate = 1.0
        return {
            **self.stats,
            "full_rewrite_rate": round(full_rate, 4),
            "n_rewrites_target": self.n_rewrites,
            "topk_per_rewrite": self.topk_per_rewrite,
            "rrf_k": self.rrf_k,
        }
