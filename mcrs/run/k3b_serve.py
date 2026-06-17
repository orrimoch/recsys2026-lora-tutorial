"""K3b -- serve/submission helper: fine-tuned LoRA adapter -> end-to-end ChainReranker.

Turns a fine-tuned K3 LoRA adapter into a drop-in ChainReranker that stacks on top of an
existing K2 reranker.

Design notes
------------
- `score_fn` is injected so callers can pass a fake scorer in tests (GPU-free).
  Only when `score_fn=None` does the factory call `build_cross_encoder_score_fn`, which
  triggers lazy imports of torch + sentence-transformers + peft.
- `build_finetuned_k3_chain` is the main factory; it wires the adapter provenance into
  NeuralReranker.model_revision so provenance travels with every ranked list (spec §8).
  `adapter` is the local dir or HF Hub repo id; `adapter_revision` optionally pins the Hub
  commit SHA/tag (spec §4.7 train==serve provenance pin).
- `k3b_submission_rows` is a thin convenience wrapper around InferenceHarness; it exists
  so notebooks only need to import one symbol.

See `.claude/documents/features/52_K3_neural_reranker.md` for the wider K3 design.
"""
from __future__ import annotations

from typing import Callable, Optional

from mcrs.contracts import SubmissionRow, TurnContext
from mcrs.rerank.neural import ChainReranker, NeuralReranker
from mcrs.run.harness import InferenceHarness

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]


def build_finetuned_k3_chain(
    catalog,
    query_builder_ce,
    k2_reranker,
    *,
    adapter: str,
    adapter_revision: Optional[str] = None,
    ce_model: str = "BAAI/bge-reranker-v2-m3",
    cross_encoder_k: int = 100,
    max_length: int = 2048,
    max_doc_tokens: int = 1100,
    dtype: str = "auto",
    device: str = "cuda",
    score_fn: Optional[ScoreFn] = None,
) -> ChainReranker:
    """Build a ChainReranker(k2_reranker, NeuralReranker) with a fine-tuned LoRA adapter.

    Parameters
    ----------
    catalog:
        Object with `id_to_metadata(track_id, enriched=True) -> str`.
    query_builder_ce:
        Object with `build(ctx: TurnContext) -> Query` (same interface as the K3 train-time QB).
    k2_reranker:
        Already-built K2 reranker (any object with `.rerank(ctx, cands) -> RankedList`).
    adapter:
        Local directory path or HF Hub repo id of the LoRA adapter.  Forwarded to
        `build_cross_encoder_score_fn` as `lora_adapter`.
    adapter_revision:
        Optional HF Hub revision (git SHA / tag) that pins the exact Hub commit of the
        adapter (spec §4.7).  Forwarded as `lora_revision`; when None the Hub HEAD is loaded.
        Also used as `model_revision` on NeuralReranker for provenance tracking (spec §8);
        falls back to `adapter` when None.
    ce_model:
        Base cross-encoder model name on HF Hub.
    cross_encoder_k:
        How many top candidates to re-score with the cross-encoder (NeuralReranker default=100).
    max_length, max_doc_tokens, dtype, device:
        Forwarded verbatim to `build_cross_encoder_score_fn`; MUST match fine-tune settings.
    score_fn:
        Injected scorer -- skips GPU model loading entirely.  Pass a fake in tests; leave None in
        production.  When None the real `build_cross_encoder_score_fn` is called (lazy import).

    Returns
    -------
    ChainReranker(k2_reranker, neural_reranker)
    """
    if score_fn is None:
        from mcrs.rerank.cross_encoder import build_cross_encoder_score_fn

        score_fn = build_cross_encoder_score_fn(
            ce_model,
            device=device,
            max_length=max_length,
            max_doc_tokens=max_doc_tokens,
            dtype=dtype,
            lora_adapter=adapter,
            lora_revision=adapter_revision,
        )

    neural = NeuralReranker(
        catalog,
        query_builder_ce,
        score_fn,
        cross_encoder_k=cross_encoder_k,
        enriched=True,
        model_revision=adapter_revision or adapter,
    )

    return ChainReranker(k2_reranker, neural)


def k3b_submission_rows(
    query_builder,
    fusion,
    assembler,
    chain: ChainReranker,
    turns,
    topk: int = 20,
) -> list[SubmissionRow]:
    """Run the full inference spine and return SubmissionRows.

    Thin wrapper around InferenceHarness; exists so notebooks import one symbol.

    Parameters
    ----------
    query_builder:
        Query builder for the retrieval/fusion stage (R1).
    fusion:
        Fusion object with `.fuse(queries, topk, ...) -> list[list[Candidate]]`.
    assembler:
        Filter object with `.apply(ranked: RankedList) -> list[str]` (L1 filter).
    chain:
        The ChainReranker produced by `build_finetuned_k3_chain`.
    turns:
        Sequence of TurnContext objects (dev or blind split).
    topk:
        Number of track_ids per row in the submission (default 20 per spec).

    Returns
    -------
    list[SubmissionRow]
    """
    harness = InferenceHarness(
        query_builder,
        fusion,
        assembler,
        reranker=chain,
        topk=topk,
    )
    return harness.run(turns)
