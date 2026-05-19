"""Tests for NV-Retriever-style hard-negative miner."""
import numpy as np
import pytest


def test_percpos_filter_drops_above_threshold():
    """Candidates scoring >= threshold * positive_score are dropped (false-neg guard)."""
    from mcrs.retrieval_modules.hn_miner import percpos_filter
    candidate_scores = [0.95, 0.82, 0.75, 0.60, 0.40]
    positive_score = 1.00
    kept = percpos_filter(candidate_scores, positive_score, threshold=0.80)
    assert kept == [False, False, True, True, True]  # 0.95 and 0.82 dropped


def test_percpos_filter_keeps_everything_at_threshold_1_0():
    """threshold=1.0 only drops candidates that exactly tie the positive."""
    from mcrs.retrieval_modules.hn_miner import percpos_filter
    kept = percpos_filter([0.99, 0.50], positive_score=1.00, threshold=1.0)
    assert kept == [True, True]


def test_sample_hard_negatives_respects_rank_range_and_cap():
    """sample_hard_negatives draws from the filtered top-K within rank_range."""
    from mcrs.retrieval_modules.hn_miner import sample_hard_negatives
    filtered_ranks = [5, 7, 10, 12, 15, 20, 30, 40, 50, 80]
    drawn = sample_hard_negatives(filtered_ranks, k=4, rank_range=(2, 200), seed=42)
    assert len(drawn) == 4
    for rank in drawn:
        assert 2 <= rank <= 200
        assert rank in filtered_ranks


def test_sample_hard_negatives_returns_all_when_pool_smaller_than_k():
    """If only 3 negatives survive the filter but k=10, return all 3 (no padding)."""
    from mcrs.retrieval_modules.hn_miner import sample_hard_negatives
    drawn = sample_hard_negatives([5, 10, 15], k=10, rank_range=(2, 200), seed=42)
    assert len(drawn) == 3


def test_mine_negatives_for_query_returns_negs_aligned_to_track_ids():
    """End-to-end mine: returns list of track_ids (strings), not indices."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t2", "t3", "t4", "t5"]
    query_emb = np.array([1.0, 0.0])
    track_embs = np.array([
        [1.0, 0.0],   # t1 — gold
        [0.95, 0.31], # t2 — hard near-gold
        [0.80, 0.60], # t3 — medium
        [0.50, 0.87], # t4 — far
        [0.0, 1.0],   # t5 — orthogonal
    ])
    track_embs = track_embs / np.linalg.norm(track_embs, axis=1, keepdims=True)
    negs = mine_negatives_for_query(
        query_emb=query_emb, track_embs=track_embs, track_ids=track_ids,
        gold_track_id="t1", percpos_threshold=0.97, k_negs=2, seed=42,
    )
    assert len(negs) <= 2
    assert "t1" not in negs
    for n in negs:
        assert n in track_ids


def test_mine_negatives_raises_on_duplicate_track_ids():
    """Catch silent label corruption: duplicate gold ID would leak as a negative."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t1", "t3"]  # duplicate
    query_emb = np.array([1.0, 0.0])
    track_embs = np.array([[1.0, 0.0], [0.9, 0.43], [0.0, 1.0]])
    track_embs = track_embs / np.linalg.norm(track_embs, axis=1, keepdims=True)
    with pytest.raises(ValueError, match="track_ids must be unique"):
        mine_negatives_for_query(
            query_emb=query_emb, track_embs=track_embs, track_ids=track_ids,
            gold_track_id="t1", k_negs=2,
        )


def test_mine_negatives_raises_on_non_unit_normed_track_embs():
    """Catch silent semantic bugs: cosine sim assumes unit-norm rows."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t2", "t3"]
    query_emb = np.array([1.0, 0.0])
    track_embs = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # NOT unit-normed
    with pytest.raises(ValueError, match="unit-normed"):
        mine_negatives_for_query(
            query_emb=query_emb, track_embs=track_embs, track_ids=track_ids,
            gold_track_id="t1", k_negs=2,
        )


def test_mine_negatives_k_negs_zero_returns_empty():
    """k_negs=0 is a valid no-op (contract pin)."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t2", "t3"]
    query_emb = np.array([1.0, 0.0])
    track_embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    track_embs = track_embs / np.linalg.norm(track_embs, axis=1, keepdims=True)
    result = mine_negatives_for_query(
        query_emb=query_emb, track_embs=track_embs, track_ids=track_ids,
        gold_track_id="t1", k_negs=0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# SimANS sampler [Zhou et al. EMNLP 2022, arXiv 2210.11773]
# ---------------------------------------------------------------------------


def test_sample_simans_returns_k_indices_in_pool_range():
    from mcrs.retrieval_modules.hn_miner import sample_simans_negatives
    pool_scores = [0.95, 0.80, 0.65, 0.50, 0.35, 0.20]
    chosen = sample_simans_negatives(pool_scores, positive_score=1.0, k=3, seed=42)
    assert len(chosen) == 3
    assert all(0 <= i < len(pool_scores) for i in chosen)
    assert len(set(chosen)) == 3  # no replacement


def test_sample_simans_caps_at_pool_size():
    """If k > pool size, return all pool indices (no padding)."""
    from mcrs.retrieval_modules.hn_miner import sample_simans_negatives
    chosen = sample_simans_negatives([0.5, 0.3], positive_score=1.0, k=10, seed=42)
    assert len(chosen) == 2


def test_sample_simans_returns_empty_for_empty_pool():
    from mcrs.retrieval_modules.hn_miner import sample_simans_negatives
    assert sample_simans_negatives([], positive_score=1.0, k=5, seed=42) == []


def test_sample_simans_is_reproducible_with_seed():
    from mcrs.retrieval_modules.hn_miner import sample_simans_negatives
    scores = [0.9, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45]
    a = sample_simans_negatives(scores, positive_score=1.0, k=4, seed=123)
    b = sample_simans_negatives(scores, positive_score=1.0, k=4, seed=123)
    assert a == b


def test_sample_simans_biases_toward_target_difficulty():
    """Weight peak is at s_pos - a; with a=0.1, samples concentrate near s=0.9
    when s_pos=1.0. Statistical check via Monte Carlo over many seeds."""
    from mcrs.retrieval_modules.hn_miner import sample_simans_negatives
    # 21 candidates evenly spaced from s=1.0 down to s=0.0
    pool_scores = [round(1.0 - 0.05 * i, 4) for i in range(21)]
    # 200 trials, k=5 each — distribution should peak at indices near s=0.9
    counts = [0] * len(pool_scores)
    for seed in range(200):
        chosen = sample_simans_negatives(pool_scores, positive_score=1.0,
                                         k=5, a=0.1, b=0.05, seed=seed)
        for i in chosen:
            counts[i] += 1
    # Index 2 corresponds to score 0.90 = peak weight. Allow ±2 indices around.
    peak_band = sum(counts[1:4])  # indices 1, 2, 3 → scores 0.95, 0.90, 0.85
    far_band = sum(counts[10:])   # indices 10+ → scores ≤ 0.50 (low weight)
    assert peak_band > far_band, (
        f"SimANS weighting should concentrate near s_pos - a = 0.9, but got "
        f"peak_band={peak_band} vs far_band={far_band}"
    )


def test_sample_simans_handles_underflow_with_uniform_fallback():
    """When all weights underflow to 0 (extreme b), fall back to uniform sampling
    so the miner doesn't silently return empty for pathological inputs."""
    from mcrs.retrieval_modules.hn_miner import sample_simans_negatives
    # b extremely small → exp(-large/tiny) underflows to 0 for all candidates
    # except those exactly at the target. Force the degenerate path.
    pool_scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    chosen = sample_simans_negatives(pool_scores, positive_score=1.0,
                                     k=3, a=0.05, b=1e-300, seed=42)
    assert len(chosen) == 3
    assert all(0 <= i < len(pool_scores) for i in chosen)


# ---------------------------------------------------------------------------
# strategy="simans" path through mine_negatives_for_query
# ---------------------------------------------------------------------------


def test_mine_negatives_simans_strategy_returns_track_ids():
    """End-to-end SimANS mine: returns list of track_ids, never includes gold."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = [f"t{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(10, 16)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    q = embs[0]  # query equals catalog[0] = gold
    negs = mine_negatives_for_query(
        query_emb=q, track_embs=embs, track_ids=track_ids,
        gold_track_id="t0", k_negs=4, pool_size=8,
        strategy="simans", simans_a=0.1, simans_b=0.05, seed=42,
    )
    assert len(negs) == 4
    assert "t0" not in negs
    for n in negs:
        assert n in track_ids


def test_mine_negatives_simans_includes_queries_percpos_would_skip():
    """The motivating case: when positive_score is low (BGE-M3 doesn't
    strongly endorse gold), percpos drops the query for lack of negatives.
    SimANS must still return k negatives — that's the whole point."""
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    # Construct a catalog where almost everything scores ~0.99 against the
    # query, including the gold. percpos at 0.95 would reject all but the
    # few far-distant candidates → likely < k_negs survive.
    track_ids = [f"t{i}" for i in range(20)]
    embs = np.zeros((20, 4), dtype=np.float32)
    # The query direction
    embs[0] = [1.0, 0.0, 0.0, 0.0]  # gold, sim=1.0
    for i in range(1, 18):
        # Almost-aligned with the query (sim ~0.99-0.995)
        embs[i] = [0.99 + 0.0003 * i, 0.05, 0.0, 0.0]
    embs[18] = [0.5, 0.5, 0.5, 0.5]
    embs[19] = [0.0, 0.0, 1.0, 0.0]
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    q = embs[0]
    negs = mine_negatives_for_query(
        query_emb=q, track_embs=embs, track_ids=track_ids,
        gold_track_id="t0", k_negs=8, pool_size=19,
        strategy="simans", seed=42,
    )
    assert len(negs) == 8  # SimANS always fills k_negs from the pool
    assert "t0" not in negs


def test_mine_negatives_rejects_unknown_strategy():
    from mcrs.retrieval_modules.hn_miner import mine_negatives_for_query
    track_ids = ["t1", "t2"]
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="unknown mining strategy"):
        mine_negatives_for_query(
            query_emb=np.array([1.0, 0.0]), track_embs=embs,
            track_ids=track_ids, gold_track_id="t1", k_negs=1,
            strategy="random",
        )
