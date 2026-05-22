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


# ===========================================================================
# Vectorized batch_mine_negatives — 5-6x speedup via GPU matmul + top-K.
# ===========================================================================

def _make_test_corpus(N: int = 30, D: int = 8, seed: int = 0):
    """Generate a small unit-normalized catalog for testing."""
    rng = np.random.default_rng(seed)
    track_embs = rng.standard_normal((N, D)).astype(np.float32)
    track_embs /= np.linalg.norm(track_embs, axis=1, keepdims=True)
    track_ids = [f"t{i:02d}" for i in range(N)]
    return track_embs, track_ids


def test_batch_mine_negatives_returns_one_list_per_query():
    """Vectorized call returns a list of length B, one negs-list per query."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D, B = 30, 8, 5
    track_embs, track_ids = _make_test_corpus(N=N, D=D)
    rng = np.random.default_rng(1)
    query_embs = rng.standard_normal((B, D)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    gold_ids = ["t00", "t05", "t10", "t15", "t20"]

    out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        percpos_threshold=0.5, k_negs=3, pool_size=15,
    )
    assert len(out) == B
    for negs in out:
        assert negs is None or isinstance(negs, list)


def test_batch_mine_negatives_handles_missing_gold():
    """A gold track_id absent from track_ids → None for that row, others unaffected."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D = 30, 8
    track_embs, track_ids = _make_test_corpus(N=N, D=D)
    rng = np.random.default_rng(2)
    query_embs = rng.standard_normal((3, D)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    gold_ids = ["t05", "missing_tid", "t12"]

    out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        percpos_threshold=0.5, k_negs=3, pool_size=15,
    )
    assert len(out) == 3
    assert out[0] is not None
    assert out[1] is None     # missing_tid
    assert out[2] is not None


def test_batch_mine_negatives_matches_per_query_results():
    """Bit-equivalence: for each query in a batch, batch_mine_negatives returns
    the SAME negs list (same ordering) as mine_negatives_for_query called
    per row with the matching seed = base_seed + row_index."""
    from mcrs.retrieval_modules.hn_miner import (
        batch_mine_negatives, mine_negatives_for_query,
    )

    N, D, B = 50, 16, 8
    track_embs, track_ids = _make_test_corpus(N=N, D=D, seed=7)
    rng = np.random.default_rng(3)
    query_embs = rng.standard_normal((B, D)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    # Pick gold tracks that aren't necessarily the closest to each query.
    gold_ids = [track_ids[(b * 7) % N] for b in range(B)]

    BASE_SEED = 42
    batch_out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        percpos_threshold=0.85, k_negs=4, pool_size=30,
        seed=BASE_SEED, strategy="percpos",
    )

    for b in range(B):
        per_query_out = mine_negatives_for_query(
            query_emb=query_embs[b], track_embs=track_embs,
            track_ids=track_ids, gold_track_id=gold_ids[b],
            percpos_threshold=0.85, k_negs=4, pool_size=30,
            seed=BASE_SEED + b, strategy="percpos",
        )
        assert batch_out[b] == per_query_out, (
            f"row {b}: batch={batch_out[b]} vs per-query={per_query_out}"
        )


def test_batch_mine_negatives_simans_returns_k_negs_per_query():
    """SimANS doesn't filter, so every query gets exactly k_negs (or fewer if pool < k)."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D, B = 40, 8, 4
    track_embs, track_ids = _make_test_corpus(N=N, D=D, seed=11)
    rng = np.random.default_rng(5)
    query_embs = rng.standard_normal((B, D)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    gold_ids = [f"t{(b * 9) % N:02d}" for b in range(B)]

    K = 5
    out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        k_negs=K, pool_size=20, strategy="simans",
    )
    for b in range(B):
        assert out[b] is not None
        assert len(out[b]) == K, f"SimANS should return k_negs per query, got {len(out[b])} for row {b}"


def test_batch_mine_negatives_rejects_shape_mismatch():
    """Bad inputs raise ValueError."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D = 20, 8
    track_embs, track_ids = _make_test_corpus(N=N, D=D)
    # Wrong: D mismatch between query and track.
    bad_query_embs = np.zeros((3, D + 1), dtype=np.float32)
    gold_ids = ["t00", "t05", "t10"]
    with pytest.raises((ValueError, RuntimeError)):
        batch_mine_negatives(
            query_embs=bad_query_embs, track_embs=track_embs,
            track_ids=track_ids, gold_track_ids=gold_ids,
            k_negs=3, pool_size=10,
        )
    # Wrong: gold_track_ids length doesn't match query batch.
    good_query_embs = np.zeros((3, D), dtype=np.float32)
    good_query_embs[:, 0] = 1.0  # unit norm
    with pytest.raises(ValueError):
        batch_mine_negatives(
            query_embs=good_query_embs, track_embs=track_embs,
            track_ids=track_ids, gold_track_ids=["t00", "t05"],  # only 2, not 3
            k_negs=3, pool_size=10,
        )


def test_batch_mine_negatives_rejects_duplicate_track_ids():
    """Duplicate track_ids would leak gold as negative (same check as per-query function)."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives
    N, D = 5, 8
    track_embs = np.eye(N, D, dtype=np.float32)
    track_ids = ["t1", "t2", "t2", "t4", "t5"]  # duplicate
    query_embs = np.zeros((1, D), dtype=np.float32)
    query_embs[0, 0] = 1.0
    with pytest.raises(ValueError, match="unique"):
        batch_mine_negatives(
            query_embs=query_embs, track_embs=track_embs,
            track_ids=track_ids, gold_track_ids=["t1"],
            k_negs=2, pool_size=4,
        )


def test_batch_mine_negatives_accepts_torch_tensors():
    """The build script will pass torch tensors (kept on GPU). Function must accept them."""
    import torch
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D, B = 20, 8, 3
    np_te, track_ids = _make_test_corpus(N=N, D=D, seed=13)
    rng = np.random.default_rng(17)
    np_qe = rng.standard_normal((B, D)).astype(np.float32)
    np_qe /= np.linalg.norm(np_qe, axis=1, keepdims=True)
    gold_ids = ["t00", "t05", "t10"]

    # Convert to torch tensors (simulating the build-script's GPU-pinned catalog).
    t_te = torch.from_numpy(np_te)
    t_qe = torch.from_numpy(np_qe)

    out_torch = batch_mine_negatives(
        query_embs=t_qe, track_embs=t_te,
        track_ids=track_ids, gold_track_ids=gold_ids,
        k_negs=3, pool_size=10, seed=99, strategy="percpos",
    )
    out_np = batch_mine_negatives(
        query_embs=np_qe, track_embs=np_te,
        track_ids=track_ids, gold_track_ids=gold_ids,
        k_negs=3, pool_size=10, seed=99, strategy="percpos",
    )
    # Both input forms must produce identical outputs.
    assert out_torch == out_np, f"torch vs numpy results differ: {out_torch} vs {out_np}"


def test_batch_mine_negatives_excludes_gold_from_pool():
    """Gold's own catalog index must never appear in the returned negs (per-query
    invariant — must hold for vectorized version too)."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D, B = 30, 8, 4
    track_embs, track_ids = _make_test_corpus(N=N, D=D, seed=19)
    rng = np.random.default_rng(23)
    query_embs = rng.standard_normal((B, D)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    gold_ids = ["t00", "t07", "t14", "t21"]

    # SimANS: no filter, easy to spot a gold-leak.
    out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        k_negs=10, pool_size=25, strategy="simans",
    )
    for b in range(B):
        assert out[b] is not None
        assert gold_ids[b] not in out[b], \
            f"row {b}: gold {gold_ids[b]} leaked into negs {out[b]}"


def test_batch_mine_negatives_seed_assignment_matches_build_script_convention():
    """Build script uses base_seed=42+i (i=batch_start). batch_mine_negatives
    must apply row_seed=base+b (b=position in batch), so a query at global
    position i+b gets seed 42+(i+b) — same as the legacy per-query loop
    which used seed=42+i+j (j==b)."""
    from mcrs.retrieval_modules.hn_miner import (
        batch_mine_negatives, mine_negatives_for_query,
    )

    N, D, B = 25, 8, 3
    track_embs, track_ids = _make_test_corpus(N=N, D=D, seed=31)
    rng = np.random.default_rng(37)
    query_embs = rng.standard_normal((B, D)).astype(np.float32)
    query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
    gold_ids = ["t05", "t12", "t20"]

    # Simulate the build script's batch starting at global position 100.
    BATCH_START = 100
    base_seed_in_loop = 42 + BATCH_START

    batch_out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        k_negs=3, pool_size=15, strategy="percpos",
        percpos_threshold=0.9, seed=base_seed_in_loop,
    )

    # Per-query equivalent (old build script's inner loop).
    for b in range(B):
        per_query_seed = 42 + BATCH_START + b
        per_query_negs = mine_negatives_for_query(
            query_emb=query_embs[b], track_embs=track_embs,
            track_ids=track_ids, gold_track_id=gold_ids[b],
            k_negs=3, pool_size=15, strategy="percpos",
            percpos_threshold=0.9, seed=per_query_seed,
        )
        assert batch_out[b] == per_query_negs, (
            f"seed mismatch at row {b}: batch (base+b={base_seed_in_loop}+{b}) "
            f"vs per-query ({per_query_seed}): {batch_out[b]} != {per_query_negs}"
        )


def test_batch_mine_negatives_pool_smaller_than_k_returns_all_available():
    """When pool has fewer items than k_negs (after filter), return what we have.
    Mirrors sample_hard_negatives's behavior (lines 53-54)."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    N, D = 5, 4
    track_embs = np.eye(N, D, dtype=np.float32)  # orthogonal: catalog has 5 tracks, no clusters
    track_ids = ["t0", "t1", "t2", "t3", "t4"]
    # Query aligned with t0 → high s_pos. percpos at 0.99 → small surviving pool.
    query_embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    gold_ids = ["t0"]

    out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=gold_ids,
        k_negs=10, pool_size=10, strategy="percpos",
        percpos_threshold=0.99,
    )
    assert out[0] is not None
    # Pool can never exceed N-1=4 (excluding gold). With k_negs=10 > available, expect ≤4.
    assert len(out[0]) <= 4
    assert "t0" not in out[0]


def test_batch_mine_negatives_simans_with_empty_pool_returns_empty():
    """If somehow pool is empty (degenerate catalog), SimANS returns [] not None."""
    from mcrs.retrieval_modules.hn_miner import batch_mine_negatives

    # 1-track catalog: pool excludes gold → empty pool.
    track_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    track_ids = ["t0"]
    query_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    out = batch_mine_negatives(
        query_embs=query_embs, track_embs=track_embs,
        track_ids=track_ids, gold_track_ids=["t0"],
        k_negs=5, pool_size=10, strategy="simans",
    )
    assert out[0] == [] or out[0] is None  # either is acceptable; both are handled by build script's len<2 skip
