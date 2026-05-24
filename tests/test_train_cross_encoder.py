"""Unit tests for the Stage B cross-encoder training helpers (Phase 6c)."""
import torch

from scripts.train_cross_encoder import flatten_ce_pairs, pairwise_bce_loss


def test_pairwise_bce_lower_for_correct_ranking():
    """Loss is lower when positives score high and negatives score low than
    when the predictions are inverted — confirms the loss drives the intended
    BCE(pos→1) + BCE(neg→0) objective."""
    pos_logits = torch.tensor([5.0, 4.0])
    neg_logits = torch.tensor([-5.0, -4.0, -3.0])

    good = pairwise_bce_loss(pos_logits, neg_logits)
    bad = pairwise_bce_loss(-pos_logits, -neg_logits)  # inverted predictions

    assert torch.isfinite(good)
    assert good < bad


def test_pairwise_bce_neg_weights_scale_negative_term():
    """Per-negative weights scale the negative BCE term — zero-weighting all
    negatives leaves only the positive term (used later for v2 rank-weighting)."""
    pos_logits = torch.tensor([2.0])
    neg_logits = torch.tensor([2.0, 2.0])

    unweighted = pairwise_bce_loss(pos_logits, neg_logits)
    zero_negs = pairwise_bce_loss(
        pos_logits, neg_logits, neg_weights=torch.zeros_like(neg_logits)
    )

    # Zeroing the negative weights must drop the (positive) negative-term loss.
    assert zero_negs < unweighted
    assert torch.isfinite(zero_negs)


def test_flatten_ce_pairs_expands_row_into_pos_and_neg_pairs():
    """Each TripleJsonlDataset row (1 gold + K negs + modalities) flattens into
    K+1 (query, doc) scoring pairs with aligned modalities and a pos/neg flag.
    Query + user_cf are repeated across the row's pairs."""
    rows = [{
        "query": "q1",
        "positive": "POS",
        "negatives": ["NEG0", "NEG1"],
        "pos_clap": [0.1], "pos_cf_track": [0.2],
        "neg_clap": [[0.3], [0.4]], "neg_cf_track": [[0.5], [0.6]],
        "user_cf": [0.9],
        "pos_tag_ids": [1, 2], "pos_year": 2000,
        "neg_tag_ids": [[3], [4]], "neg_years": [1990, 1995],
    }]

    flat = flatten_ce_pairs(rows)

    assert flat["doc_text"] == ["POS", "NEG0", "NEG1"]
    assert flat["is_positive"] == [True, False, False]
    assert flat["query"] == ["q1", "q1", "q1"]
    assert flat["user_cf"] == [[0.9], [0.9], [0.9]]
    assert flat["doc_clap"] == [[0.1], [0.3], [0.4]]
    assert flat["doc_cf"] == [[0.2], [0.5], [0.6]]
    assert flat["doc_tags"] == [[1, 2], [3], [4]]
    assert flat["doc_year"] == [2000, 1990, 1995]


def test_flatten_ce_pairs_concatenates_multiple_rows():
    """Pairs from multiple rows are concatenated in row order."""
    rows = [
        {"query": "qA", "positive": "PA", "negatives": ["NA0"],
         "pos_clap": [1.0], "pos_cf_track": [1.0], "neg_clap": [[0.0]],
         "neg_cf_track": [[0.0]], "user_cf": [0.1],
         "pos_tag_ids": [1], "pos_year": 2001, "neg_tag_ids": [[2]], "neg_years": [1999]},
        {"query": "qB", "positive": "PB", "negatives": ["NB0"],
         "pos_clap": [2.0], "pos_cf_track": [2.0], "neg_clap": [[0.0]],
         "neg_cf_track": [[0.0]], "user_cf": [0.2],
         "pos_tag_ids": [3], "pos_year": 2002, "neg_tag_ids": [[4]], "neg_years": [1998]},
    ]

    flat = flatten_ce_pairs(rows)

    assert flat["doc_text"] == ["PA", "NA0", "PB", "NB0"]
    assert flat["is_positive"] == [True, False, True, False]
    assert flat["query"] == ["qA", "qA", "qB", "qB"]
