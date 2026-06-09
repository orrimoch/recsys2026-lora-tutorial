"""Same-artist hard negatives for the two-tower (Tier-1 #3.3).

`build_artist_index` maps tracks -> artist and artist -> track-indices;
`sample_hard_negatives` draws same-artist, different-track negatives for a batch of
golds, excluding the batch golds themselves (they are in-batch positives). Pure
helpers — no model / GPU / data download.
"""
import numpy as np

from scripts.train_two_tower import build_artist_index, sample_hard_negatives


def test_build_artist_index_aligns_and_groups():
    track_ids = ['t0', 't1', 't2', 't3']
    md = {
        't0': {'artist_name': ['Tycho']},
        't1': {'artist_name': ['Tycho']},
        't2': {'artist_name': ['Muse']},
        't3': {'artist_name': []},          # missing artist -> not grouped
    }
    idx_to_artist, artist_to_idx = build_artist_index(track_ids, md)
    assert idx_to_artist[0] == 'tycho' and idx_to_artist[2] == 'muse'
    assert artist_to_idx['tycho'] == [0, 1]
    assert artist_to_idx['muse'] == [2]
    assert '' not in artist_to_idx


def test_sample_hard_negatives_same_artist():
    idx_to_artist = ['tycho', 'tycho', 'tycho', 'muse']
    artist_to_idx = {'tycho': [0, 1, 2], 'muse': [3]}
    rng = np.random.RandomState(0)
    negs = sample_hard_negatives([0], idx_to_artist, artist_to_idx, n_per=5, rng=rng)
    # gold 0 is Tycho -> same-artist others are 1, 2 (not 0 itself, not the Muse track)
    assert set(negs) == {1, 2}


def test_sample_excludes_batch_golds():
    idx_to_artist = ['tycho', 'tycho', 'tycho']
    artist_to_idx = {'tycho': [0, 1, 2]}
    rng = np.random.RandomState(0)
    # golds 0 and 1 are in-batch positives -> only track 2 can be a hard negative.
    negs = sample_hard_negatives([0, 1], idx_to_artist, artist_to_idx, n_per=5, rng=rng)
    assert set(negs) == {2}


def test_no_negatives_for_singleton_artist():
    idx_to_artist = ['solo', 'muse', 'muse']
    artist_to_idx = {'solo': [0], 'muse': [1, 2]}
    rng = np.random.RandomState(0)
    assert sample_hard_negatives([0], idx_to_artist, artist_to_idx, n_per=5, rng=rng) == []


def test_n_per_caps_negatives_drawn():
    idx_to_artist = ['a'] * 6
    artist_to_idx = {'a': [0, 1, 2, 3, 4, 5]}
    rng = np.random.RandomState(0)
    negs = sample_hard_negatives([0], idx_to_artist, artist_to_idx, n_per=2, rng=rng)
    assert len(negs) == 2 and 0 not in negs
