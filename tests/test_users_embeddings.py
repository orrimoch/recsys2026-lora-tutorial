"""F1 — Users, TrackEmbeddings, UserEmbeddings."""
from __future__ import annotations

import os

import numpy as np
import pytest

from mcrs.contracts import UserProfile
from mcrs.data.embeddings import TrackEmbeddings, UserEmbeddings
from mcrs.data.users import Users

_TRACK_ROWS = [
    {"track_id": "a", "cf-bpr": [0.1, 0.2], "audio-laion_clap": [1.0, 2.0, 3.0]},
    {"track_id": "b", "cf-bpr": [0.3, 0.4], "audio-laion_clap": [4.0, 5.0, 6.0]},
]
_USER_EMB_ROWS = [{"user_id": "u1", "cf-bpr": [0.5, 0.6]},
                  {"user_id": "u2", "cf-bpr": [0.7, 0.8]}]
_USER_META_ROWS = [{"user_id": "u1", "age": 19, "gender": "female",
                    "country_name": "Bulgaria", "country_code": "BG"}]


# ---- Users ----
def test_users_profile_builds_f2_userprofile_without_history():
    p = Users(_USER_META_ROWS).profile("u1")
    assert isinstance(p, UserProfile)
    assert (p.user_id, p.age, p.gender, p.country) == ("u1", 19, "female", "Bulgaria")
    assert p.history_tids == []  # User-Metadata carries no listening history


def test_users_contains():
    u = Users(_USER_META_ROWS)
    assert "u1" in u and "u404" not in u


# ---- TrackEmbeddings ----
def test_track_modalities_discovered():
    te = TrackEmbeddings(_TRACK_ROWS)
    assert set(te.modalities) == {"cf-bpr", "audio-laion_clap"}


def test_track_vector_returns_values():
    te = TrackEmbeddings(_TRACK_ROWS)
    np.testing.assert_allclose(te.vector("a", "cf-bpr"), [0.1, 0.2])


def test_track_matrix_shape_and_row_alignment():
    te = TrackEmbeddings(_TRACK_ROWS)
    m = te.matrix("audio-laion_clap")
    assert m.shape == (2, 3) and m.dtype == np.float32
    # row order follows index_to_id
    np.testing.assert_allclose(m[te.id_to_index["b"]], [4.0, 5.0, 6.0])


# ---- UserEmbeddings ----
def test_user_embedding_vector_and_missing():
    ue = UserEmbeddings(_USER_EMB_ROWS)
    np.testing.assert_allclose(ue.vector("u1"), [0.5, 0.6])
    assert ue.vector("cold-user") is None        # missing => cold signal
    assert "u1" in ue and "cold-user" not in ue


def test_user_embedding_empty_vector_is_cold_none():
    # cold users can have an EMPTY cf-bpr list -> must read as None, not a size-0 array
    ue = UserEmbeddings([{"user_id": "uc", "cf-bpr": []}])
    assert ue.vector("uc") is None


@pytest.mark.skipif(not os.path.isdir("data/TalkPlayData-Challenge-Track-Embeddings"),
                    reason="embeddings not on disk")
def test_track_embeddings_from_disk_cfbpr_shape():
    te = TrackEmbeddings.from_disk("data/TalkPlayData-Challenge-Track-Embeddings",
                                   split="all_tracks", modalities=["cf-bpr"])
    m = te.matrix("cf-bpr")
    assert m.shape == (47071, 128)
