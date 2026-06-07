"""Train/serve/feature-build parity for the SASRec channel's dialog text.

SASRec was TRAINED (train_sasrec._walk_split) on user-turns-only dialog built
from `prior` = turns before the music turn PLUS the current-turn user request:

    prior = df[(turn_number < tn) | ((turn_number == tn) & (role == "user"))]
    build_user_dialog(prior.to_dict("records"))

Two paths drifted from that training distribution and must be realigned:

  - Serve (crs_baseline.batch_chat) built user_dialog from data["session_memory"],
    which is the PRE-append history and EXCLUDES the current user request -> the
    weight-1.0 SASRec channel never saw the request it was trained to condition on.
  - The LGBM feature builder (build_lgbm_features.build) passed NO user_dialog at
    all, so sasrec_seq fell back to the raw query (current request + expanded
    music-turn metadata) -> the wrrf_rank / n_channels_hit features the reranker
    is trained on were computed from a context the model never saw.

These tests pin the corrected behavior: both paths include the current user
request and exclude music/assistant turns, matching training.
"""
import inspect

import pandas as pd

from mcrs.crs_baseline import CRS_BASELINE
from mcrs.retrieval_modules.sasrec_model import build_user_dialog, prior_turns


def _bare_crs():
    # Skip __init__ (no model load) -- only the pure helper is exercised.
    return CRS_BASELINE.__new__(CRS_BASELINE)


# ---- serve path: dialog turns include the current user request ----

def test_serve_sasrec_dialog_includes_current_user_query():
    prior = [
        {"role": "user", "content": "play something jazzy"},
        {"role": "assistant", "content": "META:t1", "track_id": "t1"},
    ]
    turns = _bare_crs()._sasrec_dialog_turns(prior, "now something faster")
    # build_user_dialog keeps user turns only, in order -> the current request
    # must be the trailing line (it was missing before the fix).
    assert build_user_dialog(turns) == "play something jazzy\nnow something faster"


def test_serve_sasrec_dialog_excludes_music_and_assistant_turns():
    prior = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "META:t1", "track_id": "t1"},
        {"role": "music", "content": "t2"},
    ]
    turns = _bare_crs()._sasrec_dialog_turns(prior, "more like that")
    assert build_user_dialog(turns) == "hi\nmore like that"


def test_serve_batch_chat_uses_the_dialog_helper():
    # Guard the wiring: batch_chat must build user_dialog via the parity helper
    # (not the old prior_history-only call that dropped the current request).
    src = inspect.getsource(CRS_BASELINE.batch_chat)
    assert "_sasrec_dialog_turns" in src, \
        "batch_chat must build SASRec user_dialog via _sasrec_dialog_turns"


# ---- feature builder: chunk_context must carry user_dialog (matches training) ----

def test_build_lgbm_features_threads_user_dialog():
    from scripts.build_lgbm_features import build
    src = inspect.getsource(build)
    assert "user_dialog" in src, \
        "build() must thread user_dialog into chunk_context for the SASRec channel"
    assert "build_user_dialog" in src, \
        "build() must construct user_dialog via build_user_dialog (user-turns-only)"


# ---- parity-by-construction: one shared prior_turns slice across producers ----

def test_prior_turns_includes_history_and_current_user_excludes_future():
    # The training slice: turns before the target turn (all roles) + the
    # current-turn USER request; future turns and the current music turn dropped.
    df = pd.DataFrame([
        {"turn_number": 1, "role": "user", "content": "a"},
        {"turn_number": 1, "role": "music", "content": "t1"},
        {"turn_number": 2, "role": "user", "content": "b"},
        {"turn_number": 2, "role": "music", "content": "t2"},  # target gold (excluded)
        {"turn_number": 3, "role": "user", "content": "c"},     # future (excluded)
    ])
    p = prior_turns(df, 2)
    assert list(p["content"]) == ["a", "t1", "b"]
    assert build_user_dialog(p.to_dict("records")) == "a\nb"


def test_offline_producers_share_the_prior_turns_helper():
    # Parity is by construction only if BOTH offline producers slice via the same
    # helper rather than re-inlining a (possibly divergent) boolean mask.
    from scripts.build_lgbm_features import build
    from scripts import train_sasrec
    assert "prior_turns(" in inspect.getsource(build), \
        "build_lgbm_features.build must slice prior turns via prior_turns()"
    assert "prior_turns(" in inspect.getsource(train_sasrec._walk_split), \
        "train_sasrec._walk_split must slice prior turns via prior_turns()"
