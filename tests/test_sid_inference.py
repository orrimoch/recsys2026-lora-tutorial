"""Tests for SID trie + prefix_allowed_tokens_fn (W3 subset of W4 inference)."""
import pytest


def test_build_sid_trie_two_branches():
    """Two SIDs sharing a level-0 code share a node at level 0."""
    from mcrs.sid.inference import build_sid_trie
    sids = [(1, 2, 3), (1, 2, 4), (5, 6, 7)]
    sid_lookup = {
        (0, 1): 100, (0, 5): 105,
        (1, 2): 200, (1, 6): 206,
        (2, 3): 300, (2, 4): 304, (2, 7): 307,
    }
    trie = build_sid_trie(sids, sid_lookup)
    # Root has two valid level-0 children: 100 and 105.
    assert sorted(trie.valid_next_token_ids(prefix=[])) == [100, 105]
    # After 100, only level-1 child 200 is valid.
    assert trie.valid_next_token_ids(prefix=[100]) == [200]
    # After [100, 200], two valid level-2 leaves: 300 and 304.
    assert sorted(trie.valid_next_token_ids(prefix=[100, 200])) == [300, 304]


def test_build_sid_trie_empty_after_complete_sid():
    """No more valid tokens after the 3 SID tokens are emitted."""
    from mcrs.sid.inference import build_sid_trie
    sids = [(1, 2, 3)]
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    assert trie.valid_next_token_ids(prefix=[100, 200, 300]) == []


def test_make_prefix_allowed_tokens_fn_strips_prompt(monkeypatch):
    """Function strips the input prompt prefix and walks the trie on the rest."""
    from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
    sids = [(1, 2, 3)]
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    # Simulate prompt of length 10 (any tokens), then 0 SID tokens emitted.
    prompt_lens = {0: 10}  # batch_id 0 → prompt is 10 tokens
    fn = make_prefix_allowed_tokens_fn(trie, prompt_lens)
    # input_ids has just the 10 prompt tokens (e.g., padded with 0s).
    input_ids = [0] * 10
    assert fn(0, input_ids) == [100]
    # After model emits 100, valid next is 200.
    assert fn(0, input_ids + [100]) == [200]


def test_prefix_allowed_tokens_fn_returns_pad_token_after_complete_sid():
    """When all 3 SID tokens have been emitted, return a fallback (eos or pad) so generate() halts cleanly."""
    from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
    sids = [(1, 2, 3)]
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    prompt_lens = {0: 5}
    fn = make_prefix_allowed_tokens_fn(trie, prompt_lens, eos_token_id=2)
    # 5 prompt tokens + 3 emitted SID tokens = 8 total. Next position = exhausted.
    input_ids = [0] * 5 + [100, 200, 300]
    nxt = fn(0, input_ids)
    assert nxt == [2]  # eos


def test_build_sid_trie_handles_collisions():
    """Two distinct tracks sharing the same SID triplet → trie has one path, both tracks recoverable downstream."""
    from mcrs.sid.inference import build_sid_trie
    sids = [(1, 2, 3), (1, 2, 3)]  # two tracks same SID
    lookup = {(0, 1): 100, (1, 2): 200, (2, 3): 300}
    trie = build_sid_trie(sids, lookup)
    # Trie collapses duplicates — only one path.
    assert trie.valid_next_token_ids(prefix=[100, 200]) == [300]
