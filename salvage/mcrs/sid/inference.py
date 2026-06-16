"""SID generator inference helpers — W3 ships the trie + prefix function for eval;
W4 will add the SID_GENERATOR retrieval class wrapping these.

The trie enforces that beam search only emits valid 3-token SID sequences (i.e.
SIDs that correspond to a real catalog track). HF's `generate(prefix_allowed_tokens_fn=...)`
calls our callable at every step to get the allowed next token ids.
"""
from __future__ import annotations

from typing import Iterable, Optional


class _TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children: dict[int, "_TrieNode"] = {}


class SIDTrie:
    """3-level trie keyed on SID token ids."""

    def __init__(self):
        self.root = _TrieNode()

    def insert(self, token_ids: list[int]) -> None:
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = _TrieNode()
            node = node.children[tid]

    def valid_next_token_ids(self, prefix: list[int]) -> list[int]:
        """Walk the trie along `prefix`; return children of the resulting node.

        Returns [] if prefix walks off the trie (shouldn't happen if generation
        was constrained by an earlier call) OR if the prefix is a complete SID
        (i.e. node has no children).
        """
        node = self.root
        for tid in prefix:
            if tid not in node.children:
                return []
            node = node.children[tid]
        return sorted(node.children.keys())


def build_sid_trie(
    sids: Iterable[tuple[int, int, int]],
    sid_lookup: dict[tuple[int, int], int],
) -> SIDTrie:
    """Build a 3-level trie from an iterable of (c1, c2, c3) SID triplets.

    `sid_lookup` maps (level, code) → token_id (from build_sid_to_token_id_lookup).
    """
    trie = SIDTrie()
    for (c1, c2, c3) in sids:
        token_ids = [
            sid_lookup[(0, c1)],
            sid_lookup[(1, c2)],
            sid_lookup[(2, c3)],
        ]
        trie.insert(token_ids)
    return trie


def make_prefix_allowed_tokens_fn(
    trie: SIDTrie,
    prompt_lens: dict[int, int],
    *,
    eos_token_id: Optional[int] = None,
):
    """Return a callable suitable for HF generate(prefix_allowed_tokens_fn=...).

    Args:
        trie: SIDTrie of valid SID sequences.
        prompt_lens: {batch_id: prompt_length}. Used to strip the prompt prefix
            from input_ids so we walk the trie only on the SID tokens emitted so far.
        eos_token_id: Token id to return after 3 SID tokens are emitted, so generate()
            halts cleanly. If None, returns [] (will trip a HF assertion — pass an
            EOS in production).

    Returns:
        prefix_allowed_tokens_fn(batch_id: int, input_ids: list[int]) -> list[int]
    """

    def _fn(batch_id: int, input_ids) -> list[int]:
        # input_ids may be a torch tensor in HF's call; convert if so.
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        prompt_len = prompt_lens.get(batch_id, 0)
        emitted = list(input_ids[prompt_len:])
        nxt = trie.valid_next_token_ids(prefix=emitted)
        if not nxt:
            if eos_token_id is not None:
                return [eos_token_id]
            return []
        return nxt

    return _fn
