"""Dialog-conditioned, content-fused SASRec: pure sequence builder + the
PyTorch model + the training loss. No data I/O (see scripts/train_sasrec.py)."""
from __future__ import annotations


def build_session_examples(track_seqs: list[list[int]], max_len: int) -> list[tuple[list[int], int]]:
    """For each session's ordered track-index list, yield (prefix, target) for
    every position t (0-based): prefix = the up-to-max_len tracks before t,
    target = track at t. Position 0 has an empty prefix (predict the first
    track from the dialog alone)."""
    examples: list[tuple[list[int], int]] = []
    for seq in track_seqs:
        for t in range(len(seq)):
            prefix = seq[max(0, t - max_len):t]
            examples.append((prefix, seq[t]))
    return examples
