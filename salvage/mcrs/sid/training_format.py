"""Format (query, sid) pairs into (input_ids, attention_mask, labels) for Trainer.

Labels are masked to -100 everywhere except the 3 SID positions, so the model
only learns to predict SIDs (not to regurgitate the prompt). This is the
standard 'completion-only' training pattern for instruction-tuning, applied
to a fixed-length 3-token completion.
"""
from __future__ import annotations

from typing import Any

from .vocab import encode_sid_to_token_ids


def format_example_for_training(
    query: str,
    code_1: int,
    code_2: int,
    code_3: int,
    *,
    tokenizer,
    sid_lookup: dict[tuple[int, int], int],
    max_prompt_len: int,
) -> dict[str, list[int]]:
    """Build a single training example.

    Layout: [front-truncated prompt token ids ...] [SID_L0_Cx] [SID_L1_Cy] [SID_L2_Cz]
    Labels: [-100, -100, ..., -100, sid_l0_id, sid_l1_id, sid_l2_id]

    Args:
        query: prompt string from W2 (already includes [USER]/[GOAL]/[HISTORY]/[QUERY] blocks)
        code_1, code_2, code_3: gold SID codes
        tokenizer: extended tokenizer (must have SID tokens already added)
        sid_lookup: from build_sid_to_token_id_lookup
        max_prompt_len: total input length excluding the 3 SID tokens; long prompts
            are truncated from the FRONT to preserve the most-recent context

    Returns:
        Dict with input_ids, attention_mask, labels (all list[int], same length).
    """
    sid_ids = encode_sid_to_token_ids(code_1, code_2, code_3, lookup=sid_lookup)

    # Tokenize without truncation first so we know the true length.
    prompt_ids = tokenizer.encode(query, add_special_tokens=False)
    if len(prompt_ids) > max_prompt_len:
        # Front-truncate: keep the LAST max_prompt_len tokens.
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + sid_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + sid_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_training_batch(
    examples: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, "torch.Tensor"]:
    """Left-pad a batch of formatted training examples.

    Padding side = LEFT so the rightmost tokens (which include the 3 SID positions)
    always align across the batch. attention_mask uses 0 for padded positions;
    labels use -100 for padded positions (so cross-entropy ignores them).
    """
    import torch

    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids = []
    attention_mask = []
    labels = []
    for ex in examples:
        pad_n = max_len - len(ex["input_ids"])
        input_ids.append([pad_token_id] * pad_n + ex["input_ids"])
        attention_mask.append([0] * pad_n + ex["attention_mask"])
        labels.append([-100] * pad_n + ex["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
