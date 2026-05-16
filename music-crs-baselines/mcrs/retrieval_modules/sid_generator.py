"""W4: SID-generator retrieval class.

Implements `batch_text_to_item_retrieval(queries, topk) -> list[list[str]]`,
matching the existing retrievers (BM25_MODEL, DENSE_PRECOMPUTED, etc.) so it
slots into RRF_MODEL as a 4th sub-stream transparently.

Loads the W3-trained merged Qwen model from HF Hub, builds the SID trie from
W1's track_to_sid.parquet, and per query runs trie-constrained beam search
(prefix_allowed_tokens_fn) to emit exactly 3 SID tokens. The 3 tokens are
decoded back to a SID triplet, looked up in the collision table, and one track
per beam is returned with optional spillover to fill top-K under the per-bucket
cap (per spec §2.5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcrs.sid.inference import build_sid_trie, make_prefix_allowed_tokens_fn
from mcrs.sid.vocab import build_sid_to_token_id_lookup


class SID_GENERATOR:
    """SID-generator retriever wrapping a W3 merged model + W1 trie + collision lookup."""

    def __init__(
        self,
        hub_repo: str,
        sid_lookup_path: Path,
        *,
        device: str = "cuda",
        num_beams: int = 20,
        max_prompt_len: int = 1024,
        cap_per_bucket: int = 1,
    ):
        self.hub_repo = hub_repo
        self.num_beams = num_beams
        self.max_prompt_len = max_prompt_len
        self.cap_per_bucket = cap_per_bucket

        # Load model + tokenizer from Hub.
        self.tokenizer = AutoTokenizer.from_pretrained(hub_repo)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            hub_repo, torch_dtype=dtype, device_map=device if device != "cpu" else None,
        )
        self.model.eval()
        # Pin generation config so beam search doesn't trip on missing pad/eos.
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        # Build SID lookups from W1 parquet (sorted by bucket_rank so popularity
        # order is preserved within collision buckets).
        self.sid_lookup = build_sid_to_token_id_lookup(self.tokenizer, num_levels=3, codebook_size=256)
        self.inverse_lookup = {v: k for k, v in self.sid_lookup.items()}

        t2s = pd.read_parquet(sid_lookup_path).sort_values("bucket_rank")
        self.sid_to_tracks: dict[tuple[int, int, int], list[str]] = {}
        for row in t2s.itertuples(index=False):
            key = (int(row.code_1), int(row.code_2), int(row.code_3))
            self.sid_to_tracks.setdefault(key, []).append(row.track_id)

        # Build trie of valid SID sequences.
        sids = list(t2s[["code_1", "code_2", "code_3"]].itertuples(index=False, name=None))
        self.trie = build_sid_trie(sids, self.sid_lookup)

    def batch_text_to_item_retrieval(
        self,
        queries: list[str],
        topk: int,
        user_ids: Optional[list[str]] = None,
    ) -> list[list[str]]:
        """For each query, run constrained beam search → top-K track IDs.

        Interface parity: matches BM25_MODEL.batch_text_to_item_retrieval (user_ids
        accepted but ignored — SID retrieval is text-conditioned only).
        """
        results: list[list[str]] = []
        with torch.inference_mode():
            for query in queries:
                inputs = self.tokenizer(
                    query, truncation=True, max_length=self.max_prompt_len,
                    return_tensors="pt", add_special_tokens=False,
                )
                # HF BatchEncoding supports .to(device) on CPU too (no-op when already there),
                # so the conditional guard is unnecessary in production. The test fixtures
                # stub the tokenizer to return a plain dict; guard against that case.
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.model.device)
                prompt_len = inputs["input_ids"].shape[1]

                # Build prefix-allowed-tokens fn populated for every beam id.
                prefix_fn = make_prefix_allowed_tokens_fn(
                    self.trie,
                    prompt_lens={i: prompt_len for i in range(self.num_beams)},
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                out = self.model.generate(
                    **inputs,
                    max_new_tokens=3,
                    num_beams=self.num_beams,
                    num_return_sequences=self.num_beams,
                    prefix_allowed_tokens_fn=prefix_fn,
                    do_sample=False,
                )
                # Slice off the prompt → 3 SID tokens per beam.
                beams = out.sequences[:, prompt_len:prompt_len + 3].cpu().tolist()
                tracks = self._decode_beams_to_tracks(beams, topk=topk)
                results.append(tracks)
        return results

    def _decode_beams_to_tracks(
        self, beam_token_ids: list[list[int]], topk: int,
    ) -> list[str]:
        """Convert each beam's 3-token output → SID triplet → tracks.

        Applies per-bucket cap: each beam contributes at most `cap_per_bucket`
        tracks from its collision bucket; remainder of the bucket goes to
        spillover so it can fill any dedup gap (per spec §2.5)."""
        seen: set[str] = set()
        primary: list[str] = []
        spillover: list[str] = []
        n_decode_failures = 0
        for tok_ids in beam_token_ids:
            try:
                sid = (
                    self.inverse_lookup[tok_ids[0]][1],
                    self.inverse_lookup[tok_ids[1]][1],
                    self.inverse_lookup[tok_ids[2]][1],
                )
            except (KeyError, IndexError):
                n_decode_failures += 1
                continue
            bucket = self.sid_to_tracks.get(sid, [])
            added_in_beam = 0
            for tid in bucket:
                if tid in seen:
                    continue
                if added_in_beam < self.cap_per_bucket:
                    primary.append(tid)
                    added_in_beam += 1
                else:
                    spillover.append(tid)
                seen.add(tid)
        if n_decode_failures > 0 and n_decode_failures > len(beam_token_ids) // 2:
            import warnings
            warnings.warn(
                f"SID decode failed for {n_decode_failures}/{len(beam_token_ids)} beams; "
                f"possible tokenizer/vocab mismatch between training and inference."
            )
        return (primary + spillover)[:topk]

    def text_to_item_retrieval(
        self, query: str, topk: int, user_id: Optional[str] = None,
    ) -> list[str]:
        """Single-query convenience wrapper for interface parity with other retrievers."""
        return self.batch_text_to_item_retrieval([query], topk=topk)[0]
