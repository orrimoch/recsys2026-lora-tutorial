"""Tests for the SID_GENERATOR retrieval class (W4)."""
import pytest


@pytest.fixture(scope="module")
def tiny_sid_lookup_parquet(tmp_path_factory):
    """A minimal 5-row track_to_sid.parquet for fast tests (matches W1 schema)."""
    import pandas as pd
    rows = [
        {"track_id": "t1", "code_1": 0, "code_2": 0, "code_3": 0, "popularity": 1.0, "bucket_rank": 0},
        {"track_id": "t2", "code_1": 0, "code_2": 0, "code_3": 1, "popularity": 0.9, "bucket_rank": 0},
        {"track_id": "t3", "code_1": 0, "code_2": 1, "code_3": 0, "popularity": 0.8, "bucket_rank": 0},
        # Collision: t4 and t5 share the same SID triplet (0, 1, 0)
        {"track_id": "t4", "code_1": 0, "code_2": 1, "code_3": 0, "popularity": 0.7, "bucket_rank": 1},
        {"track_id": "t5", "code_1": 1, "code_2": 0, "code_3": 0, "popularity": 0.6, "bucket_rank": 0},
    ]
    p = tmp_path_factory.mktemp("sid") / "track_to_sid.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return p


def test_sid_generator_init_loads_model_tokenizer_and_trie(monkeypatch, tiny_sid_lookup_parquet, tmp_path):
    """Init: load model + tokenizer from a (mocked) hub_repo, build trie + collision lookup."""
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR

    # Stub out heavy HF loads so the test doesn't need GPU or network.
    class _StubTokenizer:
        def __init__(self): self.pad_token_id = 0; self.eos_token_id = 1
        def get_vocab(self):
            # 768 SID tokens at ids 1000..1767 (deterministic for test)
            v = {f"<SID_L{lvl}_C{code}>": 1000 + lvl * 256 + code
                 for lvl in range(3) for code in range(256)}
            return v
        def encode(self, s, add_special_tokens=False):
            # Used by build_sid_to_token_id_lookup
            if s.startswith("<SID_L"):
                # Parse the level and code from the string
                import re
                m = re.match(r"<SID_L(\d+)_C(\d+)>", s)
                if m:
                    lvl, code = int(m.group(1)), int(m.group(2))
                    return [1000 + lvl * 256 + code]
            return [99]
        @classmethod
        def from_pretrained(cls, _): return cls()

    class _StubModel:
        device = "cpu"
        def __init__(self): self.generation_config = type("c", (), {})()
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def generate(self, *a, **k):
            raise NotImplementedError

    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoModelForCausalLM", _StubModel)

    gen = SID_GENERATOR(
        hub_repo="fake/repo",
        sid_lookup_path=tiny_sid_lookup_parquet,
        device="cpu",
        num_beams=20,
    )

    # Trie has 4 unique SID triplets ((0,0,0), (0,0,1), (0,1,0), (1,0,0))
    # because (0,1,0) appears twice via collision.
    assert len(gen.sid_to_tracks) == 4
    # Collision bucket (0,1,0) has 2 tracks, popularity-sorted
    assert gen.sid_to_tracks[(0, 1, 0)] == ["t3", "t4"]


def test_batch_text_to_item_retrieval_returns_topk_per_query(monkeypatch, tiny_sid_lookup_parquet):
    """Method emits exactly topk tracks per query, applying per-bucket cap."""
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
    import torch

    class _StubTokenizer:
        def __init__(self): self.pad_token_id = 0; self.eos_token_id = 1
        def get_vocab(self):
            return {f"<SID_L{lvl}_C{code}>": 1000 + lvl * 256 + code
                    for lvl in range(3) for code in range(256)}
        def encode(self, s, add_special_tokens=False):
            if s.startswith("<SID_L"):
                import re
                m = re.match(r"<SID_L(\d+)_C(\d+)>", s)
                if m:
                    lvl, code = int(m.group(1)), int(m.group(2))
                    return [1000 + lvl * 256 + code]
            return [99]
        @classmethod
        def from_pretrained(cls, _): return cls()
        def __call__(self, text, **k):
            return {"input_ids": torch.tensor([[10, 20, 30]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])}

    class _StubModel:
        device = "cpu"
        def __init__(self): self.generation_config = type("c", (), {})()
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def generate(self, input_ids, attention_mask, max_new_tokens, num_beams,
                     num_return_sequences, prefix_allowed_tokens_fn, **k):
            # Return 5 beams each with prompt_ids + 3 SID tokens.
            beam_sids = [
                (0, 0, 0),  # → t1
                (0, 0, 1),  # → t2
                (0, 1, 0),  # → t3 (primary), t4 (spillover)
                (1, 0, 0),  # → t5
                (0, 0, 0),  # duplicate of beam 0
            ]
            sequences = []
            prompt = input_ids[0].tolist()
            for (c1, c2, c3) in beam_sids:
                seq = prompt + [1000 + c1, 1000 + 256 + c2, 1000 + 512 + c3]
                sequences.append(seq)
            return type("o", (), {"sequences": torch.tensor(sequences)})()

    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoTokenizer", _StubTokenizer)
    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoModelForCausalLM", _StubModel)

    gen = SID_GENERATOR(
        hub_repo="fake/repo",
        sid_lookup_path=tiny_sid_lookup_parquet,
        device="cpu",
        num_beams=5,
    )

    out = gen.batch_text_to_item_retrieval(queries=["play me something"], topk=5)
    assert len(out) == 1
    tracks = out[0]
    # First 4 beams resolve to distinct primaries: t1, t2, t3, t5 (one per bucket).
    # 5th beam is duplicate (0,0,0) → already seen. Spillover from (0,1,0) bucket adds t4.
    assert tracks[:4] == ["t1", "t2", "t3", "t5"]
    assert "t4" in tracks  # spillover
    assert len(tracks) == 5


def test_batch_text_to_item_retrieval_processes_multiple_queries(monkeypatch, tiny_sid_lookup_parquet):
    """Method handles N queries → returns list of N inner lists."""
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
    import torch

    class _Tok:
        pad_token_id = 0; eos_token_id = 1
        def get_vocab(self):
            return {f"<SID_L{lvl}_C{code}>": 1000 + lvl * 256 + code
                    for lvl in range(3) for code in range(256)}
        def encode(self, s, add_special_tokens=False):
            if s.startswith("<SID_L"):
                import re
                m = re.match(r"<SID_L(\d+)_C(\d+)>", s)
                if m:
                    lvl, code = int(m.group(1)), int(m.group(2))
                    return [1000 + lvl * 256 + code]
            return [99]
        @classmethod
        def from_pretrained(cls, _): return cls()
        def __call__(self, text, **k):
            return {"input_ids": torch.tensor([[10, 20, 30]]),
                    "attention_mask": torch.tensor([[1, 1, 1]])}

    class _Model:
        device = "cpu"
        def __init__(self): self.generation_config = type("c", (), {})()
        def eval(self): return self
        @classmethod
        def from_pretrained(cls, *a, **k): return cls()
        def generate(self, input_ids, **k):
            seqs = []
            for _ in range(k["num_return_sequences"]):
                seqs.append(input_ids[0].tolist() + [1000, 1256, 1512])  # (0,0,0)
            return type("o", (), {"sequences": torch.tensor(seqs)})()

    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoTokenizer", _Tok)
    monkeypatch.setattr("mcrs.retrieval_modules.sid_generator.AutoModelForCausalLM", _Model)

    gen = SID_GENERATOR(
        hub_repo="fake/repo",
        sid_lookup_path=tiny_sid_lookup_parquet,
        device="cpu",
        num_beams=3,
    )

    out = gen.batch_text_to_item_retrieval(queries=["q1", "q2", "q3"], topk=3)
    assert len(out) == 3
    for tracks in out:
        assert len(tracks) <= 3


def test_sid_generator_interface_matches_existing_retrievers(monkeypatch, tiny_sid_lookup_parquet):
    """SID_GENERATOR.batch_text_to_item_retrieval signature must match
    BM25_MODEL's (queries: list[str], topk: int, user_ids=None) -> list[list[str]]."""
    import inspect
    from mcrs.retrieval_modules.sid_generator import SID_GENERATOR
    from mcrs.retrieval_modules.bm25 import BM25_MODEL

    sid_sig = inspect.signature(SID_GENERATOR.batch_text_to_item_retrieval)
    bm25_sig = inspect.signature(BM25_MODEL.batch_text_to_item_retrieval)

    sid_params = list(sid_sig.parameters.keys())
    bm25_params = list(bm25_sig.parameters.keys())
    # First 3 params must match: self, queries, topk; user_ids 4th
    assert sid_params[:3] == bm25_params[:3] == ["self", "queries", "topk"]
    assert "user_ids" in sid_params
    assert "user_ids" in bm25_params
