"""Tests for SIDTrainingDataset (PyTorch Dataset wrapping the train parquet)."""
import pandas as pd
import pytest


@pytest.fixture
def tiny_train_parquet(tmp_path):
    """A 4-row parquet with the exact schema build_sid_training_data.py emits."""
    df = pd.DataFrame([
        {"source": "raw", "track_id": "t1", "query": "play rock",
         "code_1": 5, "code_2": 10, "code_3": 200},
        {"source": "metadata", "track_id": "t2", "query": "track_name: Yesterday | artist: Beatles",
         "code_1": 7, "code_2": 14, "code_3": 99},
        {"source": "doc2query", "track_id": "t3", "query": "dreamy ambient track",
         "code_1": 3, "code_2": 6, "code_3": 12},
        {"source": "raw", "track_id": "t4", "query": "next",
         "code_1": 0, "code_2": 0, "code_3": 1},
    ])
    path = tmp_path / "train.parquet"
    df.to_parquet(path, index=False)
    return path


def test_sid_training_dataset_length_matches_parquet_row_count(tiny_train_parquet):
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet)
    assert len(ds) == 4


def test_sid_training_dataset_returns_query_string_and_target_codes(tiny_train_parquet):
    """__getitem__ returns (query_str, [code_1, code_2, code_3]) tuple."""
    import torch
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet)
    query, codes = ds[0]
    assert query == "play rock"
    assert isinstance(codes, torch.Tensor)
    assert codes.dtype == torch.long
    assert codes.tolist() == [5, 10, 200]


def test_sid_training_dataset_indexing_handles_all_rows(tiny_train_parquet):
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet)
    for i in range(len(ds)):
        q, c = ds[i]
        assert isinstance(q, str)
        assert c.shape == (3,)


def test_sid_training_dataset_filter_by_source_keeps_only_specified(tiny_train_parquet):
    """Optional filter_source arg keeps only rows from that source."""
    from mcrs.sid.generator_dataloader import SIDTrainingDataset
    ds = SIDTrainingDataset(tiny_train_parquet, filter_source="raw")
    assert len(ds) == 2  # 2 raw rows in the fixture
    queries = [ds[i][0] for i in range(len(ds))]
    assert "play rock" in queries
    assert "next" in queries
