"""PyTorch Dataset wrapping the W2 training parquet for W3 generator fine-tune."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class SIDTrainingDataset(Dataset):
    """Lightweight Dataset over a SID training parquet.

    __getitem__ returns (query_string, target_codes_tensor[3, dtype=long]).
    The W3 trainer is responsible for tokenizing the query string and
    constructing the loss against the 3 target codes (which map to the
    expanded vocabulary of SID tokens).
    """

    def __init__(
        self,
        parquet_path,
        filter_source: Optional[str] = None,
    ) -> None:
        df = pd.read_parquet(parquet_path)
        if filter_source is not None:
            df = df[df["source"] == filter_source].reset_index(drop=True)
        self.queries: list[str] = df["query"].tolist()
        codes = df[["code_1", "code_2", "code_3"]].to_numpy(dtype="int64")
        self.codes = torch.from_numpy(codes)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.queries[idx], self.codes[idx]
