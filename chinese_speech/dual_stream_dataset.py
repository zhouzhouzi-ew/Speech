from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _decode_transcription(value: np.ndarray) -> str:
    raw = bytes(int(x) for x in np.asarray(value).reshape(-1) if int(x) != 0)
    return raw.decode("utf-8")


def _scalar_int(value: object) -> int:
    arr = np.asarray(value)
    if arr.shape == ():
        return int(arr.item())
    if arr.size == 1:
        return int(arr.reshape(-1)[0].item())
    raise ValueError(f"Expected scalar HDF5 attr, got shape {arr.shape}")


def _remap_ids(ids: torch.Tensor, remap: Optional[Sequence[int]]) -> torch.Tensor:
    if remap is None:
        return ids.long()
    lookup = torch.tensor(remap, dtype=torch.long)
    if ids.numel() and int(ids.max()) >= len(lookup):
        raise ValueError(f"Label id {int(ids.max())} exceeds remap length {len(lookup)}")
    return lookup[ids.long()]


class ChineseDualStreamDataset(Dataset):
    def __init__(
        self,
        hdf5_paths: Sequence[Path],
        split: str,
        *,
        label_remaps: Optional[Mapping[str, Mapping[str, Sequence[int]]]] = None,
        expected_feature_dim: Optional[int] = None,
    ) -> None:
        self.split = split
        self.label_remaps = label_remaps or {}
        self.expected_feature_dim = expected_feature_dim
        self.records: List[Dict[str, object]] = []
        for day_idx, path in enumerate(hdf5_paths):
            path = Path(path)
            with h5py.File(path, "r") as handle:
                for key in sorted(handle.keys()):
                    self.records.append({"path": path, "key": key, "day_idx": day_idx})

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        with h5py.File(record["path"], "r") as handle:
            group = handle[record["key"]]
            path = Path(record["path"])
            session_name = path.parent.name
            remaps = self.label_remaps.get(session_name, {})
            features = torch.from_numpy(group["input_features"][:]).float()
            if self.expected_feature_dim is not None and features.shape[-1] != self.expected_feature_dim:
                raise ValueError(
                    f"Loaded {features.shape[-1]} neural features from {path} {record['key']}; "
                    f"expected {self.expected_feature_dim}."
                )
            syllable_ids = torch.from_numpy(group["seq_syllable_ids"][:]).long()
            tone_ids = torch.from_numpy(group["seq_tone_ids"][:]).long()
            return {
                "input_features": features,
                "seq_syllable_ids": _remap_ids(syllable_ids, remaps.get("syllable")),
                "seq_tone_ids": _remap_ids(tone_ids, remaps.get("tone")),
                "n_time_steps": _scalar_int(group.attrs["n_time_steps"]),
                "syllable_seq_len": _scalar_int(group.attrs["seq_len"]),
                "tone_seq_len": _scalar_int(group.attrs["tone_seq_len"]),
                "day_idx": int(record["day_idx"]),
                "block_num": _scalar_int(group.attrs["block_num"]),
                "trial_num": _scalar_int(group.attrs["trial_num"]),
                "transcription": _decode_transcription(group["transcription"][:]),
            }


def collate_dual_stream(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "input_features": pad_sequence(
            [record["input_features"] for record in records],
            batch_first=True,
            padding_value=0,
        ),
        "seq_syllable_ids": pad_sequence(
            [record["seq_syllable_ids"] for record in records],
            batch_first=True,
            padding_value=0,
        ),
        "seq_tone_ids": pad_sequence(
            [record["seq_tone_ids"] for record in records],
            batch_first=True,
            padding_value=0,
        ),
        "n_time_steps": torch.tensor([record["n_time_steps"] for record in records], dtype=torch.long),
        "syllable_seq_lens": torch.tensor(
            [record["syllable_seq_len"] for record in records],
            dtype=torch.long,
        ),
        "tone_seq_lens": torch.tensor(
            [record["tone_seq_len"] for record in records],
            dtype=torch.long,
        ),
        "day_indicies": torch.tensor([record["day_idx"] for record in records], dtype=torch.long),
        "block_nums": torch.tensor([record["block_num"] for record in records], dtype=torch.long),
        "trial_nums": torch.tensor([record["trial_num"] for record in records], dtype=torch.long),
        "transcriptions": [record["transcription"] for record in records],
    }
