from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from chinese_speech.dual_stream_dataset import ChineseDualStreamDataset
from chinese_speech.train_dual_stream import (
    adjusted_input_lengths,
    build_global_label_maps,
    train_from_config,
)


def _write_metadata(session_dir: Path, syllable_to_id: dict[str, int]) -> None:
    metadata = {
        "labels": {
            "syllable_to_id": syllable_to_id,
            "tone_to_id": {"<blank>": 0, "1": 1, "2": 2, "<sil>": 3},
        }
    }
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_split(
    path: Path,
    *,
    syllables: list[int],
    tones: list[int],
    n_features: int = 4,
) -> None:
    with h5py.File(path, "w") as h5:
        for trial_idx in range(3):
            group = h5.create_group(f"trial_{trial_idx:04d}")
            features = np.linspace(0.1, 1.0, num=12 * n_features, dtype=np.float32)
            group.create_dataset("input_features", data=features.reshape(12, n_features) + trial_idx)
            group.create_dataset("seq_syllable_ids", data=np.array(syllables, dtype=np.int32))
            group.create_dataset("seq_tone_ids", data=np.array(tones, dtype=np.int32))
            group.create_dataset(
                "transcription",
                data=np.frombuffer("妈\0".encode("utf-8"), dtype=np.uint8),
            )
            group.attrs["n_time_steps"] = 12
            group.attrs["seq_len"] = len(syllables)
            group.attrs["tone_seq_len"] = len(tones)
            group.attrs["block_num"] = 1
            group.attrs["trial_num"] = trial_idx


def _write_session(
    root: Path,
    name: str,
    syllable_to_id: dict[str, int],
    syllables: list[int],
    tones: list[int],
) -> Path:
    session_dir = root / name
    session_dir.mkdir(parents=True)
    _write_metadata(session_dir, syllable_to_id)
    for split in ("train", "val", "test"):
        _write_split(session_dir / f"data_{split}.hdf5", syllables=syllables, tones=tones)
    return session_dir


def test_global_label_maps_remap_conflicting_session_local_ids(tmp_path):
    first = _write_session(
        tmp_path,
        "session_a",
        {"<blank>": 0, "ma": 1, "ba": 2, "<sil>": 3},
        [3, 1, 3],
        [3, 1, 3],
    )
    second = _write_session(
        tmp_path,
        "session_b",
        {"<blank>": 0, "ba": 1, "ma": 2, "<sil>": 3},
        [3, 2, 3],
        [3, 1, 3],
    )

    maps = build_global_label_maps([first, second])
    dataset = ChineseDualStreamDataset(
        [first / "data_train.hdf5", second / "data_train.hdf5"],
        split="train",
        label_remaps=maps.remaps,
        expected_feature_dim=4,
    )

    first_item = dataset[0]
    second_item = dataset[3]
    sil_id = maps.syllable_to_id["<sil>"]
    ma_id = maps.syllable_to_id["ma"]

    assert first_item["seq_syllable_ids"].tolist() == [sil_id, ma_id, sil_id]
    assert second_item["seq_syllable_ids"].tolist() == [sil_id, ma_id, sil_id]


def test_adjusted_input_lengths_handles_patched_and_unpatched_inputs():
    lengths = torch.tensor([20, 21], dtype=torch.long)

    assert adjusted_input_lengths(lengths, patch_size=0, patch_stride=0).tolist() == [20, 21]
    assert adjusted_input_lengths(lengths, patch_size=14, patch_stride=4).tolist() == [2, 2]


def test_train_from_config_runs_independent_tiny_dual_stream_training(tmp_path):
    data_root = tmp_path / "hdf5_chinese"
    _write_session(
        data_root,
        "session_a",
        {"<blank>": 0, "ma": 1, "ba": 2, "<sil>": 3},
        [3, 1, 3],
        [3, 1, 3],
    )
    _write_session(
        data_root,
        "session_b",
        {"<blank>": 0, "ba": 1, "ma": 2, "<sil>": 3},
        [3, 2, 3],
        [3, 1, 3],
    )
    output_dir = tmp_path / "out"

    config = OmegaConf.create(
        {
            "seed": 1,
            "device": "cpu",
            "output_dir": str(output_dir),
            "save_checkpoint": True,
            "data": {
                "hdf5_root": str(data_root),
                "sessions": "all",
                "batch_size": 2,
                "num_workers": 0,
                "expected_feature_dim": 4,
            },
            "model": {
                "n_units": 8,
                "n_layers": 1,
                "rnn_dropout": 0.0,
                "input_dropout": 0.0,
                "patch_size": 0,
                "patch_stride": 0,
            },
            "training": {
                "num_batches": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "tone_weight": 0.5,
                "grad_norm_clip": 5.0,
                "smooth_data": True,
                "smooth_kernel_std": 1.0,
                "smooth_kernel_size": 9,
                "log_every": 1,
                "val_every": 1,
            },
        }
    )

    result = train_from_config(config)

    assert result["n_days"] == 2
    assert result["n_syllable_classes"] == 4
    assert result["n_tone_classes"] == 4
    assert len(result["train_losses"]) == 2
    assert all(np.isfinite(result["train_losses"]))
    assert np.isfinite(result["best_val_loss"])
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "checkpoints" / "latest.pt").exists()
