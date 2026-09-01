from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from data_augmentations import gauss_smooth  # noqa: E402
from dataset import BrainToTextDataset, train_test_split_indicies  # noqa: E402
from rnn_model import GRUDecoder  # noqa: E402


REAL_512_SESSION = (
    REPO_ROOT
    / "data"
    / "hdf5_data_512"
    / "t15.2026.08.14.10-11-24_tc_sbp_512"
)


def _write_trial_file(path: Path, n_features: int, *, include_seq_len: bool = False) -> None:
    with h5py.File(path, "w") as h5:
        group = h5.create_group("trial_0000")
        features = np.arange(18 * n_features, dtype=np.float32).reshape(18, n_features)
        group.create_dataset("input_features", data=features)
        group.create_dataset("seq_class_ids", data=np.array([34, 1, 2, 34], dtype=np.int32))
        group.create_dataset("transcription", data=np.frombuffer(b"test\0", dtype=np.uint8))
        group.attrs["n_time_steps"] = np.array([18], dtype=np.int32)
        if include_seq_len:
            group.attrs["seq_len"] = np.array([4], dtype=np.int32)
        group.attrs["block_num"] = np.array([2], dtype=np.int32)
        group.attrs["trial_num"] = np.array([1], dtype=np.int32)


def test_dataset_preserves_512_features_and_uses_label_length_when_seq_len_missing(tmp_path):
    hdf5_path = tmp_path / "data_train.hdf5"
    _write_trial_file(hdf5_path, 512, include_seq_len=False)

    dataset = BrainToTextDataset(
        {0: {"trials": [0], "session_path": str(hdf5_path)}},
        n_batches=1,
        split="train",
        batch_size=1,
        days_per_batch=1,
        random_seed=1,
        expected_feature_dim=512,
    )

    batch = dataset[0]

    assert batch["input_features"].shape == (1, 18, 512)
    assert torch.isfinite(batch["input_features"]).all()
    assert batch["phone_seq_lens"].tolist() == [4]
    assert batch["n_time_steps"].tolist() == [18]
    assert batch["block_nums"].tolist() == [2]
    assert batch["trial_nums"].tolist() == [1]


def test_dataset_rejects_feature_dim_mismatch(tmp_path):
    hdf5_path = tmp_path / "data_train.hdf5"
    _write_trial_file(hdf5_path, 256, include_seq_len=True)

    dataset = BrainToTextDataset(
        {0: {"trials": [0], "session_path": str(hdf5_path)}},
        n_batches=1,
        split="train",
        batch_size=1,
        days_per_batch=1,
        random_seed=1,
        expected_feature_dim=512,
    )

    with pytest.raises(ValueError, match="expected 512"):
        dataset[0]


def test_real_512_hdf5_short_training_step_preserves_sbp_forward_backward():
    train_path = REAL_512_SESSION / "data_train.hdf5"
    if not train_path.exists():
        pytest.skip(f"512D smoke data not found: {train_path}")

    train_trials, _ = train_test_split_indicies(
        [str(train_path)],
        test_percentage=0,
        seed=1,
    )
    dataset = BrainToTextDataset(
        train_trials,
        n_batches=1,
        split="train",
        batch_size=2,
        days_per_batch=1,
        random_seed=1,
        expected_feature_dim=512,
    )
    batch = dataset[0]

    features = batch["input_features"].float()
    assert features.shape[-1] == 512
    assert torch.isfinite(features).all()
    assert features[..., 256:].abs().sum() > 0
    assert not torch.equal(features[..., :256], features[..., 256:])

    smoothed = gauss_smooth(
        features,
        device=torch.device("cpu"),
        smooth_kernel_std=1,
        smooth_kernel_size=9,
    )
    assert smoothed.shape == features.shape

    model = GRUDecoder(
        neural_dim=512,
        n_units=16,
        n_days=1,
        n_classes=35,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=1,
        patch_size=0,
        patch_stride=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    logits = model(smoothed, batch["day_indicies"])
    assert logits.shape[:2] == smoothed.shape[:2]
    assert logits.shape[-1] == 35
    assert torch.isfinite(logits).all()

    loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)(
        logits.log_softmax(2).permute(1, 0, 2),
        batch["seq_class_ids"],
        batch["n_time_steps"],
        batch["phone_seq_lens"],
    )
    assert torch.isfinite(loss)

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    assert torch.isfinite(grad_norm)

    optimizer.step()
    assert all(torch.isfinite(param).all() for param in model.parameters())
