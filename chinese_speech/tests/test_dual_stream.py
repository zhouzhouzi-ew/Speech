import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from chinese_speech.dual_stream_dataset import ChineseDualStreamDataset, collate_dual_stream
from chinese_speech.dual_stream_model import DualStreamGRUDecoder
from chinese_speech.losses import dual_stream_ctc_loss


def _write_trial(handle, name, n_time_steps, syllables, tones):
    group = handle.create_group(name)
    group.create_dataset("input_features", data=np.ones((n_time_steps, 4), dtype=np.float32))
    group.create_dataset("seq_syllable_ids", data=np.asarray(syllables, dtype=np.int32))
    group.create_dataset("seq_tone_ids", data=np.asarray(tones, dtype=np.int32))
    group.create_dataset("seq_class_ids", data=np.asarray(syllables, dtype=np.int32))
    group.create_dataset("transcription", data=np.frombuffer("牛".encode("utf-8") + b"\0", dtype=np.uint8))
    group.attrs["n_time_steps"] = n_time_steps
    group.attrs["seq_len"] = len(syllables)
    group.attrs["tone_seq_len"] = len(tones)
    group.attrs["block_num"] = 1
    group.attrs["trial_num"] = int(name.split("_")[-1])


def test_dual_stream_dataset_collate_pads_both_target_streams(tmp_path):
    path = tmp_path / "data_train.hdf5"
    with h5py.File(path, "w") as handle:
        _write_trial(handle, "trial_0000", 5, [5, 1, 5], [6, 2, 6])
        _write_trial(handle, "trial_0001", 7, [5, 2, 3, 5], [6, 4, 1, 6])

    loader = DataLoader(
        ChineseDualStreamDataset([path], split="train"),
        batch_size=2,
        collate_fn=collate_dual_stream,
    )
    batch = next(iter(loader))

    assert batch["input_features"].shape == (2, 7, 4)
    assert batch["seq_syllable_ids"].shape == (2, 4)
    assert batch["seq_tone_ids"].shape == (2, 4)
    assert batch["syllable_seq_lens"].tolist() == [3, 4]
    assert batch["tone_seq_lens"].tolist() == [3, 4]


def test_dual_stream_model_returns_syllable_and_tone_logits():
    model = DualStreamGRUDecoder(
        neural_dim=4,
        n_units=8,
        n_days=2,
        n_syllable_classes=6,
        n_tone_classes=7,
        n_layers=1,
    )

    out = model(torch.ones(3, 10, 4), day_idx=torch.tensor([0, 1, 0]))

    assert out["syllable_logits"].shape == (3, 10, 6)
    assert out["tone_logits"].shape == (3, 10, 7)


def test_dual_stream_ctc_loss_combines_syllable_and_tone_losses():
    syllable_logits = torch.randn(2, 8, 6, requires_grad=True)
    tone_logits = torch.randn(2, 8, 7, requires_grad=True)
    batch = {
        "seq_syllable_ids": torch.tensor([[5, 1, 5, 0], [5, 2, 3, 5]], dtype=torch.long),
        "seq_tone_ids": torch.tensor([[6, 2, 6, 0], [6, 4, 1, 6]], dtype=torch.long),
        "n_time_steps": torch.tensor([8, 8], dtype=torch.long),
        "syllable_seq_lens": torch.tensor([3, 4], dtype=torch.long),
        "tone_seq_lens": torch.tensor([3, 4], dtype=torch.long),
    }

    loss, parts = dual_stream_ctc_loss(
        {"syllable_logits": syllable_logits, "tone_logits": tone_logits},
        batch,
    )

    assert loss.ndim == 0
    assert parts["syllable_loss"] > 0
    assert parts["tone_loss"] > 0
