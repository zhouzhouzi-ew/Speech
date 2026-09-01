from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

from .dual_stream_dataset import ChineseDualStreamDataset, collate_dual_stream
from .dual_stream_model import DualStreamGRUDecoder
from .losses import dual_stream_ctc_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLANK_TOKEN = "<blank>"
SIL_TOKEN = "<sil>"
MISSING_SYLLABLE_ID = -1
MISSING_TONE_ID = -1


@dataclass(frozen=True)
class LabelMaps:
    syllable_to_id: Dict[str, int]
    tone_to_id: Dict[str, int]
    remaps: Dict[str, Dict[str, List[int]]]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _read_metadata(session_dir: Path) -> Mapping[str, object]:
    metadata_path = session_dir / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _labels_from_metadata(session_dir: Path) -> tuple[Dict[str, int], Dict[str, int]]:
    metadata = _read_metadata(session_dir)
    labels = metadata["labels"]
    return dict(labels["syllable_to_id"]), dict(labels["tone_to_id"])


def _global_id_map(local_maps: Iterable[Mapping[str, int]], *, tones: bool = False) -> Dict[str, int]:
    labels: set[str] = set()
    for local in local_maps:
        labels.update(str(label) for label in local)

    ordered: List[str] = [BLANK_TOKEN]
    body = labels - {BLANK_TOKEN, SIL_TOKEN}
    if tones:
        ordered.extend(sorted(body, key=lambda label: (0, int(label)) if label.isdigit() else (1, label)))
    else:
        ordered.extend(sorted(body))
    if SIL_TOKEN in labels:
        ordered.append(SIL_TOKEN)
    return {label: idx for idx, label in enumerate(ordered)}


def _local_to_global_remap(local: Mapping[str, int], global_map: Mapping[str, int]) -> List[int]:
    max_local_id = max(int(idx) for idx in local.values())
    remap = [0] * (max_local_id + 1)
    for label, local_id in local.items():
        remap[int(local_id)] = int(global_map[str(label)])
    return remap


def build_global_label_maps(session_dirs: Sequence[Path]) -> LabelMaps:
    local_syllable_maps: Dict[str, Dict[str, int]] = {}
    local_tone_maps: Dict[str, Dict[str, int]] = {}
    for session_dir in session_dirs:
        syllable_map, tone_map = _labels_from_metadata(Path(session_dir))
        local_syllable_maps[Path(session_dir).name] = syllable_map
        local_tone_maps[Path(session_dir).name] = tone_map

    syllable_to_id = _global_id_map(local_syllable_maps.values())
    tone_to_id = _global_id_map(local_tone_maps.values(), tones=True)
    remaps = {
        session: {
            "syllable": _local_to_global_remap(local_syllable_maps[session], syllable_to_id),
            "tone": _local_to_global_remap(local_tone_maps[session], tone_to_id),
        }
        for session in local_syllable_maps
    }
    return LabelMaps(syllable_to_id=syllable_to_id, tone_to_id=tone_to_id, remaps=remaps)


def discover_session_dirs(hdf5_root: Path, sessions: object = "all") -> List[Path]:
    root = Path(hdf5_root)
    if sessions == "all":
        found = [
            path
            for path in sorted(root.iterdir())
            if path.is_dir()
            and (path / "metadata.json").exists()
            and (path / "data_train.hdf5").exists()
            and (path / "data_val.hdf5").exists()
            and (path / "data_test.hdf5").exists()
        ]
    else:
        if isinstance(sessions, str):
            names = [sessions]
        else:
            names = [str(item) for item in sessions]
        found = [root / name for name in names]

    if not found:
        raise FileNotFoundError(f"No Chinese HDF5 sessions found under {root}")
    missing = [str(path) for path in found if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Configured Chinese HDF5 session directories are missing: {missing}")
    return found


def infer_feature_dim(hdf5_path: Path) -> int:
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        first_key = sorted(handle.keys())[0]
        return int(handle[first_key]["input_features"].shape[-1])


def adjusted_input_lengths(
    n_time_steps: torch.Tensor,
    *,
    patch_size: int,
    patch_stride: int,
) -> torch.Tensor:
    if patch_size <= 0:
        return n_time_steps.to(torch.long)
    if patch_stride <= 0:
        raise ValueError("patch_stride must be positive when patch_size is enabled")
    lengths = torch.div(n_time_steps.to(torch.long) - patch_size, patch_stride, rounding_mode="floor") + 1
    return torch.clamp(lengths, min=0)


def gaussian_smooth(
    inputs: torch.Tensor,
    *,
    smooth_kernel_std: float,
    smooth_kernel_size: int,
) -> torch.Tensor:
    impulse = np.zeros(int(smooth_kernel_size), dtype=np.float32)
    impulse[int(smooth_kernel_size) // 2] = 1
    kernel = gaussian_filter1d(impulse, float(smooth_kernel_std))
    valid = np.argwhere(kernel > 0.01)
    kernel = np.squeeze(kernel[valid])
    kernel = kernel / np.sum(kernel)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=inputs.device).view(1, 1, -1)

    x = inputs.permute(0, 2, 1)
    channels = x.shape[1]
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1)
    smoothed = F.conv1d(x, kernel_tensor, padding="same", groups=channels)
    return smoothed.permute(0, 2, 1)


def _move_batch_to_device(batch: MutableMapping[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _decode_token_labels(
    ids: Sequence[int],
    id_to_label: Mapping[int, str],
    *,
    ignore_ids: set[int],
) -> List[str]:
    return [str(id_to_label.get(int(item), f"<unk:{int(item)}>")) for item in ids if int(item) not in ignore_ids]


def _loss_batch_with_adjusted_lengths(
    batch: Mapping[str, object],
    *,
    patch_size: int,
    patch_stride: int,
) -> Dict[str, torch.Tensor]:
    loss_batch = dict(batch)
    loss_batch["n_time_steps"] = adjusted_input_lengths(
        batch["n_time_steps"],
        patch_size=patch_size,
        patch_stride=patch_stride,
    )
    return loss_batch


def _edit_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _token_error_counts(
    reference: Sequence[int],
    hypothesis: Sequence[int],
    *,
    ignore_ids: set[int],
) -> tuple[int, int]:
    filtered_reference = [int(item) for item in reference if int(item) not in ignore_ids]
    filtered_hypothesis = [int(item) for item in hypothesis if int(item) not in ignore_ids]
    return _edit_distance(filtered_reference, filtered_hypothesis), len(filtered_reference)


def _paired_token_ids(
    syllable_ids: Sequence[int],
    tone_ids: Sequence[int],
    *,
    syllable_ignore_ids: set[int],
    tone_ignore_ids: set[int],
) -> List[tuple[int, int]]:
    syllables = [int(item) for item in syllable_ids if int(item) not in syllable_ignore_ids]
    tones = [int(item) for item in tone_ids if int(item) not in tone_ignore_ids]
    n_pairs = max(len(syllables), len(tones))
    return [
        (
            syllables[idx] if idx < len(syllables) else MISSING_SYLLABLE_ID,
            tones[idx] if idx < len(tones) else MISSING_TONE_ID,
        )
        for idx in range(n_pairs)
    ]


def _paired_token_error_counts(
    *,
    reference_syllables: Sequence[int],
    reference_tones: Sequence[int],
    hypothesis_syllables: Sequence[int],
    hypothesis_tones: Sequence[int],
    syllable_ignore_ids: set[int],
    tone_ignore_ids: set[int],
) -> tuple[int, int]:
    reference = _paired_token_ids(
        reference_syllables,
        reference_tones,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )
    hypothesis = _paired_token_ids(
        hypothesis_syllables,
        hypothesis_tones,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )
    return _edit_distance(reference, hypothesis), len(reference)


def decode_dual_stream_pairs(
    *,
    syllable_ids: Sequence[int],
    tone_ids: Sequence[int],
    id_to_syllable: Mapping[int, str],
    id_to_tone: Mapping[int, str],
    syllable_ignore_ids: set[int],
    tone_ignore_ids: set[int],
) -> List[str]:
    pairs = _paired_token_ids(
        syllable_ids,
        tone_ids,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )
    decoded: List[str] = []
    for syllable_id, tone_id in pairs:
        syllable = (
            "<missing_syllable>"
            if syllable_id == MISSING_SYLLABLE_ID
            else str(id_to_syllable.get(int(syllable_id), f"<unk:{syllable_id}>"))
        )
        tone = "?" if tone_id == MISSING_TONE_ID else str(id_to_tone.get(int(tone_id), f"<unk:{tone_id}>"))
        decoded.append(f"{syllable}{tone}")
    return decoded


def _ctc_greedy(logits: torch.Tensor, valid_len: int) -> List[int]:
    ids = torch.argmax(logits[:valid_len], dim=-1)
    ids = torch.unique_consecutive(ids, dim=-1)
    return [int(item) for item in ids.detach().cpu().tolist() if int(item) != 0]


def _evaluate(
    *,
    model: DualStreamGRUDecoder,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
    patch_stride: int,
    tone_weight: float,
    smooth_data: bool,
    smooth_kernel_std: float,
    smooth_kernel_size: int,
    syllable_ignore_ids: set[int],
    tone_ignore_ids: set[int],
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    syllable_losses: List[float] = []
    tone_losses: List[float] = []
    syllable_edits = 0
    syllable_total = 0
    tone_edits = 0
    tone_total = 0
    syllable_tone_edits = 0
    syllable_tone_total = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch_to_device(raw_batch, device)
            features = batch["input_features"]
            if smooth_data:
                features = gaussian_smooth(
                    features,
                    smooth_kernel_std=smooth_kernel_std,
                    smooth_kernel_size=smooth_kernel_size,
                )
            output = model(features, batch["day_indicies"])
            loss_batch = _loss_batch_with_adjusted_lengths(
                batch,
                patch_size=patch_size,
                patch_stride=patch_stride,
            )
            loss, loss_parts = dual_stream_ctc_loss(output, loss_batch, tone_weight=tone_weight)
            losses.append(float(loss.detach().cpu()))
            syllable_losses.append(float(loss_parts["syllable_loss"]))
            tone_losses.append(float(loss_parts["tone_loss"]))

            valid_lengths = loss_batch["n_time_steps"].detach().cpu().tolist()
            for row_idx, valid_len in enumerate(valid_lengths):
                pred_syllables = _ctc_greedy(output["syllable_logits"][row_idx], int(valid_len))
                true_syllables = batch["seq_syllable_ids"][row_idx][
                    : batch["syllable_seq_lens"][row_idx]
                ].detach().cpu().tolist()
                edits, total = _token_error_counts(
                    true_syllables,
                    pred_syllables,
                    ignore_ids=syllable_ignore_ids,
                )
                syllable_edits += edits
                syllable_total += total

                pred_tones = _ctc_greedy(output["tone_logits"][row_idx], int(valid_len))
                true_tones = batch["seq_tone_ids"][row_idx][
                    : batch["tone_seq_lens"][row_idx]
                ].detach().cpu().tolist()
                edits, total = _token_error_counts(
                    true_tones,
                    pred_tones,
                    ignore_ids=tone_ignore_ids,
                )
                tone_edits += edits
                tone_total += total

                edits, total = _paired_token_error_counts(
                    reference_syllables=true_syllables,
                    reference_tones=true_tones,
                    hypothesis_syllables=pred_syllables,
                    hypothesis_tones=pred_tones,
                    syllable_ignore_ids=syllable_ignore_ids,
                    tone_ignore_ids=tone_ignore_ids,
                )
                syllable_tone_edits += edits
                syllable_tone_total += total

    syllable_per = syllable_edits / max(1, syllable_total)
    tone_per = tone_edits / max(1, tone_total)
    syllable_tone_per = syllable_tone_edits / max(1, syllable_tone_total)

    return {
        "loss": float(np.mean(losses)) if losses else math.inf,
        "syllable_loss": float(np.mean(syllable_losses)) if syllable_losses else math.inf,
        "tone_loss": float(np.mean(tone_losses)) if tone_losses else math.inf,
        "syllable_per": syllable_per,
        "tone_per": tone_per,
        "syllable_tone_per": syllable_tone_per,
        "syllable_error_rate": syllable_per,
        "tone_error_rate": tone_per,
        "syllable_tone_error_rate": syllable_tone_per,
        "syllable_edits": float(syllable_edits),
        "syllable_total": float(syllable_total),
        "tone_edits": float(tone_edits),
        "tone_total": float(tone_total),
        "syllable_tone_edits": float(syllable_tone_edits),
        "syllable_tone_total": float(syllable_tone_total),
    }


def _make_loaders(
    *,
    session_dirs: Sequence[Path],
    label_maps: LabelMaps,
    batch_size: int,
    num_workers: int,
    expected_feature_dim: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_paths = [path / "data_train.hdf5" for path in session_dirs]
    val_paths = [path / "data_val.hdf5" for path in session_dirs]
    test_paths = [path / "data_test.hdf5" for path in session_dirs]
    train_dataset = ChineseDualStreamDataset(
        train_paths,
        split="train",
        label_remaps=label_maps.remaps,
        expected_feature_dim=expected_feature_dim,
    )
    val_dataset = ChineseDualStreamDataset(
        val_paths,
        split="val",
        label_remaps=label_maps.remaps,
        expected_feature_dim=expected_feature_dim,
    )
    test_dataset = ChineseDualStreamDataset(
        test_paths,
        split="test",
        label_remaps=label_maps.remaps,
        expected_feature_dim=expected_feature_dim,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_dual_stream,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dual_stream,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dual_stream,
    )
    return train_loader, val_loader, test_loader


def _next_batch(loader: DataLoader, iterator: object) -> tuple[object, object]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _device_from_config(config: DictConfig) -> torch.device:
    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _make_model(
    *,
    cfg: DictConfig,
    expected_feature_dim: int,
    n_days: int,
    label_maps: LabelMaps,
    device: torch.device,
) -> DualStreamGRUDecoder:
    return DualStreamGRUDecoder(
        neural_dim=expected_feature_dim,
        n_units=int(cfg.model.n_units),
        n_days=n_days,
        n_syllable_classes=len(label_maps.syllable_to_id),
        n_tone_classes=len(label_maps.tone_to_id),
        rnn_dropout=float(cfg.model.get("rnn_dropout", 0.0)),
        input_dropout=float(cfg.model.get("input_dropout", 0.0)),
        n_layers=int(cfg.model.n_layers),
        patch_size=int(cfg.model.get("patch_size", 0)),
        patch_stride=int(cfg.model.get("patch_stride", 0)),
    ).to(device)


def _ignore_ids(label_map: Mapping[str, int]) -> set[int]:
    ids = {int(label_map[BLANK_TOKEN])}
    if SIL_TOKEN in label_map:
        ids.add(int(label_map[SIL_TOKEN]))
    return ids


def _print_eval_metrics(prefix: str, metrics: Mapping[str, float], *, batch: int | None = None) -> None:
    batch_part = "" if batch is None else f"batch={batch} "
    fields = [f"{batch_part}{prefix}_loss={metrics['loss']:.4f}"]
    if "syllable_loss" in metrics and "tone_loss" in metrics:
        fields.extend(
            [
                f"{prefix}_syllable_loss={metrics['syllable_loss']:.4f}",
                f"{prefix}_tone_loss={metrics['tone_loss']:.4f}",
            ]
        )
    fields.extend(
        [
            f"{prefix}_syllable_per={metrics['syllable_per']:.4f}",
            f"{prefix}_tone_per={metrics['tone_per']:.4f}",
            f"{prefix}_syllable_tone_per={metrics['syllable_tone_per']:.4f}",
        ]
    )
    print(" ".join(fields))


def _is_better_checkpoint(
    metrics: Mapping[str, float],
    *,
    best_value: float,
    metric: str = "syllable_tone_per",
) -> bool:
    if metric not in metrics:
        raise KeyError(f"Validation metrics do not contain checkpoint metric {metric!r}")
    return float(metrics[metric]) < float(best_value)


def _default_checkpoint_path(output_dir: str | Path) -> Path:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    best = checkpoint_dir / "best.pt"
    latest = checkpoint_dir / "latest.pt"
    return best if best.exists() else latest


def _checkpoint_payload(
    *,
    model: DualStreamGRUDecoder,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    label_maps: LabelMaps,
    metrics: Mapping[str, object],
    batch: int,
    checkpoint_kind: str,
) -> Dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "label_maps": {
            "syllable_to_id": label_maps.syllable_to_id,
            "tone_to_id": label_maps.tone_to_id,
        },
        "metrics": dict(metrics),
        "batch": int(batch),
        "checkpoint_kind": checkpoint_kind,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: DualStreamGRUDecoder,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    label_maps: LabelMaps,
    metrics: Mapping[str, object],
    batch: int,
    checkpoint_kind: str,
) -> None:
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            label_maps=label_maps,
            metrics=metrics,
            batch=batch,
            checkpoint_kind=checkpoint_kind,
        ),
        path,
    )


def _prediction_rows(
    *,
    model: DualStreamGRUDecoder,
    loader: DataLoader,
    split: str,
    device: torch.device,
    patch_size: int,
    patch_stride: int,
    smooth_data: bool,
    smooth_kernel_std: float,
    smooth_kernel_size: int,
    label_maps: LabelMaps,
    syllable_ignore_ids: set[int],
    tone_ignore_ids: set[int],
) -> List[Dict[str, object]]:
    id_to_syllable = {idx: label for label, idx in label_maps.syllable_to_id.items()}
    id_to_tone = {idx: label for label, idx in label_maps.tone_to_id.items()}
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch_to_device(raw_batch, device)
            features = batch["input_features"]
            if smooth_data:
                features = gaussian_smooth(
                    features,
                    smooth_kernel_std=smooth_kernel_std,
                    smooth_kernel_size=smooth_kernel_size,
                )
            output = model(features, batch["day_indicies"])
            adjusted_lengths = adjusted_input_lengths(
                batch["n_time_steps"],
                patch_size=patch_size,
                patch_stride=patch_stride,
            ).detach().cpu().tolist()

            for row_idx, valid_len in enumerate(adjusted_lengths):
                pred_syllables = _ctc_greedy(output["syllable_logits"][row_idx], int(valid_len))
                pred_tones = _ctc_greedy(output["tone_logits"][row_idx], int(valid_len))
                true_syllables = batch["seq_syllable_ids"][row_idx][
                    : batch["syllable_seq_lens"][row_idx]
                ].detach().cpu().tolist()
                true_tones = batch["seq_tone_ids"][row_idx][
                    : batch["tone_seq_lens"][row_idx]
                ].detach().cpu().tolist()

                syllable_edits, syllable_total = _token_error_counts(
                    true_syllables,
                    pred_syllables,
                    ignore_ids=syllable_ignore_ids,
                )
                tone_edits, tone_total = _token_error_counts(
                    true_tones,
                    pred_tones,
                    ignore_ids=tone_ignore_ids,
                )
                paired_edits, paired_total = _paired_token_error_counts(
                    reference_syllables=true_syllables,
                    reference_tones=true_tones,
                    hypothesis_syllables=pred_syllables,
                    hypothesis_tones=pred_tones,
                    syllable_ignore_ids=syllable_ignore_ids,
                    tone_ignore_ids=tone_ignore_ids,
                )

                rows.append(
                    {
                        "split": split,
                        "session": raw_batch["session_names"][row_idx],
                        "block_num": int(raw_batch["block_nums"][row_idx]),
                        "trial_num": int(raw_batch["trial_nums"][row_idx]),
                        "transcription": raw_batch["transcriptions"][row_idx],
                        "true_syllables": " ".join(
                            _decode_token_labels(true_syllables, id_to_syllable, ignore_ids=syllable_ignore_ids)
                        ),
                        "true_tones": " ".join(_decode_token_labels(true_tones, id_to_tone, ignore_ids=tone_ignore_ids)),
                        "true_syllable_tone": " ".join(
                            decode_dual_stream_pairs(
                                syllable_ids=true_syllables,
                                tone_ids=true_tones,
                                id_to_syllable=id_to_syllable,
                                id_to_tone=id_to_tone,
                                syllable_ignore_ids=syllable_ignore_ids,
                                tone_ignore_ids=tone_ignore_ids,
                            )
                        ),
                        "pred_syllables": " ".join(
                            _decode_token_labels(pred_syllables, id_to_syllable, ignore_ids=syllable_ignore_ids)
                        ),
                        "pred_tones": " ".join(_decode_token_labels(pred_tones, id_to_tone, ignore_ids=tone_ignore_ids)),
                        "pred_syllable_tone": " ".join(
                            decode_dual_stream_pairs(
                                syllable_ids=pred_syllables,
                                tone_ids=pred_tones,
                                id_to_syllable=id_to_syllable,
                                id_to_tone=id_to_tone,
                                syllable_ignore_ids=syllable_ignore_ids,
                                tone_ignore_ids=tone_ignore_ids,
                            )
                        ),
                        "syllable_edits": syllable_edits,
                        "syllable_total": syllable_total,
                        "syllable_per": syllable_edits / max(1, syllable_total),
                        "tone_edits": tone_edits,
                        "tone_total": tone_total,
                        "tone_per": tone_edits / max(1, tone_total),
                        "syllable_tone_edits": paired_edits,
                        "syllable_tone_total": paired_total,
                        "syllable_tone_per": paired_edits / max(1, paired_total),
                    }
                )
    return rows


def _write_prediction_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "split",
        "session",
        "block_num",
        "trial_num",
        "transcription",
        "true_syllables",
        "true_tones",
        "true_syllable_tone",
        "pred_syllables",
        "pred_tones",
        "pred_syllable_tone",
        "syllable_edits",
        "syllable_total",
        "syllable_per",
        "tone_edits",
        "tone_total",
        "tone_per",
        "syllable_tone_edits",
        "syllable_tone_total",
        "syllable_tone_per",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def train_from_config(config: DictConfig | Mapping[str, object]) -> Dict[str, object]:
    cfg = OmegaConf.create(config)
    seed = int(cfg.get("seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    hdf5_root = _resolve_project_path(cfg.data.hdf5_root)
    session_dirs = discover_session_dirs(hdf5_root, cfg.data.get("sessions", "all"))
    label_maps = build_global_label_maps(session_dirs)

    first_train_file = session_dirs[0] / "data_train.hdf5"
    inferred_feature_dim = infer_feature_dim(first_train_file)
    expected_feature_dim = cfg.data.get("expected_feature_dim", None)
    if expected_feature_dim is None or str(expected_feature_dim).lower() == "auto":
        expected_feature_dim = inferred_feature_dim
    expected_feature_dim = int(expected_feature_dim)
    if inferred_feature_dim != expected_feature_dim:
        raise ValueError(
            f"First Chinese HDF5 has feature dim {inferred_feature_dim}, "
            f"but config expected {expected_feature_dim}"
        )

    output_dir = _resolve_project_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    if bool(cfg.get("save_checkpoint", True)):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = _make_loaders(
        session_dirs=session_dirs,
        label_maps=label_maps,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.get("num_workers", 0)),
        expected_feature_dim=expected_feature_dim,
    )

    device = _device_from_config(cfg)
    model = _make_model(
        cfg=cfg,
        expected_feature_dim=expected_feature_dim,
        n_days=len(session_dirs),
        label_maps=label_maps,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
    )

    patch_size = int(cfg.model.get("patch_size", 0))
    patch_stride = int(cfg.model.get("patch_stride", 0))
    tone_weight = float(cfg.training.get("tone_weight", 1.0))
    smooth_data = bool(cfg.training.get("smooth_data", True))
    smooth_kernel_std = float(cfg.training.get("smooth_kernel_std", 2.0))
    smooth_kernel_size = int(cfg.training.get("smooth_kernel_size", 100))
    grad_norm_clip = float(cfg.training.get("grad_norm_clip", 10.0))
    val_every = max(1, int(cfg.training.get("val_every", 100)))
    log_every = max(1, int(cfg.training.get("log_every", 20)))
    syllable_ignore_ids = _ignore_ids(label_maps.syllable_to_id)
    tone_ignore_ids = _ignore_ids(label_maps.tone_to_id)

    train_losses: List[float] = []
    val_metrics: List[Dict[str, float]] = []
    best_val_loss = math.inf
    best_checkpoint_metric = str(cfg.training.get("checkpoint_metric", "syllable_tone_per"))
    best_checkpoint_value = math.inf
    best_checkpoint_batch = 0
    iterator = iter(train_loader)

    for step in range(1, int(cfg.training.num_batches) + 1):
        raw_batch, iterator = _next_batch(train_loader, iterator)
        batch = _move_batch_to_device(raw_batch, device)
        features = batch["input_features"]
        if smooth_data:
            features = gaussian_smooth(
                features,
                smooth_kernel_std=smooth_kernel_std,
                smooth_kernel_size=smooth_kernel_size,
            )

        optimizer.zero_grad(set_to_none=True)
        model.train()
        output = model(features, batch["day_indicies"])
        loss_batch = _loss_batch_with_adjusted_lengths(
            batch,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        loss, loss_parts = dual_stream_ctc_loss(output, loss_batch, tone_weight=tone_weight)
        loss.backward()
        if grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm_clip)
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

        if step % log_every == 0 or step == 1:
            print(
                f"batch={step} loss={train_losses[-1]:.4f} "
                f"syllable_loss={loss_parts['syllable_loss']:.4f} "
                f"tone_loss={loss_parts['tone_loss']:.4f}"
            )

        if step % val_every == 0 or step == int(cfg.training.num_batches):
            metrics = _evaluate(
                model=model,
                loader=val_loader,
                device=device,
                patch_size=patch_size,
                patch_stride=patch_stride,
                tone_weight=tone_weight,
                smooth_data=smooth_data,
                smooth_kernel_std=smooth_kernel_std,
                smooth_kernel_size=smooth_kernel_size,
                syllable_ignore_ids=syllable_ignore_ids,
                tone_ignore_ids=tone_ignore_ids,
            )
            metrics["batch"] = step
            val_metrics.append(metrics)
            _print_eval_metrics("val", metrics, batch=step)
            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
            if _is_better_checkpoint(metrics, best_value=best_checkpoint_value, metric=best_checkpoint_metric):
                best_checkpoint_value = float(metrics[best_checkpoint_metric])
                best_checkpoint_batch = step
                if bool(cfg.get("save_checkpoint", True)):
                    _save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        cfg=cfg,
                        label_maps=label_maps,
                        metrics=metrics,
                        batch=step,
                        checkpoint_kind="best",
                    )

    final_metrics: Dict[str, object] = {}
    if bool(cfg.get("save_checkpoint", True)):
        _save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            label_maps=label_maps,
            metrics=val_metrics[-1] if val_metrics else {},
            batch=int(cfg.training.num_batches),
            checkpoint_kind="latest",
        )
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

    final_val_metrics = _evaluate(
        model=model,
        loader=val_loader,
        device=device,
        patch_size=patch_size,
        patch_stride=patch_stride,
        tone_weight=tone_weight,
        smooth_data=smooth_data,
        smooth_kernel_std=smooth_kernel_std,
        smooth_kernel_size=smooth_kernel_size,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )
    test_metrics = _evaluate(
        model=model,
        loader=test_loader,
        device=device,
        patch_size=patch_size,
        patch_stride=patch_stride,
        tone_weight=tone_weight,
        smooth_data=smooth_data,
        smooth_kernel_std=smooth_kernel_std,
        smooth_kernel_size=smooth_kernel_size,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )
    _print_eval_metrics("final_val", final_val_metrics)
    _print_eval_metrics("test", test_metrics)
    prediction_rows = []
    prediction_rows.extend(
        _prediction_rows(
            model=model,
            loader=val_loader,
            split="val",
            device=device,
            patch_size=patch_size,
            patch_stride=patch_stride,
            smooth_data=smooth_data,
            smooth_kernel_std=smooth_kernel_std,
            smooth_kernel_size=smooth_kernel_size,
            label_maps=label_maps,
            syllable_ignore_ids=syllable_ignore_ids,
            tone_ignore_ids=tone_ignore_ids,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            model=model,
            loader=test_loader,
            split="test",
            device=device,
            patch_size=patch_size,
            patch_stride=patch_stride,
            smooth_data=smooth_data,
            smooth_kernel_std=smooth_kernel_std,
            smooth_kernel_size=smooth_kernel_size,
            label_maps=label_maps,
            syllable_ignore_ids=syllable_ignore_ids,
            tone_ignore_ids=tone_ignore_ids,
        )
    )
    prediction_csv_path = output_dir / "val_test_predictions.csv"
    _write_prediction_csv(prediction_csv_path, prediction_rows)
    final_metrics["prediction_csv"] = str(prediction_csv_path)

    result: Dict[str, object] = {
        "device": str(device),
        "hdf5_root": str(hdf5_root),
        "sessions": [path.name for path in session_dirs],
        "n_days": len(session_dirs),
        "feature_dim": expected_feature_dim,
        "n_syllable_classes": len(label_maps.syllable_to_id),
        "n_tone_classes": len(label_maps.tone_to_id),
        "syllable_to_id": label_maps.syllable_to_id,
        "tone_to_id": label_maps.tone_to_id,
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_value": best_checkpoint_value,
        "best_checkpoint_batch": best_checkpoint_batch,
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        **final_metrics,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OmegaConf.save(config=cfg, f=output_dir / "config.yaml")

    return result


def evaluate_checkpoint_from_config(
    config: DictConfig | Mapping[str, object],
    *,
    checkpoint_path: str | Path | None = None,
) -> Dict[str, object]:
    cfg = OmegaConf.create(config)
    hdf5_root = _resolve_project_path(cfg.data.hdf5_root)
    session_dirs = discover_session_dirs(hdf5_root, cfg.data.get("sessions", "all"))
    label_maps = build_global_label_maps(session_dirs)

    first_train_file = session_dirs[0] / "data_train.hdf5"
    inferred_feature_dim = infer_feature_dim(first_train_file)
    expected_feature_dim = cfg.data.get("expected_feature_dim", None)
    if expected_feature_dim is None or str(expected_feature_dim).lower() == "auto":
        expected_feature_dim = inferred_feature_dim
    expected_feature_dim = int(expected_feature_dim)
    if inferred_feature_dim != expected_feature_dim:
        raise ValueError(
            f"First Chinese HDF5 has feature dim {inferred_feature_dim}, "
            f"but config expected {expected_feature_dim}"
        )

    _, val_loader, test_loader = _make_loaders(
        session_dirs=session_dirs,
        label_maps=label_maps,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.get("num_workers", 0)),
        expected_feature_dim=expected_feature_dim,
    )

    device = _device_from_config(cfg)
    model = _make_model(
        cfg=cfg,
        expected_feature_dim=expected_feature_dim,
        n_days=len(session_dirs),
        label_maps=label_maps,
        device=device,
    )
    if checkpoint_path is None:
        checkpoint_path = _default_checkpoint_path(_resolve_project_path(cfg.output_dir))
    else:
        checkpoint_path = _resolve_project_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    patch_size = int(cfg.model.get("patch_size", 0))
    patch_stride = int(cfg.model.get("patch_stride", 0))
    tone_weight = float(cfg.training.get("tone_weight", 1.0))
    smooth_data = bool(cfg.training.get("smooth_data", True))
    smooth_kernel_std = float(cfg.training.get("smooth_kernel_std", 2.0))
    smooth_kernel_size = int(cfg.training.get("smooth_kernel_size", 100))
    syllable_ignore_ids = _ignore_ids(label_maps.syllable_to_id)
    tone_ignore_ids = _ignore_ids(label_maps.tone_to_id)

    val_metrics = _evaluate(
        model=model,
        loader=val_loader,
        device=device,
        patch_size=patch_size,
        patch_stride=patch_stride,
        tone_weight=tone_weight,
        smooth_data=smooth_data,
        smooth_kernel_std=smooth_kernel_std,
        smooth_kernel_size=smooth_kernel_size,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )
    test_metrics = _evaluate(
        model=model,
        loader=test_loader,
        device=device,
        patch_size=patch_size,
        patch_stride=patch_stride,
        tone_weight=tone_weight,
        smooth_data=smooth_data,
        smooth_kernel_std=smooth_kernel_std,
        smooth_kernel_size=smooth_kernel_size,
        syllable_ignore_ids=syllable_ignore_ids,
        tone_ignore_ids=tone_ignore_ids,
    )

    result: Dict[str, object] = {
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "hdf5_root": str(hdf5_root),
        "sessions": [path.name for path in session_dirs],
        "n_days": len(session_dirs),
        "feature_dim": expected_feature_dim,
        "n_syllable_classes": len(label_maps.syllable_to_id),
        "n_tone_classes": len(label_maps.tone_to_id),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    output_dir = _resolve_project_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    prediction_rows.extend(
        _prediction_rows(
            model=model,
            loader=val_loader,
            split="val",
            device=device,
            patch_size=patch_size,
            patch_stride=patch_stride,
            smooth_data=smooth_data,
            smooth_kernel_std=smooth_kernel_std,
            smooth_kernel_size=smooth_kernel_size,
            label_maps=label_maps,
            syllable_ignore_ids=syllable_ignore_ids,
            tone_ignore_ids=tone_ignore_ids,
        )
    )
    prediction_rows.extend(
        _prediction_rows(
            model=model,
            loader=test_loader,
            split="test",
            device=device,
            patch_size=patch_size,
            patch_stride=patch_stride,
            smooth_data=smooth_data,
            smooth_kernel_std=smooth_kernel_std,
            smooth_kernel_size=smooth_kernel_size,
            label_maps=label_maps,
            syllable_ignore_ids=syllable_ignore_ids,
            tone_ignore_ids=tone_ignore_ids,
        )
    )
    prediction_csv_path = output_dir / "eval_val_test_predictions.csv"
    _write_prediction_csv(prediction_csv_path, prediction_rows)
    result["prediction_csv"] = str(prediction_csv_path)
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_eval_metrics("val", val_metrics)
    _print_eval_metrics("test", test_metrics)
    return result


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent Chinese syllable/tone decoder.")
    parser.add_argument("--config", type=str, default="chinese_speech/train_config.yaml")
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    if args.num_batches is not None:
        config.training.num_batches = int(args.num_batches)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    if args.eval_only:
        evaluate_checkpoint_from_config(config, checkpoint_path=args.checkpoint)
    else:
        train_from_config(config)


if __name__ == "__main__":
    main()
