from __future__ import annotations

import argparse
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
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    syllable_edits = 0
    syllable_total = 0
    tone_edits = 0
    tone_total = 0
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
            loss, _ = dual_stream_ctc_loss(output, loss_batch, tone_weight=tone_weight)
            losses.append(float(loss.detach().cpu()))

            valid_lengths = loss_batch["n_time_steps"].detach().cpu().tolist()
            for row_idx, valid_len in enumerate(valid_lengths):
                pred_syllables = _ctc_greedy(output["syllable_logits"][row_idx], int(valid_len))
                true_syllables = batch["seq_syllable_ids"][row_idx][
                    : batch["syllable_seq_lens"][row_idx]
                ].detach().cpu().tolist()
                syllable_edits += _edit_distance(true_syllables, pred_syllables)
                syllable_total += len(true_syllables)

                pred_tones = _ctc_greedy(output["tone_logits"][row_idx], int(valid_len))
                true_tones = batch["seq_tone_ids"][row_idx][
                    : batch["tone_seq_lens"][row_idx]
                ].detach().cpu().tolist()
                tone_edits += _edit_distance(true_tones, pred_tones)
                tone_total += len(true_tones)

    return {
        "loss": float(np.mean(losses)) if losses else math.inf,
        "syllable_error_rate": syllable_edits / max(1, syllable_total),
        "tone_error_rate": tone_edits / max(1, tone_total),
    }


def _make_loaders(
    *,
    session_dirs: Sequence[Path],
    label_maps: LabelMaps,
    batch_size: int,
    num_workers: int,
    expected_feature_dim: int,
) -> tuple[DataLoader, DataLoader]:
    train_paths = [path / "data_train.hdf5" for path in session_dirs]
    val_paths = [path / "data_val.hdf5" for path in session_dirs]
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
    return train_loader, val_loader


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

    train_loader, val_loader = _make_loaders(
        session_dirs=session_dirs,
        label_maps=label_maps,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.get("num_workers", 0)),
        expected_feature_dim=expected_feature_dim,
    )

    device = _device_from_config(cfg)
    model = DualStreamGRUDecoder(
        neural_dim=expected_feature_dim,
        n_units=int(cfg.model.n_units),
        n_days=len(session_dirs),
        n_syllable_classes=len(label_maps.syllable_to_id),
        n_tone_classes=len(label_maps.tone_to_id),
        rnn_dropout=float(cfg.model.get("rnn_dropout", 0.0)),
        input_dropout=float(cfg.model.get("input_dropout", 0.0)),
        n_layers=int(cfg.model.n_layers),
        patch_size=int(cfg.model.get("patch_size", 0)),
        patch_stride=int(cfg.model.get("patch_stride", 0)),
    ).to(device)

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

    train_losses: List[float] = []
    val_metrics: List[Dict[str, float]] = []
    best_val_loss = math.inf
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
            )
            metrics["batch"] = step
            val_metrics.append(metrics)
            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]

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
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OmegaConf.save(config=cfg, f=output_dir / "config.yaml")

    if bool(cfg.get("save_checkpoint", True)):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "label_maps": {
                "syllable_to_id": label_maps.syllable_to_id,
                "tone_to_id": label_maps.tone_to_id,
            },
            "metrics": result,
        }
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

    return result


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent Chinese syllable/tone decoder.")
    parser.add_argument("--config", type=str, default="chinese_speech/train_config.yaml")
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
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
    train_from_config(config)


if __name__ == "__main__":
    main()
