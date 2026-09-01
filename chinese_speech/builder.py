from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from chinese_speech.labels import (
    BLANK_TOKEN,
    SIL_TOKEN,
    LabelSchema,
    Pronunciation,
    load_default_pronunciation_lexicon,
    normalize_chinese_text,
)

N_ELECTRODES = 256
BIN_SIZE_MS = 20
HISTORY_TRIALS = 20
STD_FLOOR = 0.05
Z_CLIP = 20.0


@dataclass(frozen=True)
class ChineseSession:
    session_name: str
    session_dir: Path
    csv_path: Path
    trial_data_path: Path
    config_path: Optional[Path]


@dataclass(frozen=True)
class TaskTrial:
    trial_num: int
    block_num: int
    text: str
    condition: str


class RollingChannelStats:
    def __init__(self, n_channels: int, max_trials: int):
        self.n_channels = n_channels
        self.max_trials = max_trials
        self.entries: Deque[Tuple[np.ndarray, np.ndarray, int]] = deque()
        self.total_sum = np.zeros(n_channels, dtype=np.float64)
        self.total_sumsq = np.zeros(n_channels, dtype=np.float64)
        self.total_count = 0

    def push(self, features: np.ndarray) -> None:
        arr = np.asarray(features, dtype=np.float64)
        sums = arr.sum(axis=0)
        sumsq = np.square(arr).sum(axis=0)
        n_rows = int(arr.shape[0])
        self.entries.append((sums, sumsq, n_rows))
        self.total_sum += sums
        self.total_sumsq += sumsq
        self.total_count += n_rows
        while len(self.entries) > self.max_trials:
            old_sum, old_sumsq, old_n = self.entries.popleft()
            self.total_sum -= old_sum
            self.total_sumsq -= old_sumsq
            self.total_count -= old_n

    def has_history(self) -> bool:
        return self.total_count > 0

    def mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.total_count <= 0:
            return (
                np.zeros(self.n_channels, dtype=np.float32),
                np.ones(self.n_channels, dtype=np.float32),
            )
        mean = self.total_sum / self.total_count
        var = self.total_sumsq / self.total_count - mean**2
        std = np.sqrt(np.maximum(var, 1e-8))
        std = np.maximum(std, STD_FLOOR)
        return mean.astype(np.float32), std.astype(np.float32)


def discover_chinese_sessions(root: Path) -> List[ChineseSession]:
    sessions: List[ChineseSession] = []
    for session_dir in sorted(Path(root).iterdir()):
        if not session_dir.is_dir():
            continue
        csv_files = sorted(session_dir.glob("data_*.csv"))
        trial_data = session_dir / "trial_data.mat"
        if not csv_files or not trial_data.exists():
            continue
        config_path = session_dir / "config.toml"
        sessions.append(
            ChineseSession(
                session_name=session_dir.name,
                session_dir=session_dir,
                csv_path=csv_files[0],
                trial_data_path=trial_data,
                config_path=config_path if config_path.exists() else None,
            )
        )
    return sessions


def load_task_trials(csv_path: Path) -> List[TaskTrial]:
    trials: List[TaskTrial] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("EventType") != "trial_start":
                continue
            trial_num = int(row["Data1"])
            block_num = int(row["Data2"])
            text = normalize_chinese_text(row.get("Data3", ""))
            trials.append(
                TaskTrial(
                    trial_num=trial_num,
                    block_num=block_num,
                    text=text,
                    condition="speech" if text else "blank",
                )
            )
    return trials


def _session_output_name(session_name: str) -> str:
    parts = session_name.split("-S", maxsplit=1)
    if len(parts) != 2:
        return f"sub01_zh_{session_name}_syllable_tone"
    date, suffix = parts
    date_part = date.replace("-", ".")
    return f"t15.{date_part}.S{suffix}_zh_syllable_tone"


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_trial_bounds(mask: np.ndarray) -> Dict[int, Tuple[int, int]]:
    flat = np.asarray(mask).reshape(-1)
    changes = np.flatnonzero(np.diff(flat) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [flat.size]))
    return {int(flat[start]): (int(start), int(end)) for start, end in zip(starts, ends)}


def _read_windows(state_bin: np.ndarray, trial_bounds: Mapping[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
    state = np.asarray(state_bin).reshape(-1)
    windows: Dict[int, Tuple[int, int]] = {}
    for trial_num, (start, end) in trial_bounds.items():
        trial_state = state[start:end]
        read_indices = np.flatnonzero(trial_state == 2)
        if read_indices.size == 0:
            raise ValueError(f"Trial {trial_num} has no state_bin == 2 read region.")
        windows[trial_num] = (start + int(read_indices[0]), start + int(read_indices[-1]) + 1)
    return windows


def _membership_from_electrodes(
    array_channel_unit: np.ndarray,
    neuron_mask: np.ndarray,
    n_electrodes: int,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    valid_cols = np.flatnonzero(np.asarray(neuron_mask).reshape(-1) > 0)
    if valid_cols.size == 0:
        raise ValueError("No valid units found in neuron_mask.")

    acu = np.asarray(array_channel_unit)
    electrodes = acu[3, valid_cols].astype(int) - 1
    if np.any((electrodes < 0) | (electrodes >= n_electrodes)):
        bad = sorted(set((electrodes[(electrodes < 0) | (electrodes >= n_electrodes)] + 1).tolist()))
        raise ValueError(f"Invalid electrode ids in array_channel_unit: {bad}")

    membership = np.zeros((valid_cols.size, n_electrodes), dtype=np.float32)
    membership[np.arange(valid_cols.size), electrodes] = 1.0
    dead_electrodes = [idx + 1 for idx in range(n_electrodes) if not np.any(membership[:, idx])]
    return membership, dead_electrodes, valid_cols


def _encode_transcription(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8") + b"\0", dtype=np.uint8)


def _pronunciation_string(pronunciation: Pronunciation) -> str:
    return " ".join(f"{syllable}{tone}" for syllable, tone in pronunciation)


def _is_diagnostic_trial(trial: TaskTrial) -> bool:
    return trial.condition == "speech" and len(trial.text) <= 1


def _is_included_trial(
    trial: TaskTrial,
    *,
    include_blank_trials: bool,
    include_diagnostic_trials: bool,
) -> bool:
    if trial.condition == "blank":
        return include_blank_trials
    if _is_diagnostic_trial(trial):
        return include_diagnostic_trials
    return trial.condition == "speech"


def _split_trials(
    trials: Sequence[TaskTrial],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Dict[int, str]:
    rng = np.random.default_rng(seed)
    trial_nums = np.asarray([trial.trial_num for trial in trials], dtype=int)
    rng.shuffle(trial_nums)
    n_total = len(trial_nums)
    n_test = 0 if test_fraction <= 0 else max(1, int(math.floor(n_total * test_fraction)))
    n_val = 0 if val_fraction <= 0 else max(1, int(math.floor(n_total * val_fraction)))
    if n_val + n_test >= n_total:
        raise ValueError(f"Invalid split fractions for {n_total} trials.")
    split_by_trial = {int(x): "test" for x in trial_nums[:n_test]}
    split_by_trial.update({int(x): "val" for x in trial_nums[n_test : n_test + n_val]})
    split_by_trial.update({int(x): "train" for x in trial_nums[n_test + n_val :]})
    return split_by_trial


class ChineseSpeechBuilder:
    def __init__(
        self,
        *,
        session_dir: Path,
        output_root: Path,
        subject: str = "sub-01",
        split_seed: int = 1,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        include_blank_trials: bool = False,
        include_diagnostic_trials: bool = False,
        overwrite: bool = False,
        n_electrodes: int = N_ELECTRODES,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.output_root = Path(output_root)
        self.subject = subject
        self.split_seed = split_seed
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.include_blank_trials = include_blank_trials
        self.include_diagnostic_trials = include_diagnostic_trials
        self.overwrite = overwrite
        self.n_electrodes = n_electrodes

        csv_files = sorted(self.session_dir.glob("data_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No data_*.csv found in {self.session_dir}")
        self.csv_path = csv_files[0]
        self.trial_data_path = self.session_dir / "trial_data.mat"
        if not self.trial_data_path.exists():
            raise FileNotFoundError(f"Missing trial_data.mat in {self.session_dir}")

    def build(self) -> Dict[str, object]:
        lexicon = load_default_pronunciation_lexicon()
        task_trials = load_task_trials(self.csv_path)
        labeled_trials = [
            trial
            for trial in task_trials
            if _is_included_trial(
                trial,
                include_blank_trials=self.include_blank_trials,
                include_diagnostic_trials=self.include_diagnostic_trials,
            )
        ]
        if not labeled_trials:
            raise ValueError(f"No included speech trials found in {self.session_dir}")
        schema = LabelSchema.from_texts([trial.text for trial in labeled_trials if trial.text], lexicon)
        split_by_trial = _split_trials(
            labeled_trials,
            seed=self.split_seed,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
        )

        output_dir = self.output_root / _session_output_name(self.session_dir.name)
        _ensure_output_dir(output_dir, overwrite=self.overwrite)

        h5_handles = {
            "train": h5py.File(output_dir / "data_train.hdf5", "w"),
            "val": h5py.File(output_dir / "data_val.hdf5", "w"),
            "test": h5py.File(output_dir / "data_test.hdf5", "w"),
        }
        split_counts = {"train": 0, "val": 0, "test": 0}
        manifest_rows: List[Dict[str, object]] = []
        read_bins: List[int] = []
        max_abs_z = 0.0
        raw_first_trials = 0

        try:
            with h5py.File(self.trial_data_path, "r") as raw:
                membership, dead_electrodes, valid_cols = _membership_from_electrodes(
                    raw["array_channel_unit"][:],
                    raw["neuron_mask"][:],
                    self.n_electrodes,
                )
                spike_bin = raw["spike_bin"]
                trial_bounds = _build_trial_bounds(raw["trial_mask"][:])
                windows = _read_windows(raw["state_bin"][:], trial_bounds)

                rolling = RollingChannelStats(self.n_electrodes, HISTORY_TRIALS)
                for task_trial in task_trials:
                    if _is_diagnostic_trial(task_trial) and not self.include_diagnostic_trials:
                        manifest_rows.append(
                            {
                                "subject": self.subject,
                                "session": self.session_dir.name,
                                "output_session": _session_output_name(self.session_dir.name),
                                "trial_num": task_trial.trial_num,
                                "block_num": task_trial.block_num,
                                "sentence_label": task_trial.text,
                                "condition": "diagnostic",
                                "split": "excluded_diagnostic",
                                "hdf5_group": "",
                                "pronunciation": "",
                            }
                        )
                        continue

                    if task_trial.condition == "blank" and not self.include_blank_trials:
                        manifest_rows.append(
                            {
                                "subject": self.subject,
                                "session": self.session_dir.name,
                                "output_session": _session_output_name(self.session_dir.name),
                                "trial_num": task_trial.trial_num,
                                "block_num": task_trial.block_num,
                                "sentence_label": "",
                                "condition": "blank",
                                "split": "no_action",
                                "hdf5_group": "",
                            }
                        )
                        continue

                    if task_trial.trial_num not in windows:
                        raise ValueError(f"CSV trial {task_trial.trial_num} not present in trial_mask.")
                    start, end = windows[task_trial.trial_num]
                    raw_slice = np.asarray(spike_bin[start:end, :], dtype=np.float32)[:, valid_cols]
                    n_raw = raw_slice.shape[0]
                    n_bins = n_raw // BIN_SIZE_MS
                    if n_bins <= 0:
                        raise ValueError(f"Trial {task_trial.trial_num} has no complete 20 ms bins.")
                    raw_slice = raw_slice[: n_bins * BIN_SIZE_MS]
                    binned_units = raw_slice.reshape(n_bins, BIN_SIZE_MS, valid_cols.size).sum(axis=1)
                    electrode_features = binned_units @ membership
                    split = split_by_trial[task_trial.trial_num]

                    if rolling.has_history():
                        mean, std = rolling.mean_std()
                        features = np.clip((electrode_features - mean) / std, -Z_CLIP, Z_CLIP)
                        max_abs_z = max(max_abs_z, float(np.abs(features).max()))
                    else:
                        features = electrode_features
                        raw_first_trials += 1
                    if split != "test":
                        rolling.push(electrode_features)
                    read_bins.append(int(n_bins))

                    if task_trial.text:
                        encoded = schema.encode_text(task_trial.text, lexicon)
                    else:
                        sil_syllable = schema.syllable_to_id[SIL_TOKEN]
                        sil_tone = schema.tone_to_id[SIL_TOKEN]
                        encoded = type("EncodedBlank", (), {})()
                        encoded.text = ""
                        encoded.pronunciation = []
                        encoded.syllable_ids = [sil_syllable]
                        encoded.tone_ids = [sil_tone]

                    group_name = f"trial_{split_counts[split]:04d}"
                    group = h5_handles[split].create_group(group_name)
                    group.create_dataset("input_features", data=features.astype(np.float32))
                    group.create_dataset("seq_class_ids", data=np.asarray(encoded.syllable_ids, dtype=np.int32))
                    group.create_dataset("seq_syllable_ids", data=np.asarray(encoded.syllable_ids, dtype=np.int32))
                    group.create_dataset("seq_tone_ids", data=np.asarray(encoded.tone_ids, dtype=np.int32))
                    group.create_dataset("transcription", data=_encode_transcription(task_trial.text))
                    group.attrs["subject"] = self.subject
                    group.attrs["session"] = _session_output_name(self.session_dir.name)
                    group.attrs["raw_session"] = self.session_dir.name
                    group.attrs["date"] = self.session_dir.name[:10]
                    group.attrs["block_num"] = int(task_trial.block_num)
                    group.attrs["trial_num"] = int(task_trial.trial_num)
                    group.attrs["split"] = split
                    group.attrs["corpus"] = "Mandarin"
                    group.attrs["sentence_label"] = task_trial.text.encode("utf-8")
                    group.attrs["n_time_steps"] = int(n_bins)
                    group.attrs["seq_len"] = len(encoded.syllable_ids)
                    group.attrs["tone_seq_len"] = len(encoded.tone_ids)
                    group.attrs["feature_type"] = (
                        "syllable_tone_electrode_zscore_prev20_non_test_statebin_read_"
                        "stdfloor0.05_clip20"
                    )
                    group.attrs["pronunciation"] = _pronunciation_string(encoded.pronunciation)
                    group.attrs["target_syllables"] = " ".join(s for s, _ in encoded.pronunciation)
                    group.attrs["target_tones"] = " ".join(str(t) for _, t in encoded.pronunciation)

                    manifest_rows.append(
                        {
                            "subject": self.subject,
                            "session": self.session_dir.name,
                            "output_session": _session_output_name(self.session_dir.name),
                            "trial_num": task_trial.trial_num,
                            "block_num": task_trial.block_num,
                            "sentence_label": task_trial.text,
                            "condition": task_trial.condition,
                            "split": split,
                            "hdf5_group": group_name,
                            "pronunciation": _pronunciation_string(encoded.pronunciation),
                        }
                    )
                    split_counts[split] += 1
        finally:
            for handle in h5_handles.values():
                handle.close()

        self._write_manifest(output_dir, manifest_rows)
        metadata = self._write_metadata(
            output_dir,
            schema=schema,
            n_task_trials=len(task_trials),
            n_labeled_trials=len(labeled_trials),
            n_diagnostic_trials=sum(1 for trial in task_trials if _is_diagnostic_trial(trial)),
            split_counts=split_counts,
            dead_electrodes=dead_electrodes,
            read_bins=read_bins,
            raw_first_trials=raw_first_trials,
            max_abs_z=max_abs_z,
        )

        return {
            "session": self.session_dir.name,
            "output_session": _session_output_name(self.session_dir.name),
            "output_dir": str(output_dir),
            "n_task_trials": len(task_trials),
            "n_labeled_trials": len(labeled_trials),
            "n_diagnostic_trials": sum(1 for trial in task_trials if _is_diagnostic_trial(trial)),
            "n_blank_trials": len(task_trials) - len([trial for trial in task_trials if trial.condition == "speech"]),
            "split_counts": split_counts,
            "metadata": metadata,
        }

    def _write_manifest(self, output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
        fieldnames = [
            "subject",
            "session",
            "output_session",
            "trial_num",
            "block_num",
            "sentence_label",
            "condition",
            "split",
            "hdf5_group",
            "pronunciation",
        ]
        with open(output_dir / "trial_manifest.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    def _write_metadata(
        self,
        output_dir: Path,
        *,
        schema: LabelSchema,
        n_task_trials: int,
        n_labeled_trials: int,
        n_diagnostic_trials: int,
        split_counts: Mapping[str, int],
        dead_electrodes: Sequence[int],
        read_bins: Sequence[int],
        raw_first_trials: int,
        max_abs_z: float,
    ) -> Dict[str, object]:
        metadata = {
            "subject": self.subject,
            "session": _session_output_name(self.session_dir.name),
            "raw_session": self.session_dir.name,
            "date": self.session_dir.name[:10],
            "source": {
                "session_dir": str(self.session_dir),
                "csv_path": str(self.csv_path),
                "trial_data_path": str(self.trial_data_path),
            },
            "n_task_trials": int(n_task_trials),
            "n_labeled_trials": int(n_labeled_trials),
            "include_blank_trials": bool(self.include_blank_trials),
            "include_diagnostic_trials": bool(self.include_diagnostic_trials),
            "n_diagnostic_trials": int(n_diagnostic_trials),
            "n_train": int(split_counts.get("train", 0)),
            "n_val": int(split_counts.get("val", 0)),
            "n_test": int(split_counts.get("test", 0)),
            "split": {
                "seed": int(self.split_seed),
                "train_fraction": 1.0 - self.val_fraction - self.test_fraction,
                "val_fraction": self.val_fraction,
                "test_fraction": self.test_fraction,
                "mode": "random_per_session_included_sentence_trials_only",
            },
            "features": {
                "mode": "electrode_sorted_spike_aggregation",
                "n_features": self.n_electrodes,
                "n_electrodes": self.n_electrodes,
                "dead_electrodes": list(dead_electrodes),
                "bin_size_ms": BIN_SIZE_MS,
                "read_window": "state_bin==2",
                "normalization": "causal_prev20_read_epochs_previous_non_test_trials_train_val_history",
                "smoothing": "none_in_preprocessing",
            },
            "labels": {
                **schema.to_json(),
                "blank_idx": schema.syllable_to_id[BLANK_TOKEN],
                "sil_idx": schema.syllable_to_id[SIL_TOKEN],
                "tone_blank_idx": schema.tone_to_id[BLANK_TOKEN],
                "tone_sil_idx": schema.tone_to_id[SIL_TOKEN],
                "scheme": "dual_stream_syllable_base_plus_tone_number_with_sil_start_end",
                "pronunciation_source": "chinese_speech.labels built-in overrides; no tone sandhi except explicit phrase overrides",
                "seq_class_ids_alias": "seq_syllable_ids",
                "english_compatible": "seq_class_ids is present as the syllable stream only; use dual-stream training for tone.",
            },
            "roll_stats": {
                "read_bins_per_trial": {
                    "min": int(np.min(read_bins)) if read_bins else 0,
                    "mean": float(np.mean(read_bins)) if read_bins else 0.0,
                    "max": int(np.max(read_bins)) if read_bins else 0,
                },
                "z_raw_first_trials": int(raw_first_trials),
                "max_abs_z": float(max_abs_z),
            },
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        return metadata


def build_all_sessions(
    *,
    speech_root: Path,
    output_root: Path,
    overwrite: bool,
    split_seed: int,
    include_blank_trials: bool,
    include_diagnostic_trials: bool = False,
) -> List[Dict[str, object]]:
    results = []
    for session in discover_chinese_sessions(speech_root):
        task_trials = load_task_trials(session.csv_path)
        if not any(
            _is_included_trial(
                trial,
                include_blank_trials=include_blank_trials,
                include_diagnostic_trials=include_diagnostic_trials,
            )
            for trial in task_trials
        ):
            continue
        builder = ChineseSpeechBuilder(
            session_dir=session.session_dir,
            output_root=output_root,
            split_seed=split_seed,
            include_blank_trials=include_blank_trials,
            include_diagnostic_trials=include_diagnostic_trials,
            overwrite=overwrite,
        )
        results.append(builder.build())
    return results


def default_output_root(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    return Path(project_root) / "data" / "hdf5_chinese"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_speech_root = project_root.parent / "sub-01" / "speech"

    parser = argparse.ArgumentParser(description="Build Mandarin syllable/tone HDF5 data.")
    parser.add_argument("--speech-root", type=Path, default=default_speech_root)
    parser.add_argument("--output-root", type=Path, default=default_output_root(project_root))
    parser.add_argument("--session", type=str, default=None, help="Optional single session folder name.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--include-blank-trials", action="store_true")
    parser.add_argument("--include-diagnostic-trials", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sessions = discover_chinese_sessions(args.speech_root)
    if args.session is not None:
        sessions = [session for session in sessions if session.session_name == args.session]
    if not sessions:
        raise SystemExit(f"No matching Chinese speech sessions found under {args.speech_root}")

    if args.dry_run:
        for session in sessions:
            trials = load_task_trials(session.csv_path)
            n_speech = sum(1 for trial in trials if trial.condition == "speech")
            n_diagnostic = sum(1 for trial in trials if _is_diagnostic_trial(trial))
            n_included = sum(
                1
                for trial in trials
                if _is_included_trial(
                    trial,
                    include_blank_trials=args.include_blank_trials,
                    include_diagnostic_trials=args.include_diagnostic_trials,
                )
            )
            print(
                f"{session.session_name}: trials={len(trials)} speech={n_speech} "
                f"sentence={n_speech-n_diagnostic} diagnostic={n_diagnostic} "
                f"blank={len(trials)-n_speech} included={n_included}"
            )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        task_trials = load_task_trials(session.csv_path)
        if not any(
            _is_included_trial(
                trial,
                include_blank_trials=args.include_blank_trials,
                include_diagnostic_trials=args.include_diagnostic_trials,
            )
            for trial in task_trials
        ):
            print(f"[skip] {session.session_name} has no included sentence trials")
            continue
        result = ChineseSpeechBuilder(
            session_dir=session.session_dir,
            output_root=args.output_root,
            split_seed=args.seed,
            include_blank_trials=args.include_blank_trials,
            include_diagnostic_trials=args.include_diagnostic_trials,
            overwrite=args.overwrite,
        ).build()
        counts = result["split_counts"]
        print(
            f"[ok] {result['session']} -> {result['output_session']} "
            f"(train={counts['train']}, val={counts['val']}, test={counts['test']})"
        )


if __name__ == "__main__":
    main()
