from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

from chinese_speech.labels import (
    SIL_TOKEN,
    LabelSchema,
    Pronunciation,
    load_default_pronunciation_lexicon,
    normalize_chinese_text,
)


# ============================================================================
# Defaults
# ============================================================================
#
# Windows:
#   D:\wwl\data\self_mat\chinese_hdf5_self
#
# WSL automatically mounts D: under /mnt/d, so the same folder is:
#   /mnt/d/wwl/data/self_mat/chinese_hdf5_self
#
# Output requested for this project:
#   /home/speech/nejm-brain-to-text-cn/data/hdf5_chinese
#
DEFAULT_SOURCE_ROOT = Path("/mnt/d/wwl/data/self_mat/chinese_hdf5_self")
DEFAULT_OUTPUT_ROOT = Path("/home/speech/nejm-brain-to-text-cn/data/hdf5_chinese")

EXPECTED_SPLITS = ("train", "val", "test")
N_INPUT_FEATURES = 512

# Raw MATLAB session folders are expected to look like:
#   20260824-143620
#   20260824-xxxxxx
SESSION_RE = re.compile(
    r"^(?P<date>\d{8})-(?P<time>\d{6})(?:[-_].*)?$"
)


# ============================================================================
# Data classes
# ============================================================================

@dataclass(frozen=True)
class MatlabSession:
    raw_name: str
    session_dir: Path
    date_compact: str       # e.g. 20260824
    time_compact: str       # e.g. 143620
    session_code: str       # S2 or S4
    output_name: str        # t15.2026.08.24.S2_zh_syllable_tone
    mat_paths: Mapping[str, Path]
    manifest_path: Path


@dataclass(frozen=True)
class ManifestTrial:
    global_id: int
    task_trial_id: int
    trial_num: int
    block_index: int
    block_num: int
    split: str
    sentence_label: str
    calibration_source_type: str
    calibration_source_block_num: Optional[float]
    read_begin: str
    read_end: str
    n_20ms_bins: Optional[int]
    max_ptp_gap_ms: Optional[float]


# ============================================================================
# Small helpers
# ============================================================================

def _parse_optional_int(value: object) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_optional_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_date(date_compact: str) -> str:
    if len(date_compact) != 8 or not date_compact.isdigit():
        raise ValueError(f"Invalid compact date: {date_compact}")
    return (
        f"{date_compact[0:4]}-"
        f"{date_compact[4:6]}-"
        f"{date_compact[6:8]}"
    )


def _format_date_dotted(date_compact: str) -> str:
    return _format_date(date_compact).replace("-", ".")


def _session_output_name(date_compact: str, session_code: str) -> str:
    return (
        f"t15.{_format_date_dotted(date_compact)}."
        f"{session_code}_zh_syllable_tone"
    )


def _pronunciation_string(pronunciation: Pronunciation) -> str:
    return " ".join(f"{syllable}{tone}" for syllable, tone in pronunciation)


def _encode_transcription(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8") + b"\0", dtype=np.uint8)


def _ensure_clean_output_dir(path: Path, overwrite: bool) -> bool:
    """
    Return True when the caller should build this output directory.

    Default incremental behavior:
      - existing output + overwrite=False -> skip it
      - overwrite=True -> remove and rebuild
    """
    if path.exists():
        if not overwrite:
            print(f"[skip] output already exists: {path}")
            return False
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return True


# ============================================================================
# Session discovery and S2/S4 assignment
# ============================================================================

def discover_matlab_sessions(source_root: Path) -> List[MatlabSession]:
    """
    Scan immediate subfolders of source_root.

    Per-day mapping rule requested:
      first chronological folder  -> S2
      second chronological folder -> S4

    If more than two valid folders are present on one day, only the first two
    are assigned because no rule for a third session was specified.
    """
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(
            "Source root does not exist. In WSL, Windows D: should normally be "
            f"mounted under /mnt/d.\nSource root: {source_root}"
        )

    candidates: List[Tuple[str, str, str, Path]] = []

    for session_dir in sorted(source_root.iterdir(), key=lambda p: p.name):
        if not session_dir.is_dir():
            continue

        match = SESSION_RE.match(session_dir.name)
        if match is None:
            print(
                f"[skip] {session_dir.name}: folder name does not match "
                "YYYYMMDD-HHMMSS"
            )
            continue

        date_compact = match.group("date")
        time_compact = match.group("time")

        mat_paths = {
            split: session_dir / f"data_{split}.mat"
            for split in EXPECTED_SPLITS
            if (session_dir / f"data_{split}.mat").is_file()
        }

        # Requested behavior: folders with no .mat are simply ignored.
        any_mat = any(session_dir.glob("*.mat"))
        if not any_mat:
            print(f"[skip] {session_dir.name}: no .mat files")
            continue

        # We specifically know how to convert the MATLAB preprocessing outputs
        # data_train.mat / data_val.mat / data_test.mat.
        if not mat_paths:
            print(
                f"[skip] {session_dir.name}: .mat exists, but none of "
                "data_train.mat/data_val.mat/data_test.mat were found"
            )
            continue

        manifest_path = session_dir / "trial_manifest.csv"
        if not manifest_path.is_file():
            print(
                f"[skip] {session_dir.name}: missing trial_manifest.csv "
                "(needed to map global_id to sentence labels)"
            )
            continue

        candidates.append(
            (date_compact, time_compact, session_dir.name, session_dir)
        )

    # Group by day, then sort each day chronologically.
    grouped: Dict[str, List[Tuple[str, str, Path]]] = {}
    for date_compact, time_compact, raw_name, session_dir in candidates:
        grouped.setdefault(date_compact, []).append(
            (time_compact, raw_name, session_dir)
        )

    sessions: List[MatlabSession] = []

    for date_compact in sorted(grouped):
        day_sessions = sorted(
            grouped[date_compact],
            key=lambda x: (x[0], x[1]),
        )

        if len(day_sessions) > 2:
            extras = ", ".join(item[1] for item in day_sessions[2:])
            print(
                f"[warn] {date_compact}: found {len(day_sessions)} convertible "
                f"sessions. Only first two are mapped (S2/S4); extra folders "
                f"will be skipped: {extras}"
            )

        for day_index, (time_compact, raw_name, session_dir) in enumerate(
            day_sessions[:2]
        ):
            session_code = "S2" if day_index == 0 else "S4"

            mat_paths = {
                split: session_dir / f"data_{split}.mat"
                for split in EXPECTED_SPLITS
                if (session_dir / f"data_{split}.mat").is_file()
            }

            sessions.append(
                MatlabSession(
                    raw_name=raw_name,
                    session_dir=session_dir,
                    date_compact=date_compact,
                    time_compact=time_compact,
                    session_code=session_code,
                    output_name=_session_output_name(
                        date_compact, session_code
                    ),
                    mat_paths=mat_paths,
                    manifest_path=session_dir / "trial_manifest.csv",
                )
            )

    return sessions


# ============================================================================
# Manifest loading
# ============================================================================

def load_trial_manifest(path: Path) -> Dict[int, ManifestTrial]:
    rows: Dict[int, ManifestTrial] = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            "global_id",
            "trial_num",
            "block_num",
            "split",
            "sentence_label",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        for raw in reader:
            global_id = int(raw["global_id"])
            if global_id in rows:
                raise ValueError(
                    f"Duplicate global_id={global_id} in {path}"
                )

            rows[global_id] = ManifestTrial(
                global_id=global_id,
                task_trial_id=int(raw.get("task_trial_id", global_id)),
                trial_num=int(raw["trial_num"]),
                block_index=int(raw.get("block_index", 0) or 0),
                block_num=int(float(raw["block_num"])),
                split=str(raw["split"]).strip(),
                sentence_label=normalize_chinese_text(
                    raw.get("sentence_label", "")
                ),
                calibration_source_type=str(
                    raw.get("calibration_source_type", "")
                ).strip(),
                calibration_source_block_num=_parse_optional_float(
                    raw.get("calibration_source_block_num", "")
                ),
                read_begin=str(raw.get("read_begin", "")).strip(),
                read_end=str(raw.get("read_end", "")).strip(),
                n_20ms_bins=_parse_optional_int(
                    raw.get("n_20ms_bins", "")
                ),
                max_ptp_gap_ms=_parse_optional_float(
                    raw.get("max_ptp_gap_ms", "")
                ),
            )

    return rows


# ============================================================================
# MATLAB v7.3 / HDF5 readers
# ============================================================================

def _require_dataset_group(handle: h5py.File) -> h5py.Group:
    if "dataset" not in handle:
        raise KeyError(
            "Top-level MATLAB variable 'dataset' was not found. "
            "Expected files created by save(..., 'dataset', '-v7.3')."
        )

    group = handle["dataset"]
    if not isinstance(group, h5py.Group):
        raise TypeError("'dataset' exists but is not an HDF5 group.")

    return group


def _matlab_numeric_vector(
    group: h5py.Group,
    field_name: str,
    dtype: np.dtype,
) -> np.ndarray:
    if field_name not in group:
        raise KeyError(
            f"MATLAB dataset is missing field '{field_name}'."
        )

    obj = group[field_name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"dataset.{field_name} is not an HDF5 dataset."
        )

    ref_type = h5py.check_dtype(ref=obj.dtype)
    if ref_type is not None:
        raise TypeError(
            f"dataset.{field_name} unexpectedly contains object references."
        )

    values = np.asarray(obj)
    # MATLAB column vectors often appear as 1 x N through h5py because
    # MATLAB v7.3 stores dimensions in HDF5 order. Flattening is sufficient
    # for these scalar-per-trial fields.
    return values.reshape(-1).astype(dtype, copy=False)


def _matlab_cell_references(
    group: h5py.Group,
    field_name: str,
) -> np.ndarray:
    if field_name not in group:
        raise KeyError(
            f"MATLAB dataset is missing cell field '{field_name}'."
        )

    obj = group[field_name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"dataset.{field_name} is not an HDF5 dataset."
        )

    if h5py.check_dtype(ref=obj.dtype) is None:
        raise TypeError(
            f"dataset.{field_name} is expected to be a MATLAB cell array "
            "(HDF5 object references)."
        )

    return np.asarray(obj).reshape(-1)


def _orient_trial_matrix(
    array: np.ndarray,
    *,
    n_features: int,
    field_name: str,
    global_id: int,
) -> np.ndarray:
    """
    Recover the MATLAB T x F orientation.

    MATLAB v7.3 matrices are commonly exposed by h5py with reversed
    dimensions, so a MATLAB T x 512 matrix is usually seen as 512 x T.
    """
    array = np.asarray(array)

    if array.ndim == 1:
        if array.size != n_features:
            raise ValueError(
                f"global_id={global_id} {field_name}: one-dimensional array "
                f"has {array.size} values; expected {n_features}."
            )
        array = array.reshape(1, n_features)

    if array.ndim != 2:
        raise ValueError(
            f"global_id={global_id} {field_name}: expected a 2-D matrix, "
            f"got shape {array.shape}."
        )

    if array.shape[1] == n_features:
        result = array
    elif array.shape[0] == n_features:
        result = array.T
    else:
        raise ValueError(
            f"global_id={global_id} {field_name}: cannot identify feature "
            f"dimension {n_features} from HDF5 shape {array.shape}."
        )

    if result.shape[0] <= 0:
        raise ValueError(
            f"global_id={global_id} {field_name}: no time bins."
        )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"global_id={global_id} {field_name}: contains NaN or Inf."
        )

    return np.asarray(result, dtype=np.float32)


def _read_feature_ref(
    handle: h5py.File,
    ref: h5py.Reference,
    *,
    global_id: int,
) -> np.ndarray:
    if not ref:
        raise ValueError(
            f"global_id={global_id}: empty HDF5 reference in input_features."
        )

    obj = handle[ref]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"global_id={global_id}: input_features cell does not reference "
            "an HDF5 dataset."
        )

    return _orient_trial_matrix(
        np.asarray(obj),
        n_features=N_INPUT_FEATURES,
        field_name="input_features",
        global_id=global_id,
    )


def inspect_mat_split(
    mat_path: Path,
) -> Tuple[np.ndarray, int]:
    """
    Return ordered global_ids and number of input_features cells.
    """
    try:
        with h5py.File(mat_path, "r") as handle:
            group = _require_dataset_group(handle)
            global_ids = _matlab_numeric_vector(
                group, "global_id", np.int64
            )
            refs = _matlab_cell_references(
                group, "input_features"
            )
    except OSError as exc:
        raise OSError(
            f"{mat_path} is not readable as MATLAB v7.3/HDF5. "
            "The source MATLAB script should save with -v7.3."
        ) from exc

    if global_ids.size != refs.size:
        raise ValueError(
            f"{mat_path}: global_id has {global_ids.size} trials but "
            f"input_features has {refs.size} cells."
        )

    return global_ids, int(refs.size)


# ============================================================================
# HDF5 conversion
# ============================================================================

class ChineseSpeechBuilder:
    def __init__(
        self,
        *,
        session: MatlabSession,
        output_root: Path,
        subject: str = "sub-01",
        overwrite: bool = False,
    ) -> None:
        self.session = session
        self.output_root = Path(output_root)
        self.subject = subject
        self.overwrite = overwrite

    def build(self) -> Optional[Dict[str, object]]:
        output_dir = self.output_root / self.session.output_name

        if not _ensure_clean_output_dir(
            output_dir, overwrite=self.overwrite
        ):
            return None

        try:
            manifest = load_trial_manifest(
                self.session.manifest_path
            )

            # Inspect available split MAT files first. This gives us the exact
            # trial IDs actually present in each MAT and lets us build one
            # consistent label schema for train/val/test.
            split_global_ids: Dict[str, np.ndarray] = {}
            all_global_ids: List[int] = []

            for split in EXPECTED_SPLITS:
                mat_path = self.session.mat_paths.get(split)
                if mat_path is None:
                    print(
                        f"[warn] {self.session.raw_name}: "
                        f"data_{split}.mat is missing; that split will not "
                        "be written"
                    )
                    continue

                global_ids, _ = inspect_mat_split(mat_path)
                split_global_ids[split] = global_ids
                all_global_ids.extend(
                    int(x) for x in global_ids.tolist()
                )

            if not split_global_ids:
                raise ValueError(
                    f"{self.session.raw_name}: no convertible data_*.mat files."
                )

            if len(set(all_global_ids)) != len(all_global_ids):
                raise ValueError(
                    f"{self.session.raw_name}: the same global_id appears "
                    "in more than one split MAT file."
                )

            missing_manifest = [
                gid for gid in all_global_ids if gid not in manifest
            ]
            if missing_manifest:
                preview = missing_manifest[:10]
                raise ValueError(
                    f"{self.session.raw_name}: MAT global_id values are missing "
                    f"from trial_manifest.csv, e.g. {preview}"
                )

            # Check that the split stored in the manifest agrees with which
            # MAT file the trial came from.
            for split, gids in split_global_ids.items():
                bad = [
                    int(gid)
                    for gid in gids
                    if manifest[int(gid)].split
                    and manifest[int(gid)].split != split
                ]
                if bad:
                    raise ValueError(
                        f"{self.session.raw_name}: manifest split disagrees "
                        f"with data_{split}.mat for global_id(s) {bad[:10]}"
                    )

            texts = [
                manifest[gid].sentence_label
                for gid in all_global_ids
                if manifest[gid].sentence_label
            ]
            if not texts:
                raise ValueError(
                    f"{self.session.raw_name}: no non-empty sentence labels "
                    "found for the MAT trials."
                )

            lexicon = load_default_pronunciation_lexicon()
            schema = LabelSchema.from_texts(texts, lexicon)

            output_manifest_rows: List[Dict[str, object]] = []
            split_counts: Dict[str, int] = {
                "train": 0,
                "val": 0,
                "test": 0,
            }
            feature_lengths: List[int] = []
            max_abs_input = 0.0

            for split in EXPECTED_SPLITS:
                mat_path = self.session.mat_paths.get(split)
                if mat_path is None:
                    continue

                out_h5 = output_dir / f"data_{split}.hdf5"
                count, rows, lengths, max_abs = self._convert_split(
                    split=split,
                    mat_path=mat_path,
                    output_path=out_h5,
                    manifest=manifest,
                    schema=schema,
                    lexicon=lexicon,
                )

                split_counts[split] = count
                output_manifest_rows.extend(rows)
                feature_lengths.extend(lengths)
                max_abs_input = max(max_abs_input, max_abs)

            self._write_manifest(
                output_dir,
                output_manifest_rows,
            )
            metadata = self._write_metadata(
                output_dir,
                schema=schema,
                split_counts=split_counts,
                feature_lengths=feature_lengths,
                max_abs_input=max_abs_input,
            )

            return {
                "raw_session": self.session.raw_name,
                "session_code": self.session.session_code,
                "output_session": self.session.output_name,
                "output_dir": str(output_dir),
                "split_counts": split_counts,
                "metadata": metadata,
            }

        except Exception:
            # Do not leave a half-built t15... folder behind. This is
            # especially useful when the script is rerun incrementally.
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise

    def _convert_split(
        self,
        *,
        split: str,
        mat_path: Path,
        output_path: Path,
        manifest: Mapping[int, ManifestTrial],
        schema: LabelSchema,
        lexicon,
    ) -> Tuple[
        int,
        List[Dict[str, object]],
        List[int],
        float,
    ]:
        rows: List[Dict[str, object]] = []
        lengths: List[int] = []
        max_abs_input = 0.0

        with h5py.File(mat_path, "r") as source, h5py.File(
            output_path, "w"
        ) as target:
            group = _require_dataset_group(source)

            global_ids = _matlab_numeric_vector(
                group, "global_id", np.int64
            )
            feature_refs = _matlab_cell_references(
                group, "input_features"
            )

            if global_ids.size != feature_refs.size:
                raise ValueError(
                    f"{mat_path}: global_id count {global_ids.size} != "
                    f"input_features count {feature_refs.size}"
                )

            for local_index, (global_id_raw, feature_ref) in enumerate(
                zip(global_ids.tolist(), feature_refs.tolist())
            ):
                global_id = int(global_id_raw)
                trial = manifest[global_id]

                features = _read_feature_ref(
                    source,
                    feature_ref,
                    global_id=global_id,
                )

                if trial.n_20ms_bins is not None:
                    if features.shape[0] != trial.n_20ms_bins:
                        raise ValueError(
                            f"{self.session.raw_name} global_id={global_id}: "
                            f"MAT has {features.shape[0]} time bins but "
                            f"manifest says {trial.n_20ms_bins}."
                        )

                text = trial.sentence_label

                if text:
                    encoded = schema.encode_text(text, lexicon)
                    pronunciation = encoded.pronunciation
                    syllable_ids = np.asarray(
                        encoded.syllable_ids,
                        dtype=np.int32,
                    )
                    tone_ids = np.asarray(
                        encoded.tone_ids,
                        dtype=np.int32,
                    )
                else:
                    # Kept for robustness, although the current sentence MAT
                    # files are expected to contain sentence trials.
                    sil_syllable = schema.syllable_to_id[SIL_TOKEN]
                    sil_tone = schema.tone_to_id[SIL_TOKEN]
                    pronunciation = []
                    syllable_ids = np.asarray(
                        [sil_syllable], dtype=np.int32
                    )
                    tone_ids = np.asarray(
                        [sil_tone], dtype=np.int32
                    )

                group_name = f"trial_{local_index:04d}"
                out_group = target.create_group(group_name)

                # IMPORTANT: input_features already contains the final MATLAB
                # preprocessing output:
                #   T x 512 = [TC(256) | SBP(256)]
                # and, for the uploaded preprocessing configuration, already
                # includes the causal previous-20-trial z-score.
                # Do NOT normalize it again here.
                out_group.create_dataset(
                    "input_features",
                    data=features,
                    dtype=np.float32,
                )
                out_group.create_dataset(
                    "seq_class_ids",
                    data=syllable_ids,
                )
                out_group.create_dataset(
                    "seq_syllable_ids",
                    data=syllable_ids,
                )
                out_group.create_dataset(
                    "seq_tone_ids",
                    data=tone_ids,
                )
                out_group.create_dataset(
                    "transcription",
                    data=_encode_transcription(text),
                )

                out_group.attrs["subject"] = self.subject
                out_group.attrs["session"] = self.session.output_name
                out_group.attrs["raw_session"] = self.session.raw_name
                out_group.attrs["date"] = _format_date(
                    self.session.date_compact
                )
                out_group.attrs["session_code"] = (
                    self.session.session_code
                )
                out_group.attrs["global_id"] = global_id
                out_group.attrs["task_trial_id"] = int(
                    trial.task_trial_id
                )
                out_group.attrs["block_num"] = int(
                    trial.block_num
                )
                out_group.attrs["trial_num"] = int(
                    trial.trial_num
                )
                out_group.attrs["split"] = split
                out_group.attrs["corpus"] = "Mandarin"
                out_group.attrs["sentence_label"] = text.encode(
                    "utf-8"
                )
                out_group.attrs["n_time_steps"] = int(
                    features.shape[0]
                )
                out_group.attrs["n_input_features"] = int(
                    features.shape[1]
                )
                out_group.attrs["seq_len"] = int(
                    syllable_ids.size
                )
                out_group.attrs["tone_seq_len"] = int(
                    tone_ids.size
                )
                out_group.attrs["feature_type"] = (
                    "tc_sbp_512_prevblock_lrr_rms_prev20z"
                )
                out_group.attrs["feature_order"] = (
                    "[TC physical electrodes 1:256 | "
                    "SBP physical electrodes 1:256]"
                )
                out_group.attrs["source_mat"] = mat_path.name
                out_group.attrs[
                    "calibration_source_type"
                ] = trial.calibration_source_type

                if (
                    trial.calibration_source_block_num
                    is not None
                ):
                    out_group.attrs[
                        "calibration_source_block_num"
                    ] = float(
                        trial.calibration_source_block_num
                    )

                pronunciation_text = _pronunciation_string(
                    pronunciation
                )
                out_group.attrs[
                    "pronunciation"
                ] = pronunciation_text
                out_group.attrs["target_syllables"] = " ".join(
                    syllable for syllable, _ in pronunciation
                )
                out_group.attrs["target_tones"] = " ".join(
                    str(tone) for _, tone in pronunciation
                )

                n_time_steps = int(features.shape[0])
                lengths.append(n_time_steps)
                max_abs_input = max(
                    max_abs_input,
                    float(np.max(np.abs(features))),
                )

                rows.append(
                    {
                        "subject": self.subject,
                        "raw_session": self.session.raw_name,
                        "output_session": self.session.output_name,
                        "session_code": self.session.session_code,
                        "global_id": global_id,
                        "task_trial_id": trial.task_trial_id,
                        "trial_num": trial.trial_num,
                        "block_index": trial.block_index,
                        "block_num": trial.block_num,
                        "sentence_label": text,
                        "split": split,
                        "hdf5_group": group_name,
                        "n_time_steps": n_time_steps,
                        "read_begin": trial.read_begin,
                        "read_end": trial.read_end,
                        "calibration_source_type": (
                            trial.calibration_source_type
                        ),
                        "calibration_source_block_num": (
                            ""
                            if trial.calibration_source_block_num
                            is None
                            else trial.calibration_source_block_num
                        ),
                        "pronunciation": pronunciation_text,
                    }
                )

        return (
            len(rows),
            rows,
            lengths,
            max_abs_input,
        )

    def _write_manifest(
        self,
        output_dir: Path,
        rows: Sequence[Mapping[str, object]],
    ) -> None:
        fieldnames = [
            "subject",
            "raw_session",
            "output_session",
            "session_code",
            "global_id",
            "task_trial_id",
            "trial_num",
            "block_index",
            "block_num",
            "sentence_label",
            "split",
            "hdf5_group",
            "n_time_steps",
            "read_begin",
            "read_end",
            "calibration_source_type",
            "calibration_source_block_num",
            "pronunciation",
        ]

        with open(
            output_dir / "trial_manifest.csv",
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        name: row.get(name, "")
                        for name in fieldnames
                    }
                )

    def _write_metadata(
        self,
        output_dir: Path,
        *,
        schema: LabelSchema,
        split_counts: Mapping[str, int],
        feature_lengths: Sequence[int],
        max_abs_input: float,
    ) -> Dict[str, object]:
        source_mats = {
            split: str(path)
            for split, path in self.session.mat_paths.items()
        }

        metadata = {
            "subject": self.subject,
            "session": self.session.output_name,
            "raw_session": self.session.raw_name,
            "date": _format_date(
                self.session.date_compact
            ),
            "session_code": self.session.session_code,
            "session_assignment": {
                "rule": (
                    "within each date, first chronological "
                    "convertible folder -> S2; second -> S4"
                ),
                "raw_time": self.session.time_compact,
            },
            "source": {
                "session_dir": str(
                    self.session.session_dir
                ),
                "trial_manifest_csv": str(
                    self.session.manifest_path
                ),
                "mat_files": source_mats,
            },
            "n_train": int(
                split_counts.get("train", 0)
            ),
            "n_val": int(
                split_counts.get("val", 0)
            ),
            "n_test": int(
                split_counts.get("test", 0)
            ),
            "split": {
                "mode": (
                    "preserve_existing_matlab_train_val_test_split"
                ),
            },
            "features": {
                "mode": (
                    "direct_from_matlab_dataset.input_features"
                ),
                "n_features": N_INPUT_FEATURES,
                "feature_order": (
                    "[TC physical electrodes 1:256 | "
                    "SBP physical electrodes 1:256]"
                ),
                "bin_size_ms": 20,
                "normalization": (
                    "already_applied_in_matlab_causal_prev20"
                ),
                "additional_python_normalization": False,
                "time_bins_per_trial": {
                    "min": (
                        int(np.min(feature_lengths))
                        if feature_lengths
                        else 0
                    ),
                    "mean": (
                        float(np.mean(feature_lengths))
                        if feature_lengths
                        else 0.0
                    ),
                    "max": (
                        int(np.max(feature_lengths))
                        if feature_lengths
                        else 0
                    ),
                },
                "max_abs_input_features": float(
                    max_abs_input
                ),
            },
            "labels": {
                **schema.to_json(),
                "scheme": (
                    "dual_stream_syllable_base_plus_"
                    "tone_number_with_sil_start_end"
                ),
                "seq_class_ids_alias": (
                    "seq_syllable_ids"
                ),
            },
        }

        with open(
            output_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        return metadata


# ============================================================================
# Batch build
# ============================================================================

def build_all_sessions(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    overwrite: bool = False,
    subject: str = "sub-01",
    only_session: Optional[str] = None,
    fail_fast: bool = False,
) -> List[Dict[str, object]]:
    sessions = discover_matlab_sessions(source_root)

    if only_session is not None:
        sessions = [
            session
            for session in sessions
            if session.raw_name == only_session
            or session.output_name == only_session
        ]

    if not sessions:
        print(
            f"[done] no convertible sessions found under {source_root}"
        )
        return []

    Path(output_root).mkdir(
        parents=True, exist_ok=True
    )

    results: List[Dict[str, object]] = []
    failures: List[Tuple[str, str]] = []

    for session in sessions:
        available = ", ".join(
            f"data_{split}.mat"
            for split in EXPECTED_SPLITS
            if split in session.mat_paths
        )

        print(
            f"\n=== {session.raw_name} -> "
            f"{session.output_name} ==="
        )
        print(
            f"source: {session.session_dir}"
        )
        print(
            f"MAT: {available}"
        )

        try:
            result = ChineseSpeechBuilder(
                session=session,
                output_root=output_root,
                subject=subject,
                overwrite=overwrite,
            ).build()

            if result is None:
                continue

            counts = result["split_counts"]
            print(
                f"[ok] {session.raw_name} -> "
                f"{session.output_name} "
                f"(train={counts['train']}, "
                f"val={counts['val']}, "
                f"test={counts['test']})"
            )
            results.append(result)

        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(
                (session.raw_name, message)
            )
            print(
                f"[error] {session.raw_name}: "
                f"{message}"
            )
            if fail_fast:
                raise

    print("\n=== Batch summary ===")
    print(
        f"converted={len(results)} "
        f"failed={len(failures)}"
    )

    if failures:
        for raw_name, message in failures:
            print(
                f"  FAILED {raw_name}: {message}"
            )

    return results


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MATLAB sentence preprocessing outputs under the "
            "Windows D: drive (mounted in WSL as /mnt/d) into the Mandarin "
            "training HDF5 layout."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=(
            "Source root containing YYYYMMDD-HHMMSS folders. "
            "Default: /mnt/d/wwl/data/self_mat/chinese_hdf5_self"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "WSL output root. "
            "Default: /home/speech/nejm-brain-to-text-cn/data/hdf5_chinese"
        ),
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Optional single raw folder name (e.g. 20260824-143620) "
            "or output folder name."
        ),
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-01",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild output folders that already exist. "
            "Without this flag, existing t15... folders are skipped."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only show discovered sessions, S2/S4 assignments, "
            "MAT availability, and intended output paths."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Stop immediately on the first conversion error. "
            "Default behavior logs the error and continues to later sessions."
        ),
    )

    args = parser.parse_args()

    sessions = discover_matlab_sessions(
        args.source_root
    )

    if args.session is not None:
        sessions = [
            session
            for session in sessions
            if session.raw_name == args.session
            or session.output_name == args.session
        ]

    if args.dry_run:
        if not sessions:
            print(
                f"No convertible sessions found under "
                f"{args.source_root}"
            )
            return

        print(
            f"source_root = {args.source_root}"
        )
        print(
            f"output_root = {args.output_root}"
        )

        for session in sessions:
            mats = ", ".join(
                f"data_{split}.mat"
                for split in EXPECTED_SPLITS
                if split in session.mat_paths
            )
            output_dir = (
                args.output_root
                / session.output_name
            )
            status = (
                "exists -> skip unless --overwrite"
                if output_dir.exists()
                else "new"
            )

            print(
                f"{session.raw_name} -> "
                f"{session.session_code} -> "
                f"{session.output_name} | "
                f"MAT=[{mats}] | {status}"
            )
        return

    build_all_sessions(
        source_root=args.source_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        subject=args.subject,
        only_session=args.session,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
