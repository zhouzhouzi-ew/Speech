from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tomllib
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from nltk.corpus import cmudict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
RAW_SUBJECT_ROOT = REPO_ROOT / "sub-01"
DEFAULT_EXCEL_CANDIDATES = sorted(list(PROJECT_ROOT.glob("*.xlsx")) + list(REPO_ROOT.glob("*.xlsx")))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "hdf5_data_final"
DEFAULT_B2TXT_CSV = PROJECT_ROOT / "data" / "t15_copyTaskData_description.csv"
DEFAULT_EXCLUDED_TRIALS_CSV = REPO_ROOT / "sub01_rest_viewer" / "rest_trials_params.csv"

EXPECTED_SENTENCE_TRIALS = 189
N_ELECTRODES = 256
BIN_SIZE_MS = 20
HISTORY_TRIALS = 20
STD_FLOOR = 0.05
Z_CLIP = 20.0
BLANK_IDX = 0


@dataclass(frozen=True)
class RawSession:
    date: str
    raw_session_name: str
    raw_session_path: Path
    mat_path: Path
    trial_count: int
    first_texts: Tuple[str, ...]
    kind: str
    raw_config_block_num: Optional[int]
    minutes_of_day: float


@dataclass(frozen=True)
class ExcludedTrial:
    date: str
    session: str
    session_key: str
    global_id: int
    trial_0_based: Optional[int]
    target: Optional[str]


def _select_sheet_name(sheet_names: Sequence[str], token: str) -> str:
    matches = [name for name in sheet_names if token in name]
    if not matches:
        raise ValueError(f"Could not find a sheet containing {token!r}")
    return matches[0]


def _normalize_compare(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _parse_raw_session_time(session_name: str) -> float:
    match = re.match(r"session-(\d{2})-(\d{2})-(\d{2})", session_name)
    if not match:
        return float("inf")
    hh, mm, ss = (int(match.group(i)) for i in range(1, 4))
    return hh * 60.0 + mm + ss / 60.0


def _session_time_key(session_name: str) -> str:
    match = re.search(r"session-\d{2}-\d{2}-\d{2}", str(session_name))
    if match:
        return match.group(0)
    return str(session_name)


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find any of columns {list(candidates)!r} in {list(df.columns)!r}")


def _load_excluded_trial_rows(csv_path: Optional[Path]) -> List[ExcludedTrial]:
    if csv_path is None:
        return []
    if not csv_path.exists():
        raise FileNotFoundError(f"Excluded-trials CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = _first_existing_column(df, ("date", "Date", "\u65e5\u671f"))
    session_col = _first_existing_column(df, ("session", "Session"))
    global_id_col = _first_existing_column(df, ("trial_id(params)", "global_id", "trial_id"))
    trial0_col = next((col for col in ("trial(0-based)", "trial_0_based") if col in df.columns), None)
    target_col = next((col for col in ("target", "Target", "\u76ee\u6807") if col in df.columns), None)

    excluded: List[ExcludedTrial] = []
    for _, row in df.iterrows():
        date = str(row[date_col]).strip()
        session = str(row[session_col]).strip()
        if not date or not session:
            continue
        global_id = int(row[global_id_col])
        trial_0_based = int(row[trial0_col]) if trial0_col is not None and pd.notna(row[trial0_col]) else None
        target = str(row[target_col]).strip() if target_col is not None and pd.notna(row[target_col]) else None
        excluded.append(
            ExcludedTrial(
                date=date,
                session=session,
                session_key=_session_time_key(session),
                global_id=global_id,
                trial_0_based=trial_0_based,
                target=target,
            )
        )
    return excluded


def _select_excluded_global_ids(
    excluded_trials: Sequence[ExcludedTrial],
    date: str,
    raw_session_name: str,
) -> Set[int]:
    session_key = _session_time_key(raw_session_name)
    return {
        int(row.global_id)
        for row in excluded_trials
        if row.date == date and row.session_key == session_key
    }


def _filter_sentence_rows_for_exclusions(
    sentence_rows: Sequence[Dict[str, object]],
    excluded_global_ids: Set[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kept_rows: List[Dict[str, object]] = []
    removed_rows: List[Dict[str, object]] = []
    for row in sentence_rows:
        target = removed_rows if int(row["gid"]) in excluded_global_ids else kept_rows
        target.append(dict(row))
    return kept_rows, removed_rows


def _load_config_block_num(config_path: Path) -> Optional[int]:
    try:
        with open(config_path, "rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    engine = data.get("engine", {})
    block_num = engine.get("block_num")
    if block_num is None:
        return None
    try:
        return int(block_num)
    except Exception:
        return None


def _decode_target_content_dataset(f: h5py.File) -> List[str]:
    target_content = f["target_content"]
    texts: List[str] = []
    for i in range(target_content.shape[0]):
        ref = target_content[i, 0]
        obj = f[ref]
        arr = np.asarray(obj[:]).reshape(-1)
        if arr.dtype.kind in {"u", "i"}:
            text = "".join(chr(int(x)) for x in arr if int(x) != 0)
        elif arr.dtype.kind == "S":
            text = b"".join(arr.tolist()).decode("utf-8", errors="ignore")
        elif arr.dtype.kind == "U":
            text = "".join(arr.tolist())
        else:
            text = str(arr.tolist())
        texts.append(text)
    return texts


def _load_sentence_rows(excel_path: Path) -> List[Dict[str, object]]:
    workbook = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl", header=None)
    sheet = workbook[_select_sheet_name(workbook.keys(), "04")]
    rows: List[Dict[str, object]] = []
    for _, row in sheet.iterrows():
        gid, block, trial, text = row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3]
        if pd.notna(gid) and pd.notna(block) and pd.notna(text) and isinstance(gid, (int, float, np.integer, np.floating)):
            rows.append(
                {
                    "gid": int(gid),
                    "block": int(block),
                    "trial": int(trial) if pd.notna(trial) else None,
                    "text": str(text).strip(),
                }
            )
    rows.sort(key=lambda item: item["gid"])
    if len(rows) != EXPECTED_SENTENCE_TRIALS:
        raise ValueError(f"Expected {EXPECTED_SENTENCE_TRIALS} sentence rows, found {len(rows)}")
    return rows


def _load_vocab_rows(excel_path: Path) -> List[str]:
    workbook = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl", header=None)
    sheet = workbook[_select_sheet_name(workbook.keys(), "03")]
    vocab: List[str] = []
    for _, row in sheet.iterrows():
        seq, word = row.iloc[0], row.iloc[2]
        if pd.notna(seq) and pd.notna(word) and isinstance(seq, (int, float, np.integer, np.floating)):
            vocab.append(str(word).strip().lower())
    return list(dict.fromkeys(vocab))


def _build_phoneme_table(vocab: Sequence[str]) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[int, str]]:
    cmu = cmudict.dict()
    word_phonemes: Dict[str, List[str]] = {}
    for word in vocab:
        if word not in cmu:
            raise ValueError(f"Word {word!r} not in CMUdict")
        word_phonemes[word] = [re.sub(r"[0-9]", "", str(phoneme)) for phoneme in cmu[word][0]]

    phoneme_to_id: Dict[str, int] = {"<blank>": BLANK_IDX}
    id_to_phoneme: Dict[int, str] = {BLANK_IDX: "<blank>"}
    for word in vocab:
        for phoneme in word_phonemes[word]:
            if phoneme not in phoneme_to_id:
                phoneme_to_id[phoneme] = len(phoneme_to_id)
                id_to_phoneme[phoneme_to_id[phoneme]] = phoneme
    phoneme_to_id["<sil>"] = len(phoneme_to_id)
    id_to_phoneme[phoneme_to_id["<sil>"]] = "<sil>"
    return word_phonemes, phoneme_to_id, id_to_phoneme


def _sentence_to_ids(text: str, word_phonemes: Dict[str, List[str]], phoneme_to_id: Dict[str, int]) -> List[int]:
    words = re.findall(r"[a-z']+", _normalize_compare(text).lower())
    sil = phoneme_to_id["<sil>"]
    ids: List[int] = [sil]
    for idx, word in enumerate(words):
        if idx > 0:
            ids.append(sil)
        if word not in word_phonemes:
            raise KeyError(f"Word {word!r} not found in 50-word vocabulary")
        ids.extend(phoneme_to_id[phoneme] for phoneme in word_phonemes[word])
    ids.append(sil)
    return ids


def _ascii_transcription(text: str) -> np.ndarray:
    normalized = unicodedata.normalize("NFKC", text)
    encoded = normalized.encode("ascii", errors="ignore")
    return np.frombuffer(encoded, dtype=np.uint8)


def _build_trial_bounds(target_id: np.ndarray) -> Dict[int, Tuple[int, int]]:
    tid = np.asarray(target_id).reshape(-1)
    changes = np.flatnonzero(np.diff(tid)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [tid.size]))
    return {int(tid[starts[i]]): (int(starts[i]), int(ends[i])) for i in range(len(starts))}


def _build_read_windows(
    state_bin: np.ndarray,
    trial_bounds: Dict[int, Tuple[int, int]],
    gids: Iterable[int],
) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    for gid in gids:
        start, end = trial_bounds[gid]
        seg = state_bin[start:end]
        read_bins = np.flatnonzero(seg == 2)
        if len(read_bins) == 0:
            raise ValueError(f"trial {gid}: no state_bin==2 region in [{start}, {end})")
        out[gid] = (start + int(read_bins[0]), start + int(read_bins[-1]) + 1)
    return out


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
        s = arr.sum(axis=0)
        ss = np.square(arr).sum(axis=0)
        n = int(arr.shape[0])
        self.entries.append((s, ss, n))
        self.total_sum += s
        self.total_sumsq += ss
        self.total_count += n
        while len(self.entries) > self.max_trials:
            old_s, old_ss, old_n = self.entries.popleft()
            self.total_sum -= old_s
            self.total_sumsq -= old_ss
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
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        std = np.maximum(std, STD_FLOOR)
        return mean.astype(np.float32), std.astype(np.float32)


def _discover_raw_sessions(raw_root: Path, selected_dates: Optional[Sequence[str]]) -> List[RawSession]:
    if selected_dates is None:
        date_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", p.name)])
    else:
        date_dirs = [raw_root / date for date in selected_dates]

    sessions: List[RawSession] = []
    for date_dir in date_dirs:
        if not date_dir.exists():
            raise FileNotFoundError(f"Missing raw date directory: {date_dir}")
        for session_dir in sorted([p for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("session-")]):
            mat_path = session_dir / "EnglishSpeech" / "trial_data.mat"
            if not mat_path.exists():
                continue
            with h5py.File(mat_path, "r") as f:
                texts = _decode_target_content_dataset(f)
                trial_count = len(texts)
            kind = "ignored"
            if trial_count == EXPECTED_SENTENCE_TRIALS and not any(_contains_cjk(text) for text in texts[:5]):
                kind = "sentence"
            elif not any(_contains_cjk(text) for text in texts[:5]):
                kind = "diagnostic"
            config_path = session_dir / "EnglishSpeech" / "config.toml"
            sessions.append(
                RawSession(
                    date=date_dir.name,
                    raw_session_name=session_dir.name,
                    raw_session_path=session_dir,
                    mat_path=mat_path,
                    trial_count=trial_count,
                    first_texts=tuple(texts[:5]),
                    kind=kind,
                    raw_config_block_num=_load_config_block_num(config_path),
                    minutes_of_day=_parse_raw_session_time(session_dir.name),
                )
            )
    return sessions


def _pair_diagnostic_sessions(sessions: Sequence[RawSession]) -> Dict[str, Optional[RawSession]]:
    by_date: Dict[str, List[RawSession]] = {}
    for session in sessions:
        by_date.setdefault(session.date, []).append(session)

    paired: Dict[str, Optional[RawSession]] = {}
    for date, day_sessions in by_date.items():
        diagnostic_candidates = [s for s in day_sessions if s.kind == "diagnostic"]
        sentence_sessions = [s for s in day_sessions if s.kind == "sentence"]
        for sentence_session in sentence_sessions:
            if not diagnostic_candidates:
                paired[sentence_session.raw_session_name] = None
                continue
            diag = min(
                diagnostic_candidates,
                key=lambda candidate: abs(candidate.minutes_of_day - sentence_session.minutes_of_day),
            )
            paired[sentence_session.raw_session_name] = diag
    return paired


def _build_block_description_rows(date: str, sentence_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    blocks = {}
    for row in sentence_rows:
        block = int(row["block"])
        blocks.setdefault(block, 0)
        blocks[block] += 1
    rows: List[Dict[str, object]] = []
    for block in sorted(blocks):
        rows.append(
            {
                "Date": date,
                "Post-implant day": "",
                "Block number": block,
                "Number of sentences": blocks[block],
                "Corpus": "50-Word",
                "Split": "NA",
            }
        )
    return rows


def _build_sentence_lookup(sentence_rows: Sequence[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    lookup: Dict[int, Dict[str, object]] = {}
    for row in sentence_rows:
        lookup[int(row["gid"])] = dict(row)
    return lookup


def _choose_output_session_name(date: str, raw_session_name: str) -> str:
    match = re.match(r"session-(\d{2})-(\d{2})-(\d{2})", raw_session_name)
    if match:
        time_part = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return f"t15.{date.replace('-', '.')}.{time_part}_electrode"
    return f"t15.{date.replace('-', '.')}.{raw_session_name.replace('session-', '')}_electrode"


def _write_session_b2txt_description(output_dir: Path, date: str, sentence_rows: Sequence[Dict[str, object]]) -> None:
    rows = _build_block_description_rows(date, sentence_rows)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "b2txt_description.csv", index=False)


def _write_session_metadata(
    output_dir: Path,
    *,
    subject: str,
    date: str,
    raw_session_name: str,
    output_session_name: str,
    paired_diagnostic_session: Optional[RawSession],
    sentence_rows: Sequence[Dict[str, object]],
    n_trials_before_exclusion: int,
    excluded_sentence_rows: Sequence[Dict[str, object]],
    exclusion_source: Optional[Path],
    split_counts: Dict[str, int],
    split_seed: int,
    val_fraction: float,
    test_fraction: float,
    n_feat: int,
    dead_electrodes: Sequence[int],
    phoneme_to_id: Dict[str, int],
    id_to_phoneme: Dict[int, str],
    roll_stats: Dict[str, object],
) -> Dict[str, object]:
    metadata = {
        "subject": subject,
        "date": date,
        "raw_session": raw_session_name,
        "session": output_session_name,
        "paired_diagnostic_session": paired_diagnostic_session.raw_session_name if paired_diagnostic_session else None,
        "paired_diagnostic_block_num": paired_diagnostic_session.raw_config_block_num if paired_diagnostic_session else None,
        "n_trials": len(sentence_rows),
        "n_trials_before_exclusion": int(n_trials_before_exclusion),
        "n_excluded_trials": len(excluded_sentence_rows),
        "excluded_trials_csv": str(exclusion_source) if exclusion_source is not None else None,
        "excluded_trials": [
            {
                "global_id": int(row["gid"]),
                "block": int(row["block"]),
                "trial": int(row["trial"]) if row["trial"] is not None else None,
                "sentence_label": str(row["text"]),
            }
            for row in excluded_sentence_rows
        ],
        "n_train": split_counts.get("train", 0),
        "n_val": split_counts.get("val", 0),
        "n_test": split_counts.get("test", 0),
        "split": {
            "seed": split_seed,
            "train_fraction": 1.0 - val_fraction - test_fraction,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "mode": "random_per_session",
        },
        "features": {
            "mode": "electrode",
            "n_features": n_feat,
            "n_electrodes": N_ELECTRODES,
            "dead_electrodes": list(dead_electrodes),
            "bin_size_ms": BIN_SIZE_MS,
            "read_window": "state_bin==2 (authoritative, on spike_bin clock)",
            "normalization": "causal_prev20_read_epochs",
            "smoothing": "none_in_preprocessing_training_code_applies_gauss",
        },
        "labels": {
            "n_classes": len(phoneme_to_id),
            "blank_idx": BLANK_IDX,
            "sil_idx": phoneme_to_id["<sil>"],
            "phoneme_to_id": phoneme_to_id,
            "id_to_phoneme": {str(k): v for k, v in id_to_phoneme.items()},
            "scheme": "SIL at start + between words + end",
        },
        "roll_stats": roll_stats,
        "note": "English 50-word sentence sessions only; Chinese sessions are skipped.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return metadata


def _ensure_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                import shutil

                shutil.rmtree(child)
    path.mkdir(parents=True, exist_ok=True)


def _process_session(
    raw_session: RawSession,
    *,
    subject: str,
    sentence_rows: Sequence[Dict[str, object]],
    word_phonemes: Dict[str, List[str]],
    phoneme_to_id: Dict[str, int],
    id_to_phoneme: Dict[int, str],
    paired_diagnostic_session: Optional[RawSession],
    output_root: Path,
    excluded_global_ids: Set[int],
    exclusion_source: Optional[Path],
    split_seed: int,
    val_fraction: float,
    test_fraction: float,
    overwrite: bool,
    session_index: int,
) -> Dict[str, object]:
    output_session_name = _choose_output_session_name(raw_session.date, raw_session.raw_session_name)
    output_dir = output_root / output_session_name
    _ensure_dir(output_dir, overwrite=overwrite)

    kept_sentence_rows, excluded_sentence_rows = _filter_sentence_rows_for_exclusions(sentence_rows, excluded_global_ids)
    if not kept_sentence_rows:
        raise ValueError(
            f"All sentence trials were excluded for {raw_session.raw_session_name}; "
            f"check {exclusion_source}."
        )
    sentence_lookup = _build_sentence_lookup(kept_sentence_rows)

    with h5py.File(raw_session.mat_path, "r") as raw:
        acu = raw["array_channel_unit"][:]
        neuron_mask = raw["neuron_mask"][:].reshape(-1)
        spike_bin = raw["spike_bin"]
        state_bin = raw["state_bin"][:].reshape(-1)
        target_id = raw["target_id"][:].reshape(-1)
        raw_texts = _decode_target_content_dataset(raw)

        valid_cols = np.flatnonzero(neuron_mask > 0)
        n_units = len(valid_cols)
        if n_units == 0:
            raise ValueError(f"No valid sorted units found for {raw_session.raw_session_name}")

        electrodes = acu[3, valid_cols].astype(int) - 1
        if np.any((electrodes < 0) | (electrodes >= N_ELECTRODES)):
            raise ValueError(f"Found invalid electrode ids in {raw_session.raw_session_name}")

        membership = np.zeros((n_units, N_ELECTRODES), dtype=np.float32)
        membership[np.arange(n_units), electrodes] = 1.0
        dead_electrodes = [int(e + 1) for e in range(N_ELECTRODES) if not np.any(membership[:, e])]

        read_windows = _build_read_windows(state_bin, _build_trial_bounds(target_id), sentence_lookup.keys())

        split_rng = np.random.default_rng(split_seed + session_index)
        trial_order = np.array([int(row["gid"]) for row in kept_sentence_rows], dtype=int)
        split_rng.shuffle(trial_order)
        n_total = len(kept_sentence_rows)
        n_test = 0 if test_fraction <= 0 else max(1, int(math.floor(n_total * test_fraction)))
        n_val = 0 if val_fraction <= 0 else max(1, int(math.floor(n_total * val_fraction)))
        if n_test + n_val >= n_total:
            raise ValueError(
                f"Invalid split fractions for {raw_session.raw_session_name}: "
                f"val={val_fraction}, test={test_fraction}"
            )
        test_ids = set(int(x) for x in trial_order[:n_test])
        val_ids = set(int(x) for x in trial_order[n_test : n_test + n_val])
        train_ids = set(int(x) for x in trial_order[n_test + n_val :])
        split_map = {gid: "test" for gid in test_ids}
        split_map.update({gid: "val" for gid in val_ids})
        split_map.update({gid: "train" for gid in train_ids})

        files = {
            "train": h5py.File(output_dir / "data_train.hdf5", "w"),
            "val": h5py.File(output_dir / "data_val.hdf5", "w"),
            "test": h5py.File(output_dir / "data_test.hdf5", "w"),
        }
        counters = {"train": 0, "val": 0, "test": 0}

        rolling = RollingChannelStats(N_ELECTRODES, HISTORY_TRIALS)
        read_bins_per_trial: List[int] = []
        ctc_margin_min = float("inf")
        max_abs_z = 0.0
        raw_first_trials = 0
        trial_rows: List[Dict[str, object]] = []

        excluded_found: Set[int] = set()

        for gid in sorted(sentence_lookup):
            row = sentence_lookup[gid]
            sentence = str(row["text"])
            expected_raw = _normalize_compare(raw_texts[gid - 1])
            if expected_raw and expected_raw != _normalize_compare(sentence):
                raise ValueError(
                    f"Sentence text mismatch for {raw_session.raw_session_name} gid {gid}: "
                    f"raw={expected_raw!r} excel={sentence!r}"
                )

            split = split_map[gid]
            start, end = read_windows[gid]
            raw_slice = np.asarray(spike_bin[start:end, :], dtype=np.float32)[:, valid_cols]
            n_raw = end - start
            n_bins = n_raw // BIN_SIZE_MS
            raw_slice = raw_slice[: n_bins * BIN_SIZE_MS]
            binned = raw_slice.reshape(n_bins, BIN_SIZE_MS, n_units).sum(axis=1)
            tc = binned @ membership

            if rolling.has_history():
                mean, std = rolling.mean_std()
                normed = (tc - mean) / std
                normed = np.clip(normed, -Z_CLIP, Z_CLIP)
                max_abs_z = max(max_abs_z, float(np.abs(normed).max()))
            else:
                normed = tc.copy()
                raw_first_trials += 1
            rolling.push(tc)
            read_bins_per_trial.append(int(n_bins))

            ids = _sentence_to_ids(sentence, word_phonemes, phoneme_to_id)
            margin = (n_bins - 14) / 4 + 1 - len(ids)
            ctc_margin_min = min(ctc_margin_min, margin)
            if margin < 3:
                raise ValueError(
                    f"trial {gid}: CTC margin {margin:.1f} < 3 (n_bins={n_bins}, seq={len(ids)})"
                )

            transcription = _ascii_transcription(sentence)
            grp = files[split].create_group(f"trial_{counters[split]:04d}")
            grp.create_dataset("input_features", data=normed.astype(np.float32))
            grp.create_dataset("seq_class_ids", data=np.asarray(ids, dtype=np.int32))
            grp.create_dataset(
                "transcription",
                data=np.pad(transcription, (0, max(0, 1)), constant_values=0) if transcription.size == 0 else transcription,
            )
            grp.attrs["subject"] = subject
            grp.attrs["date"] = raw_session.date
            grp.attrs["session"] = output_session_name
            grp.attrs["raw_session"] = raw_session.raw_session_name
            grp.attrs["paired_diagnostic_session"] = (
                paired_diagnostic_session.raw_session_name if paired_diagnostic_session else ""
            )
            grp.attrs["paired_diagnostic_block_num"] = (
                int(paired_diagnostic_session.raw_config_block_num)
                if paired_diagnostic_session and paired_diagnostic_session.raw_config_block_num is not None
                else -1
            )
            grp.attrs["block_num"] = int(row["block"])
            grp.attrs["trial_num"] = int(row["trial"]) if row["trial"] is not None else int(gid)
            grp.attrs["global_id"] = int(gid)
            grp.attrs["split"] = split
            grp.attrs["corpus"] = "50-Word"
            grp.attrs["sentence_label"] = sentence.encode("utf-8")
            grp.attrs["n_time_steps"] = int(n_bins)
            grp.attrs["seq_len"] = len(ids)
            grp.attrs["feature_type"] = "tc_electrode_zscore_prev20_statebin_read_stdfloor0.05_clip20"
            grp.attrs["bin_size_ms"] = BIN_SIZE_MS
            grp.attrs["raw_config_block_num"] = (
                int(raw_session.raw_config_block_num) if raw_session.raw_config_block_num is not None else -1
            )
            grp.attrs["raw_trial_count"] = raw_session.trial_count

            trial_rows.append(
                {
                    "subject": subject,
                    "date": raw_session.date,
                    "session": output_session_name,
                    "raw_session": raw_session.raw_session_name,
                    "paired_diagnostic_session": paired_diagnostic_session.raw_session_name
                    if paired_diagnostic_session
                    else None,
                    "paired_diagnostic_block_num": paired_diagnostic_session.raw_config_block_num
                    if paired_diagnostic_session
                    else None,
                    "global_id": gid,
                    "block": int(row["block"]),
                    "trial": int(row["trial"]) if row["trial"] is not None else int(gid),
                    "sentence_label": sentence,
                    "split": split,
                    "corpus": "50-Word",
                }
            )
            counters[split] += 1

        excluded_found = set(int(row["gid"]) for row in excluded_sentence_rows)
        missing_excluded = sorted(excluded_global_ids - excluded_found)
        if missing_excluded:
            raise ValueError(
                f"Excluded gids not found in {raw_session.raw_session_name}: {missing_excluded}"
            )

        for handle in files.values():
            handle.close()

    if excluded_sentence_rows:
        pd.DataFrame(excluded_sentence_rows).to_csv(output_dir / "excluded_trials.csv", index=False)

    pd.DataFrame(trial_rows).to_csv(output_dir / "trial_manifest.csv", index=False)
    _write_session_b2txt_description(output_dir, raw_session.date, kept_sentence_rows)

    metadata = _write_session_metadata(
        output_dir,
        subject=subject,
        date=raw_session.date,
        raw_session_name=raw_session.raw_session_name,
        output_session_name=output_session_name,
        paired_diagnostic_session=paired_diagnostic_session,
        sentence_rows=kept_sentence_rows,
        n_trials_before_exclusion=len(sentence_rows),
        excluded_sentence_rows=excluded_sentence_rows,
        exclusion_source=exclusion_source,
        split_counts=counters,
        split_seed=split_seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        n_feat=N_ELECTRODES,
        dead_electrodes=dead_electrodes,
        phoneme_to_id=phoneme_to_id,
        id_to_phoneme=id_to_phoneme,
        roll_stats={
            "read_bins_per_trial": {
                "min": int(np.min(read_bins_per_trial)),
                "mean": float(np.mean(read_bins_per_trial)),
                "max": int(np.max(read_bins_per_trial)),
            },
            "ctc_min_margin": float(ctc_margin_min),
            "z_raw_first_trials": raw_first_trials,
            "max_abs_z": float(max_abs_z),
        },
    )

    return {
        "subject": subject,
        "date": raw_session.date,
        "raw_session": raw_session.raw_session_name,
        "output_session": output_session_name,
        "paired_diagnostic_session": paired_diagnostic_session.raw_session_name if paired_diagnostic_session else None,
        "paired_diagnostic_block_num": paired_diagnostic_session.raw_config_block_num if paired_diagnostic_session else None,
        "n_trials": len(kept_sentence_rows),
        "n_trials_before_exclusion": len(sentence_rows),
        "n_excluded_trials": len(excluded_sentence_rows),
        "n_train": counters["train"],
        "n_val": counters["val"],
        "n_test": counters["test"],
        "output_dir": str(output_dir),
        "metadata_path": str(output_dir / "metadata.json"),
        "trial_rows": trial_rows,
        "excluded_sentence_rows": excluded_sentence_rows,
        "metadata": metadata,
    }


def _resolve_excel_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Excel workbook not found: {explicit}")
        return explicit

    candidates = [path for path in DEFAULT_EXCEL_CANDIDATES if path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")]
    if not candidates:
        candidates = [
            path
            for path in list(PROJECT_ROOT.glob("*.xlsx")) + list(REPO_ROOT.glob("*.xlsx"))
            if path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
        ]
    if not candidates:
        raise FileNotFoundError(f"No .xlsx workbook found in {PROJECT_ROOT}")
    candidates.sort(key=lambda p: (0 if "50" in p.name else 1, p.name))
    return candidates[0]


def _resolve_default_output_root(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    return DEFAULT_OUTPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sub-01 English 50-word electrode HDF5 sessions.")
    parser.add_argument("--subject", default="sub-01", help="Subject folder name under the repository root.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_SUBJECT_ROOT,
        help="Root directory containing raw date folders.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        help="Optional list of dates to process (YYYY-MM-DD). If omitted, all available dates are used.",
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=None,
        help="Optional path to the 50-word Excel workbook. If omitted, the workbook is auto-discovered.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root for hdf5_data_final. Defaults to nejm-brain-to-text_self/data/hdf5_data_final.",
    )
    parser.add_argument(
        "--exclude-trials-csv",
        type=Path,
        default=DEFAULT_EXCLUDED_TRIALS_CSV if DEFAULT_EXCLUDED_TRIALS_CSV.exists() else None,
        help="Optional CSV of abnormal trials to exclude, matched by date + session time + trial_id(params).",
    )
    parser.add_argument(
        "--ignore-excluded-trials",
        action="store_true",
        help="Disable exclusion CSV handling even if the default abnormal-trial file exists.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for per-session train/val/test splitting.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction within each session.")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction within each session.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing session output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Discover sessions and print the plan without writing files.")
    args = parser.parse_args()

    excel_path = _resolve_excel_path(args.excel_path)
    output_root = _resolve_default_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    excluded_trials: List[ExcludedTrial] = []
    exclusion_source: Optional[Path] = None
    if not args.ignore_excluded_trials and args.exclude_trials_csv is not None:
        exclusion_source = args.exclude_trials_csv
        excluded_trials = _load_excluded_trial_rows(args.exclude_trials_csv)

    sentence_rows = _load_sentence_rows(excel_path)
    sentence_lookup = _build_sentence_lookup(sentence_rows)
    vocab = _load_vocab_rows(excel_path)
    word_phonemes, phoneme_to_id, id_to_phoneme = _build_phoneme_table(vocab)

    raw_sessions = _discover_raw_sessions(args.raw_root, args.dates)
    sentence_sessions = [session for session in raw_sessions if session.kind == "sentence"]
    if not sentence_sessions:
        raise SystemExit("No English 189-trial sentence sessions found for the requested dates.")
    paired_diagnostics = _pair_diagnostic_sessions(raw_sessions)

    if args.dry_run:
        print(f"Excel workbook: {excel_path}")
        print(f"Output root: {output_root}")
        print(f"Excluded trials CSV: {exclusion_source if exclusion_source is not None else 'none'}")
        for session in sentence_sessions:
            diag = paired_diagnostics.get(session.raw_session_name)
            excluded_for_session = _select_excluded_global_ids(excluded_trials, session.date, session.raw_session_name)
            print(
                f"{session.date} | {session.raw_session_name} -> "
                f"{_choose_output_session_name(session.date, session.raw_session_name)} | "
                f"diag={diag.raw_session_name if diag else None} | "
                f"block={session.raw_config_block_num} | trials={session.trial_count} | "
                f"excluded={len(excluded_for_session)}"
            )
        return

    manifest_rows: List[Dict[str, object]] = []
    for session_index, session in enumerate(sentence_sessions):
        if session.date not in (args.dates or [s.date for s in sentence_sessions]):
            continue
        excluded_for_session = _select_excluded_global_ids(excluded_trials, session.date, session.raw_session_name)
        result = _process_session(
            session,
            subject=args.subject,
            sentence_rows=sentence_rows,
            word_phonemes=word_phonemes,
            phoneme_to_id=phoneme_to_id,
            id_to_phoneme=id_to_phoneme,
            paired_diagnostic_session=paired_diagnostics.get(session.raw_session_name),
            output_root=output_root,
            excluded_global_ids=excluded_for_session,
            exclusion_source=exclusion_source,
            split_seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            overwrite=args.overwrite,
            session_index=session_index,
        )
        manifest_rows.append(
            {
                "subject": result["subject"],
                "date": result["date"],
                "raw_session": result["raw_session"],
                "paired_diagnostic_session": result["paired_diagnostic_session"],
                "paired_diagnostic_block_num": result["paired_diagnostic_block_num"],
                "output_session": result["output_session"],
                "n_trials": result["n_trials"],
                "n_trials_before_exclusion": result["n_trials_before_exclusion"],
                "n_excluded_trials": result["n_excluded_trials"],
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_test": result["n_test"],
                "output_dir": result["output_dir"],
                "metadata_path": result["metadata_path"],
            }
        )
        print(
            f"[ok] {session.raw_session_name} -> {result['output_session']} "
            f"(train={result['n_train']}, val={result['n_val']}, test={result['n_test']})"
        )

    manifest_csv = output_root / "sub01_english_session_manifest.csv"
    if manifest_csv.exists():
        existing_manifest = pd.read_csv(manifest_csv)
        combined_manifest = pd.concat([existing_manifest, pd.DataFrame(manifest_rows)], ignore_index=True)
        if "output_session" in combined_manifest.columns:
            combined_manifest = combined_manifest.drop_duplicates(subset=["output_session"], keep="last")
    else:
        combined_manifest = pd.DataFrame(manifest_rows)
    combined_manifest.to_csv(manifest_csv, index=False)

    manifest_json = output_root / "sub01_english_session_manifest.json"
    existing_json = {}
    if manifest_json.exists():
        existing_json = json.loads(manifest_json.read_text(encoding="utf-8"))
    existing_sessions = existing_json.get("sessions", [])
    session_map = {row.get("output_session"): row for row in existing_sessions if isinstance(row, dict)}
    for row in manifest_rows:
        session_map[row["output_session"]] = row
    combined_sessions = list(session_map.values())
    manifest_json.write_text(
        json.dumps(
            {
                "subject": args.subject,
                "selected_dates": sorted({row["date"] for row in combined_sessions if row.get("date")}),
                "seed": args.seed,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
                "excel_path": str(excel_path),
                "output_root": str(output_root),
                "excluded_trials_csv": str(exclusion_source) if exclusion_source is not None else None,
                "sessions": combined_sessions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_csv}")
    print(f"Wrote manifest: {manifest_json}")


if __name__ == "__main__":
    main()
