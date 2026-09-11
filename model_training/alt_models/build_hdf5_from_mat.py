#!/usr/bin/env python3
"""Corrected mat -> hdf5 builder for the English 512D pipeline.

Fixes two classes of defect in the ad-hoc builder that produced
`can't locate attribute: 'block_num'`:

1. MISSING ATTRIBUTES. `dataset.py` reads `n_time_steps`, `block_num` and
   `trial_num` off each trial group with no `in` guard, and
   `evaluate_model_helpers.py` additionally requires `session`. A builder that
   writes only `global_id` / `sentence_label` / `feature_type` fails on the
   first batch. Everything needed is already in `trial_manifest.csv`.

2. RE-DERIVED PHONEME VOCABULARY. Deriving `phoneme_to_id` per session from
   cmudict assigns ids in first-encounter order over alphabetically sorted
   words. That is not the ground-truth ordering: re-deriving from the *existing*
   session's own manifest remaps 28 of 35 classes. The ids stay in range 0..34,
   so nothing errors -- but class k means a different phoneme on each day, and a
   model trained across those days is supervised on contradictory labels. The
   vocabulary is FIXED for this corpus; it is hardcoded below.

The encoding *scheme* (silence at the start, between every word, and at the end;
CMUdict first pronunciation; stress digits stripped) was already correct and is
preserved.

    python alt_models/build_hdf5_from_mat.py \
        --source-root /path/to/self_mat/myday \
        --output-root ~/nejm-brain-to-text-en/data/hdf5_data_512 \
        --session-name t15.2026.09.01.10-00-00_tc_sbp_512

The session name you pass must match, character for character, both the output
directory name and the entry you add to `sessions:` in the training config --
`evaluate_model_extended.py` does `sessions.index(session)` on the per-trial
`session` attr.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from nltk.corpus import cmudict
except ImportError:  # pragma: no cover
    cmudict = None

N_FEATURES = 512

#: Ground-truth class ordering for the English 512D copy task, taken from the
#: existing session's `metadata.json` (`labels.phoneme_to_id`). This is a fixed
#: property of the corpus and must NOT be re-derived per session.
PHONEME_TO_ID = {
    "<blank>": 0,
    "AE": 1, "M": 2, "AA": 3, "R": 4, "B": 5, "D": 6, "IH": 7, "NG": 8,
    "K": 9, "L": 10, "IY": 11, "N": 12, "OW": 13, "S": 14, "ER": 15,
    "AH": 16, "F": 17, "T": 18, "P": 19, "Y": 20, "UW": 21, "EY": 22,
    "TH": 23, "G": 24, "Z": 25, "UH": 26, "AY": 27, "HH": 28, "V": 29,
    "EH": 30, "AW": 31, "DH": 32, "W": 33, "<sil>": 34,
}
SIL = PHONEME_TO_ID["<sil>"]
N_CLASSES = len(PHONEME_TO_ID)

_WORD_RE = re.compile(r"[a-z']+")
_STRESS_RE = re.compile(r"[0-9]")

_CMU_CACHE = None


def _cmu():
    """`cmudict.dict()` costs ~1 s; loading it per trial dominates a full pass."""
    global _CMU_CACHE
    if _CMU_CACHE is None:
        if cmudict is None:
            raise RuntimeError("nltk is required (pip install nltk; nltk.download('cmudict'))")
        _CMU_CACHE = cmudict.dict()
    return _CMU_CACHE


def ascii_text(text) -> np.ndarray:
    return np.frombuffer(str(text).encode("ascii", errors="ignore"), dtype=np.uint8)


def load_manifest(csv_path: Path) -> dict:
    with open(csv_path, encoding="utf-8-sig") as f:
        return {int(r["global_id"]): r for r in csv.DictReader(f)}


def sentence_to_ids(text: str, warn=print, strict: bool = False) -> list:
    """Silence-delimited phoneme ids, using the fixed ground-truth vocabulary.

    With `strict=True` a word that CMUdict cannot pronounce is a hard error
    instead of a warning. The builder runs strict because the failure it guards
    against is not "one odd word" -- feeding it non-English data makes EVERY
    word unpronounceable, at which point every label collapses to a lone
    `<sil>` and the session is silently worthless.
    """
    if cmudict is None:
        raise RuntimeError("nltk is required (pip install nltk; nltk.download('cmudict'))")
    cmu = _cmu()

    ids = [SIL]
    words = _WORD_RE.findall(str(text).lower())

    # `_WORD_RE` is ASCII-only, so text in another script matches nothing at all
    # and would sail through as a bare <sil> with no OOV ever reported. That is
    # the exact shape of the failure this guards: a Chinese session built with
    # the English pipeline yields one <sil> per trial, silently.
    if strict and not words and str(text).strip():
        raise ValueError(
            f"sentence {text!r} contains no ASCII words, so it cannot be encoded as "
            "English phonemes. Non-English sessions (e.g. Chinese) must not go through "
            "this builder."
        )

    for word in words:
        pron = cmu.get(word)
        if not pron:
            message = (
                f"{word!r} is not in CMUdict and cannot be encoded. If this happens "
                "for most words, the session is not English copy-task data."
            )
            if strict:
                raise ValueError(message)
            warn(f"  WARNING: {message} -- dropped from the label")
            continue
        for p in pron[0]:
            ids.append(PHONEME_TO_ID[_STRESS_RE.sub("", p)])
        ids.append(SIL)
    return ids


def read_feature(f: h5py.File, ref) -> np.ndarray:
    """Return a `(T, 512)` float32 array.

    MATLAB writes whatever orientation the producing code chose, so this
    normalises -- but refuses to guess when the array is square, because at
    T == 512 transposing is shape-preserving and would silently corrupt the
    trial instead of raising.
    """
    arr = np.asarray(f[ref])
    if arr.ndim != 2:
        raise RuntimeError(f"feature ndim error {arr.shape}")

    rows, cols = arr.shape
    if cols == N_FEATURES and rows != N_FEATURES:
        pass  # already (T, 512)
    elif rows == N_FEATURES and cols != N_FEATURES:
        arr = arr.T
    elif rows == N_FEATURES and cols == N_FEATURES:
        raise RuntimeError(
            f"ambiguous feature shape {arr.shape}: both dimensions are {N_FEATURES}, "
            "so transposing is shape-preserving and cannot be detected. Check how "
            "the MATLAB side writes input_features (expected (T, 512) or (512, T)) "
            "and declare the orientation explicitly for this trial."
        )
    else:
        raise RuntimeError(f"bad feature shape {arr.shape}, expected one dim == {N_FEATURES}")

    return np.ascontiguousarray(arr, dtype=np.float32)


def convert_session(source: Path, output_root: Path, session_name: str,
                    overwrite: bool = False) -> Path:
    source = Path(source)
    manifest_path = source / "trial_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"no trial_manifest.csv under {source}")

    manifest = load_manifest(manifest_path)
    out = Path(output_root) / session_name
    if out.exists() and not overwrite:
        raise SystemExit(
            f"{out} already exists. Pass --overwrite to replace it, or choose "
            "another --session-name."
        )
    out.mkdir(parents=True, exist_ok=True)

    vocab_rows = []
    for split in ("train", "val", "test"):
        mat_file = source / f"data_{split}.mat"
        if not mat_file.exists():
            print(f"skip {split}: no {mat_file.name}")
            continue

        print(f"processing {split}")
        with h5py.File(mat_file, "r") as mf, h5py.File(out / f"data_{split}.hdf5", "w") as hf:
            dataset = mf["dataset"]
            gids = np.asarray(dataset["global_id"]).reshape(-1)
            refs = np.asarray(dataset["input_features"]).reshape(-1)

            written = 0
            degenerate = 0
            for gid, ref in zip(gids, refs):
                gid = int(gid)
                row = manifest.get(gid)
                if row is None:
                    print(f"  WARNING: global_id {gid} not in the manifest -- skipped")
                    continue

                feat = read_feature(mf, ref)
                label = sentence_to_ids(row["sentence_label"], strict=True)
                if label == [SIL]:
                    degenerate += 1

                g = hf.create_group(f"trial_{written:04d}")
                g.create_dataset("input_features", data=feat)
                g.create_dataset("seq_class_ids", data=np.asarray(label, dtype=np.int32))
                g.create_dataset("transcription", data=ascii_text(row["sentence_label"]))

                # --- the attributes dataset.py and evaluate_model_helpers.py read ---
                # `n_time_steps`, `block_num`, `trial_num` are read with no `in`
                # guard in dataset.py; `session` likewise in the helpers.
                g.attrs["n_time_steps"] = np.int32(feat.shape[0])
                g.attrs["block_num"] = np.int32(row["block"])
                g.attrs["trial_num"] = np.int32(row["trial"])
                g.attrs["global_id"] = np.int32(gid)
                g.attrs["sentence_label"] = str(row["sentence_label"])
                g.attrs["session"] = session_name
                g.attrs["subject"] = str(row.get("subject", "sub-01"))
                g.attrs["date"] = str(row.get("date", ""))
                g.attrs["split"] = split
                # `corpus` is what the evaluator uses to pick the task vocabulary.
                # Without it the helper falls back to a (Date, Block number) lookup
                # in t15_copyTaskData_description.csv, which raises if the new day
                # is not in that file.
                g.attrs["corpus"] = str(row.get("corpus", "50-Word"))
                g.attrs["feature_type"] = "tc_sbp_512"
                g.attrs["n_input_features"] = np.int32(N_FEATURES)
                if row.get("raw_session"):
                    g.attrs["raw_session"] = str(row["raw_session"])
                if row.get("paired_diagnostic_session"):
                    g.attrs["paired_diagnostic_session"] = str(row["paired_diagnostic_session"])
                if row.get("paired_diagnostic_block_num"):
                    g.attrs["paired_diagnostic_block_num"] = np.int32(
                        row["paired_diagnostic_block_num"]
                    )

                vocab_rows.append(
                    {"global_id": gid, "split": split, "n_phonemes": len(label)}
                )
                written += 1

            print(f"  wrote {written} trials")
            if written and degenerate > written * 0.2:
                raise SystemExit(
                    f"\n{degenerate}/{written} {split} trials encoded to a bare <sil>. "
                    "The sentences in this source are not pronounceable English, so the "
                    "labels carry no phonetic content. This is what happens when a "
                    "non-English (e.g. Chinese) session is built with the English "
                    "pipeline. Aborting rather than writing a session that trains to "
                    "nothing and reports a plausible-looking PER."
                )

    (out / "metadata.json").write_text(
        json.dumps(
            {
                "subject": "sub-01",
                "session": session_name,
                "n_features": N_FEATURES,
                "features": "TC256+SBP256",
                "labels": {
                    "n_classes": N_CLASSES,
                    "blank_idx": 0,
                    "sil_idx": SIL,
                    "phoneme_to_id": PHONEME_TO_ID,
                    "scheme": "SIL at start + between words + end",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nDONE: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True,
                        help="directory holding data_{train,val,test}.mat + trial_manifest.csv")
    parser.add_argument("--output-root", required=True,
                        help="parent dir that holds the t15.* session directories")
    parser.add_argument("--session-name", required=True,
                        help="e.g. t15.2026.09.01.10-00-00_tc_sbp_512 (must match the config's sessions entry)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_session(Path(args.source_root), Path(args.output_root),
                    args.session_name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
