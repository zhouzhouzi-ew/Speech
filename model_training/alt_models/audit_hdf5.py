"""Audit every session under a dataset dir against the contract the trainer needs.

Checks, per session and per split:

  * the files the pipeline reads exist at all (`data_{train,val,test}.hdf5`)
  * the attributes read *without* an `in` guard are present
    (`dataset.py:164,167,168` -> n_time_steps, block_num, trial_num;
     `evaluate_model_helpers.py:178,183,190,191` -> + session)
  * attribute *types* are the ones the numpy-2 code paths tolerate
  * `input_features` is `(T, 512)` float32 and `n_time_steps` agrees with it
  * `seq_class_ids` are in range and, crucially, **re-encode** from the trial's
    own `sentence_label` under the fixed vocabulary -- this is what catches a
    session built with a re-derived phoneme order, which is otherwise invisible
    because the ids stay in 0..34
  * every session carries a `metadata.json` with the label map
  * the per-trial `session` attr matches the directory name, since
    `evaluate_model_extended.py:199` does `sessions.index(session)`

    cd Speech/model_training
    python alt_models/audit_hdf5.py --dataset_dir ../data/hdf5_data_512

Exits non-zero if any session has a problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hdf5_from_mat import N_FEATURES, PHONEME_TO_ID, sentence_to_ids  # noqa: E402

# Read with no `in` guard anywhere in the pipeline.
REQUIRED_ATTRS = ("n_time_steps", "block_num", "trial_num", "session", "corpus")
SPLITS = ("train", "val", "test")


def _scalar(value):
    return int(np.asarray(value).reshape(-1)[0])


def audit_session(
    session_dir: Path,
    max_trials: int | None = None,
    n_features: int = N_FEATURES,
) -> list:
    problems = []
    name = session_dir.name

    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        problems.append(f"{name}: no metadata.json")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            labels = meta.get("labels", {})
            if "phoneme_to_id" not in labels:
                problems.append(f"{name}: metadata.json has no labels.phoneme_to_id")
            elif labels["phoneme_to_id"] != PHONEME_TO_ID:
                problems.append(
                    f"{name}: metadata.json phoneme_to_id differs from the ground-truth "
                    "vocabulary used for training"
                )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: metadata.json unreadable: {exc}")

    total = 0
    for split in SPLITS:
        path = session_dir / f"data_{split}.hdf5"
        if not path.exists():
            if split == "train":
                problems.append(f"{name}: missing data_train.hdf5")
            continue

        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            total += len(keys)
            if not keys:
                problems.append(f"{name}/{split}: no trials")
                continue

            checked = keys if max_trials is None else keys[:max_trials]
            label_bad = 0
            for key in checked:
                g = f[key]

                for attr in REQUIRED_ATTRS:
                    if attr not in g.attrs:
                        problems.append(f"{name}/{split}/{key}: missing attr {attr!r}")

                if "input_features" not in g:
                    problems.append(f"{name}/{split}/{key}: no input_features")
                    continue

                feats = g["input_features"]
                if feats.ndim != 2 or feats.shape[1] != n_features:
                    problems.append(
                        f"{name}/{split}/{key}: input_features shape {feats.shape}, "
                        f"expected (T, {n_features})"
                    )
                elif feats.dtype != np.float32:
                    problems.append(
                        f"{name}/{split}/{key}: input_features dtype {feats.dtype}, expected float32"
                    )

                if "n_time_steps" in g.attrs and feats.ndim == 2:
                    if _scalar(g.attrs["n_time_steps"]) != feats.shape[0]:
                        problems.append(
                            f"{name}/{split}/{key}: n_time_steps "
                            f"{_scalar(g.attrs['n_time_steps'])} != input_features rows {feats.shape[0]}"
                        )

                if g.attrs.get("session") is not None:
                    got = g.attrs["session"]
                    got = got.decode() if isinstance(got, bytes) else str(got)
                    if got != name:
                        problems.append(
                            f"{name}/{split}/{key}: session attr {got!r} != directory name "
                            f"{name!r} -- evaluate_model_extended does sessions.index(session)"
                        )

                if "seq_class_ids" not in g:
                    problems.append(f"{name}/{split}/{key}: no seq_class_ids")
                    continue

                ids = [int(x) for x in g["seq_class_ids"][:]]
                if not ids:
                    problems.append(f"{name}/{split}/{key}: empty seq_class_ids")
                    continue
                if min(ids) < 0 or max(ids) >= len(PHONEME_TO_ID):
                    problems.append(
                        f"{name}/{split}/{key}: seq_class_ids out of range "
                        f"[{min(ids)}, {max(ids)}] for {len(PHONEME_TO_ID)} classes"
                    )

                label = g.attrs.get("sentence_label")
                if label is None:
                    continue
                if isinstance(label, bytes):
                    label = label.decode()
                expected = sentence_to_ids(str(label), warn=lambda m: None)
                if expected != ids:
                    label_bad += 1

            if label_bad:
                problems.append(
                    f"{name}/{split}: {label_bad}/{len(checked)} trials have seq_class_ids "
                    "that do NOT re-encode from sentence_label under the fixed vocabulary. "
                    "The session was built with a different phoneme ordering and its "
                    "labels are incompatible with the other days."
                )

    print(f"  {name}: {total} trials across {sum((session_dir / f'data_{s}.hdf5').exists() for s in SPLITS)} split(s)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=str, default="../data/hdf5_data_512")
    parser.add_argument("--max_trials", type=int, default=None,
                        help="check only the first N trials per split (faster)")
    parser.add_argument(
        "--n_features",
        type=int,
        default=N_FEATURES,
        help=(
            "expected width of `input_features`. 512 is the TC+SBP contract this "
            "auditor was written for; the 2026-09 spike-only sessions have no NS6 "
            "behind them, so they are TC-only and must be audited with 256."
        ),
    )
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        print(f"dataset_dir not found: {root}", file=sys.stderr)
        return 2

    sessions = sorted(p for p in root.iterdir() if p.is_dir() and (p / "data_train.hdf5").exists())
    if not sessions:
        print(f"no sessions with data_train.hdf5 under {root}", file=sys.stderr)
        return 2

    print(f"Auditing {len(sessions)} session(s) under {root}\n")
    all_problems = []
    for session_dir in sessions:
        all_problems.extend(audit_session(session_dir, args.max_trials, args.n_features))

    print()
    if all_problems:
        for p in all_problems:
            print(f"PROBLEM: {p}")
        print(f"\n{len(all_problems)} problem(s).")
        return 1

    print(f"OK -- {len(sessions)} session(s) pass the structural + label contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
