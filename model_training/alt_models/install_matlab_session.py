#!/usr/bin/env python3
"""Install a MATLAB-produced session directory into the trainer's dataset dir.

The MATLAB 512D pipeline
(`run_20260814_144544_512_hdf5_prevblock_calibration.m`) writes its HDF5s into
a folder named after the **raw recording** --

    D:\\wwl\\data\\self_mat\\hdf5_data_512\\20260814-144544\\

-- while stamping every trial's `session` attribute with the **session** name:

    t15.2026.08.14.15-05-37_tc_sbp_512_prevblockcal

Those are two different strings and *both* are load-bearing:

  * `rnn_trainer.py:163` builds `<dataset_dir>/<config entry>/data_train.hdf5`,
    so the config entry must equal the DIRECTORY name;
  * `evaluate_model_extended.py:199` does `sessions.index(session)` where
    `session` is read from the per-trial ATTRIBUTE.

Three strings must therefore agree character-for-character: the directory name
under `dataset_dir`, the entry in the config's `sessions:` list, and the
per-trial `session` attribute. MATLAB only guarantees the third. This script
makes all three agree, and writes the `metadata.json` that MATLAB never emits
(without it `evaluate_model_extended.py:179` silently falls back to the
built-in phoneme order instead of the session's own).

The label side needs no attention: the MATLAB script copies `seq_class_ids`
and `transcription` straight out of the reference day's HDF5, so day 2 is
guaranteed to use the same 35-class vocabulary as day 1. That is the whole
reason the vulnerable path is worth avoiding.

Usage:

    cd Speech/model_training
    python alt_models/install_matlab_session.py \
        --source-dir /mnt/d/wwl/data/self_mat/hdf5_data_512/20260814-144544 \
        --dataset-dir ../data/hdf5_data_512 \
        --session-name t15.2026.08.14.15-05-37_tc_sbp_512

Omit `--session-name` to use the name MATLAB already stamped (keeps the
`_prevblockcal` suffix). Then add the same string to the config's `sessions:`
list and run `alt_models/check_config.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_hdf5 import audit_session  # noqa: E402
from build_hdf5_from_mat import PHONEME_TO_ID  # noqa: E402

SPLITS = ("train", "val", "test")
CARRY_OVER = (
    "trial_manifest.csv",
    "b2txt_description.csv",
    "feature_trial_summary.csv",
)


def _text(value):
    """MATLAB writes attrs as fixed-length byte strings; h5py may hand back either."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def read_session_attr(hdf5_path: Path):
    """The name MATLAB stamped on the trials, read from the first group."""
    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            value = f[key].attrs.get("session")
            if value is not None:
                return _text(value)
    return None


def rewrite_session_attr(hdf5_path: Path, session_name: str):
    """Force every trial's `session` attr to `session_name`.

    Returns `(changed, total)`. Nothing else in the file is touched -- in
    particular `n_time_steps` / `block_num` / `trial_num`, which `dataset.py`
    reads with no guard at all.
    """
    changed = 0
    with h5py.File(hdf5_path, "r+") as f:
        keys = list(f.keys())
        for key in keys:
            group = f[key]
            if _text(group.attrs.get("session")) != session_name:
                group.attrs["session"] = session_name
                changed += 1
    return changed, len(keys)


def resolve_metadata(source: Path, dataset_dir: Path, session_name: str,
                     metadata_from: Path | None):
    """Find a metadata.json carrying `labels.phoneme_to_id`.

    Order: the one named on the command line, then the source dir's own, then
    anything already installed under `dataset_dir` (including a same-named
    session being reinstalled). The vocabulary is a fixed property of the
    corpus, so borrowing it from another session is correct -- but borrowing is
    only safe because `audit_hdf5` re-encodes every label against this exact map
    afterwards.

    Returns `(meta, path)` on success, or `(None, [reason, ...])` -- the reasons
    are the whole point: "not found" without saying where it looked is what makes
    a wrong `--metadata-from` path indistinguishable from a file that is present
    but missing the key.
    """
    candidates = []
    if metadata_from is not None:
        candidates.append(Path(metadata_from) / "metadata.json"
                          if Path(metadata_from).is_dir() else Path(metadata_from))
    candidates.append(source / "metadata.json")
    dataset_dir = Path(dataset_dir)
    if dataset_dir.is_dir():
        candidates.extend(sorted(
            p / "metadata.json"
            for p in dataset_dir.iterdir()
            if p.is_dir()
        ))

    rejected = []
    for path in candidates:
        if not path.exists():
            rejected.append(f"{path}: no such file")
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{path}: unreadable ({type(exc).__name__}: {exc})")
            continue
        if (meta.get("labels") or {}).get("phoneme_to_id"):
            return meta, path
        rejected.append(f"{path}: has no labels.phoneme_to_id")
    return None, rejected


def install(source: Path, dataset_dir: Path, session_name: str | None,
            metadata_from: Path | None, overwrite: bool) -> Path:
    source = Path(source)
    dataset_dir = Path(dataset_dir)

    if not source.is_dir():
        raise SystemExit(f"source dir not found: {source}")
    if not (source / "data_train.hdf5").exists():
        raise SystemExit(
            f"{source} has no data_train.hdf5. This script installs the output of "
            "the MATLAB 512D pipeline; a folder holding data_*.mat is a different "
            "stage (use build_hdf5_from_mat.py for those)."
        )

    stamped = read_session_attr(source / "data_train.hdf5")
    if session_name is None:
        if stamped is None:
            raise SystemExit(
                "no `session` attribute in the source HDF5, so the name cannot be "
                "inferred -- pass --session-name"
            )
        session_name = stamped
    elif stamped is not None and stamped != session_name:
        print(f"renaming session: {stamped!r} -> {session_name!r}")

    dest = dataset_dir / session_name
    if dest.exists() and not overwrite:
        raise SystemExit(f"{dest} already exists. Pass --overwrite to replace it.")

    # Resolve the vocabulary BEFORE touching `dest`: the rmtree below is
    # unconditional, so anything that can fail has to fail first. It used to run
    # after, which meant a bad `--metadata-from` destroyed a working session and
    # left a half-built one in its place.
    meta, found = resolve_metadata(source, dataset_dir, session_name, metadata_from)
    if meta is None:
        # Nothing on this machine could supply the vocabulary. The 35-class map
        # is a fixed property of the corpus, and `audit_session` re-encodes every
        # label against exactly this constant at the end of the install, so
        # writing it here is not a guess -- but it does mean nothing on this box
        # could cross-check it, so say so rather than passing quietly.
        detail = "\n".join(f"      - {r}" for r in found) or "      (nothing to try)"
        print("  WARNING: no metadata.json found anywhere on this machine:\n"
              f"{detail}\n"
              "           Writing the built-in 35-class vocabulary instead. The "
              "audit below\n"
              "           still re-encodes every label against it.")
        meta = {"labels": {"phoneme_to_id": dict(PHONEME_TO_ID)}}
        found = "built-in PHONEME_TO_ID"

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    total = 0
    for split in SPLITS:
        src_file = source / f"data_{split}.hdf5"
        if not src_file.exists():
            print(f"  {split}: not present in source, skipped")
            continue
        dst_file = dest / src_file.name
        shutil.copy2(src_file, dst_file)
        changed, trials = rewrite_session_attr(dst_file, session_name)
        total += trials
        print(f"  {split}: {trials} trials, session attr rewritten on {changed}")

    if not (dest / "data_test.hdf5").exists():
        print("  WARNING: no data_test.hdf5 -- evaluation must use --eval_type val")

    for name in CARRY_OVER:
        if (source / name).exists():
            shutil.copy2(source / name, dest / name)
            print(f"  carried over {name}")

    meta["session"] = session_name
    (dest / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  metadata.json <- {found}")

    print(f"\nInstalled {total} trials at {dest}\n")
    problems = audit_session(dest)
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        raise SystemExit(f"\n{len(problems)} problem(s) -- do not train on this.")
    print("Audit passed: structural + label contract OK.")
    print(f"\nNext: add \"{session_name}\" to `sessions:` in the training config "
          "and run alt_models/check_config.py.")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True,
                        help="MATLAB output dir, e.g. .../self_mat/hdf5_data_512/20260814-144544")
    parser.add_argument("--dataset-dir", required=True,
                        help="parent dir that holds the t15.* session directories")
    parser.add_argument("--session-name", default=None,
                        help="name for the installed session; defaults to the one MATLAB stamped")
    parser.add_argument("--metadata-from", default=None,
                        help="session dir (or metadata.json) to borrow labels.phoneme_to_id from")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    install(Path(args.source_dir), Path(args.dataset_dir), args.session_name,
            Path(args.metadata_from) if args.metadata_from else None,
            args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
