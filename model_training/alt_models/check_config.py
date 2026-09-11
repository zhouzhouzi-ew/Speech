"""Pre-flight check for a training config's `sessions:` block.

Exists because a wrong `sessions:` list fails *silently*. Nothing scans
`dataset_dir` -- `rnn_trainer.py:163` builds `<dataset_dir>/<session>/data_train.hdf5`
straight from the list -- so a session you forgot to add, a typo'd directory name,
or a `dataset_probability_val:` of the wrong length all produce a run that starts
fine, trains fine, and reports a PER number for the wrong dataset. There is no
error to read afterwards.

Run this before every multi-day run:

    cd Speech/model_training
    python alt_models/check_config.py rnn_args.yaml
    python alt_models/check_config.py alt_models/rnn_args_diphone_alldays.yaml

Exits non-zero if anything is wrong, so it chains:

    python alt_models/check_config.py rnn_args.yaml && python train_model.py rnn_args.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

# `<something>.<YYYY>.<MM>.<DD>.<HH-MM-SS>_<suffix>`. Used only to flag a session
# list that is not in chronological order, since sessions[i] IS day index i.
_DATE_RE = re.compile(r"\.(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})-(\d{2})-(\d{2})")


def _date_key(session: str):
    m = _DATE_RE.search(session)
    return tuple(m.groups()) if m else None


def check(config_path: Path) -> int:
    problems = []
    warnings = []

    if not config_path.exists():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2

    cfg = OmegaConf.load(config_path)
    dataset = cfg.get("dataset")
    if dataset is None:
        print(f"{config_path}: no `dataset:` block", file=sys.stderr)
        return 2

    dataset_dir = Path(str(dataset.get("dataset_dir", "")))
    # The config is normally run from model_training/, so a relative dataset_dir is
    # relative to the config's parent's parent -- but be forgiving and try both.
    candidates = [dataset_dir]
    candidates.append((config_path.parent / dataset_dir).resolve())
    resolved = next((c for c in candidates if c.is_dir()), None)

    sessions = list(dataset.get("sessions") or [])
    probs = list(dataset.get("dataset_probability_val") or [])

    print(f"config:    {config_path}")
    print(f"dataset:   {dataset_dir}  (resolved: {resolved if resolved else 'NOT FOUND'})")
    print()

    if resolved is None:
        problems.append(f"dataset_dir does not exist: {dataset_dir}")
    if not sessions:
        problems.append("`sessions` is empty -- nothing would be trained")

    if len(probs) != len(sessions):
        problems.append(
            f"`dataset_probability_val` has {len(probs)} entries but `sessions` has "
            f"{len(sessions)}; they must be equal length and in the same order"
        )

    print(f"{len(sessions)} session(s) -- sessions[i] IS day index i:")
    for i, session in enumerate(sessions):
        flag = ""
        if resolved is not None:
            session_dir = resolved / session
            if not session_dir.is_dir():
                problems.append(f"[{i}] {session}: directory does not exist")
                flag = "  <-- MISSING DIR"
            else:
                missing = [
                    name
                    for name in ("data_train.hdf5", "data_val.hdf5")
                    if not (session_dir / name).exists()
                ]
                if missing:
                    problems.append(f"[{i}] {session}: missing {', '.join(missing)}")
                    flag = f"  <-- MISSING {', '.join(missing)}"
                else:
                    flag = "  ok"
        val = probs[i] if i < len(probs) else "?"
        print(f"  [{i}] {session}")
        print(f"       dataset_probability_val={val}{flag}")

    if len(sessions) > 1:
        keys = [_date_key(s) for s in sessions]
        known = [k for k in keys if k is not None]
        if len(known) == len(keys) and known != sorted(known):
            warnings.append(
                "sessions are not in chronological order. Day indices are assigned by "
                "list position, so this trains day layers against the wrong days. "
                "Fix the order unless you meant it."
            )
        if any(p == 0 for p in probs):
            warnings.append(
                "dataset_probability_val has a 0: that day is excluded from the "
                "validation loop entirely (rnn_trainer.validation skips it)."
            )

    print()
    for w in warnings:
        print(f"WARNING: {w}")
    for p in problems:
        print(f"PROBLEM: {p}")

    if problems:
        print(f"\n{len(problems)} problem(s). Do not start training with this config.")
        return 1

    if len(sessions) == 1:
        print(
            "OK -- but only 1 day. Training will produce an n_days=1 model, and "
            "evaluation can only ever cover that one day (the session list is frozen "
            "into the checkpoint's args.yaml)."
        )
    else:
        print(f"OK -- {len(sessions)} days.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="path to a training config yaml")
    args = parser.parse_args()
    return check(Path(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
