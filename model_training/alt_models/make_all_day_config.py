"""Generate a diphone config that trains on every HDF5 day present on disk.

The `sessions:` list in `rnn_args_diphone.yaml` has to be written by hand and
has to stay in sync with `dataset_probability_val:` (one entry per day, in the
same order).  This script scans the dataset directory, finds every session that
has `data_train.hdf5`, and rewrites those two keys in place.

    cd Speech/model_training
    python alt_models/make_all_day_config.py \
        --dataset_dir ../data/hdf5_data_512 \
        --config alt_models/rnn_args_diphone.yaml \
        --output alt_models/rnn_args_diphone_alldays.yaml

Session directories are named `<something>.<YYYY>.<MM>.<DD>.<HH-MM-SS>_<suffix>`,
so a lexical sort is also a chronological sort -- which matters, because
`sessions[i]` is the *day index* the model uses for its day-specific input
layer.  Renaming or reordering the list after training makes the day layers
mean something else, so keep the generated file and train from it directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py  # noqa: F401  (imported for the availability check / clear error)
from omegaconf import OmegaConf


def discover_sessions(dataset_dir: Path) -> list:
    sessions = []
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and (child / "data_train.hdf5").exists():
            sessions.append(child.name)
    return sessions


def describe_session(session_dir: Path) -> str:
    """Best-effort one-line summary so the user can sanity-check the day list."""
    train = session_dir / "data_train.hdf5"
    try:
        with h5py.File(train, "r") as f:
            keys = list(f.keys())
            n_train = len(keys)
            n_val = n_test = 0
            for name in ("data_val.hdf5", "data_test.hdf5"):
                other = session_dir / name
                if other.exists():
                    with h5py.File(other, "r") as g:
                        if name.startswith("data_val"):
                            n_val = len(list(g.keys()))
                        else:
                            n_test = len(list(g.keys()))
        return f"train={n_train} val={n_val} test={n_test}"
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"<could not read: {exc}>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "Where to look for day directories. Defaults to the base config's own "
            "dataset.dataset_dir. When passed, the value is also written into the "
            "generated config -- otherwise the day list would be enumerated from "
            "one directory while the config (and check_config.py) pointed at another."
        ),
    )
    parser.add_argument(
        "--config", type=str, default="alt_models/rnn_args_diphone.yaml"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alt_models/rnn_args_diphone_alldays.yaml",
        help="Where to write the multi-day config. Use '-' to print only.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Override output_dir/checkpoint_dir in the generated config. "
            "Defaults to the base config's output_dir with the day count appended."
        ),
    )
    parser.add_argument(
        "--day_calibration",
        type=str,
        default=None,
        choices=("baseline", "hammer_scalpel"),
        help=(
            "Set model.day_calibration in the generated config. `hammer_scalpel` "
            "is the FiLM + learned-gate day layer; it is only meaningful with more "
            "than one day. Defaults to whatever the base config already says."
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # Enumerate days from --dataset_dir when given, otherwise from whatever the
    # base config already points at. Passing it explicitly also pins it into the
    # output, so the enumerated day list and the directory check_config.py later
    # validates are guaranteed to be the same one.
    if args.dataset_dir:
        cfg.dataset.dataset_dir = args.dataset_dir
    dataset_dir = Path(str(cfg.dataset.dataset_dir))
    if not dataset_dir.is_dir():
        print(f"dataset_dir not found: {dataset_dir}", file=sys.stderr)
        raise SystemExit(1)

    sessions = discover_sessions(dataset_dir)
    if not sessions:
        print(f"No session directories with data_train.hdf5 under {dataset_dir}", file=sys.stderr)
        raise SystemExit(1)

    cfg.dataset.sessions = sessions
    # 1 == "validate on this day" (see rnn_trainer.validation). Every day is worth
    # reporting on; the train/val trial split itself comes from test_percentage.
    cfg.dataset.dataset_probability_val = [1] * len(sessions)

    if args.day_calibration:
        cfg.model.day_calibration = args.day_calibration

    # A multi-day run must not write into the single-day run's output_dir, or the
    # two checkpoints overwrite each other and args.yaml stops describing the
    # weights sitting next to it.
    base_output_dir = str(cfg.output_dir)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    elif len(sessions) > 1 and not base_output_dir.endswith(f"_{len(sessions)}day"):
        cfg.output_dir = f"{base_output_dir}_{len(sessions)}day"
    cfg.checkpoint_dir = f"{cfg.output_dir}/checkpoint"

    print(f"Found {len(sessions)} day(s) under {dataset_dir}:")
    for idx, session in enumerate(sessions):
        print(f"  [{idx}] {session}  {describe_session(dataset_dir / session)}")
    print()

    if args.output == "-":
        print(OmegaConf.to_yaml(cfg))
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    print(f"Wrote {out_path}  (output_dir={cfg.output_dir})")


if __name__ == "__main__":
    main()
