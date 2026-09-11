"""Tests for the day-list generator's duplicate guards.

`sessions[i]` IS the day index -- it selects the model's day-specific input
layer -- so a duplicate directory is not a cosmetic problem. Two directories
holding one recording produce a config with more day layers than there are days,
and the model trains twice on the same 189 sentences. Nothing about the
resulting checkpoint looks wrong.

The interesting case is telling a duplicate apart from a real second day. Both
start at `global_id` 0 (the id is per-recording), so the id alone cannot do it;
the pair (global_id, n_time_steps) can, because a copy has identical frame
counts and a different recording does not.

    cd Speech/model_training
    python alt_models/tests/test_make_all_day_config.py
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ALT = _HERE.parent
_PARENT = _ALT.parent
for _p in (str(_ALT), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py  # noqa: E402

from make_all_day_config import (  # noqa: E402
    check_for_duplicate_recordings,
    check_for_trim_duplicates,
    discover_sessions,
    recording_fingerprint,
)


def _session(root: Path, name: str, lengths: list) -> Path:
    """A minimal train split whose trials are `(global_id=i, n_time_steps=L)`."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / "data_train.hdf5", "w") as f:
        for i, n in enumerate(lengths):
            g = f.create_group(f"trial_{i:04d}")
            g.create_dataset("input_features", data=np.zeros((n, 4), np.float32))
            g.attrs["global_id"] = np.int32(i)
            g.attrs["n_time_steps"] = np.int32(n)
    return d


def _expect_refusal(fn, *args) -> str:
    """Run `fn` expecting a refusal, and return what it told the user.

    The guards explain themselves on stderr and then `raise SystemExit(1)`, so
    `str(exc)` is just `"1"` -- the message has to be read from stderr.
    """
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            fn(*args)
    except SystemExit as exc:
        assert exc.code not in (None, 0), "exited cleanly despite a duplicate"
        return captured.getvalue()
    raise AssertionError("expected a refusal, got a clean run")


# ---------------------------------------------------------------------------
def test_a_copy_of_a_session_is_refused_whatever_it_is_named():
    """The `_prevblockcal` suffix, a `-copy`, anything: same trials, refused."""
    tmp = Path(tempfile.mkdtemp(prefix="days_"))
    try:
        root = tmp / "hdf5_data_512"
        lengths = [100, 250, 80]
        real = "t15.2026.08.14.15-05-37_tc_sbp_512"
        _session(root, real, lengths)
        _session(root, real + "_prevblockcal", lengths)      # identical copy

        message = _expect_refusal(check_for_duplicate_recordings, root,
                                  discover_sessions(root))
        assert real in message and real + "_prevblockcal" in message
        assert "same global_id AND same n_time_steps" in message
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_two_real_days_are_not_flagged():
    """Both days start at global_id 0 -- that alone must not look like a copy."""
    tmp = Path(tempfile.mkdtemp(prefix="days_"))
    try:
        root = tmp / "hdf5_data_512"
        _session(root, "t15.2026.08.14.10-11-24_tc_sbp_512", [100, 250, 80])
        _session(root, "t15.2026.08.14.15-05-37_tc_sbp_512", [130, 240, 95])
        check_for_duplicate_recordings(root, discover_sessions(root))  # must not raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_session_without_attrs_is_not_accused():
    """An odd or truncated session degrades to 'cannot tell', not a false hit."""
    tmp = Path(tempfile.mkdtemp(prefix="days_"))
    try:
        root = tmp / "hdf5_data_512"
        _session(root, "day_a", [10, 20])
        d = root / "day_b"
        d.mkdir()
        with h5py.File(d / "data_train.hdf5", "w") as f:
            f.create_group("t0").create_dataset("input_features",
                                                data=np.zeros((5, 4), np.float32))
        assert recording_fingerprint(d) is None
        check_for_duplicate_recordings(root, discover_sessions(root))  # must not raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_trim_pair_is_still_caught_by_its_own_more_specific_message():
    tmp = Path(tempfile.mkdtemp(prefix="days_"))
    try:
        root = tmp / "hdf5_data_512"
        _session(root, "day_a", [10, 20])
        _session(root, "day_a_trim", [8, 17])
        message = _expect_refusal(check_for_trim_duplicates, root,
                                  discover_sessions(root))
        assert "trimmed session next to its untrimmed original" in message
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_discovery_ignores_directories_without_a_train_split():
    tmp = Path(tempfile.mkdtemp(prefix="days_"))
    try:
        root = tmp / "hdf5_data_512"
        _session(root, "has_data", [10])
        (root / "empty_dir").mkdir(parents=True)
        (root / "a_file.txt").write_text("x", encoding="utf-8")
        assert discover_sessions(root) == ["has_data"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
