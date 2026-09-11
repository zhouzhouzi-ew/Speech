"""Tests for the MATLAB session installer.

The installer's job is to make three strings agree -- the directory name, the
config entry, and the per-trial `session` attribute -- and to write the
`metadata.json` MATLAB never emits. Two things are load-bearing here:

1. **A failed install must not cost you the session you had.** `install()`
   rmtree's `dest` unconditionally, so anything that can fail has to fail
   first. It used to resolve the vocabulary *after* that rmtree, which turned a
   wrong `--metadata-from` into a destroyed session plus a half-built
   replacement.

2. **A box with no `metadata.json` anywhere still has to work.** The 35-class
   vocabulary is a fixed property of the corpus, and `audit_session` re-encodes
   every label against exactly that constant, so falling back to it is safe --
   but it has to be announced, not silent.

    cd Speech/model_training
    python alt_models/tests/test_install_matlab_session.py
"""

from __future__ import annotations

import json
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

from install_matlab_session import (  # noqa: E402
    install,
    resolve_metadata,
    rewrite_session_attr,
)

_SESSION = "t15.2026.08.14.10-11-24_tc_sbp_512"
_REAL = _PARENT.parent / "data" / "hdf5_data_512" / _SESSION


def _labels():
    return {"labels": {"phoneme_to_id": {"<blank>": 0, "AE": 1, "<sil>": 2}}}


# ---------------------------------------------------------------------------
# resolve_metadata
# ---------------------------------------------------------------------------
def test_metadata_is_read_from_the_explicit_path():
    tmp = Path(tempfile.mkdtemp(prefix="meta_"))
    try:
        d = tmp / "day1"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps(_labels()), encoding="utf-8")
        meta, found = resolve_metadata(tmp / "nothing", tmp, "x", d)
        assert meta is not None, found
        assert Path(found) == d / "metadata.json"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_metadata_can_be_borrowed_from_another_session_in_the_dataset_dir():
    """The vocabulary is corpus-fixed, so a sibling session is a valid source."""
    tmp = Path(tempfile.mkdtemp(prefix="meta_"))
    try:
        ds = tmp / "hdf5_data_512"
        other = ds / "some_other_day"
        other.mkdir(parents=True)
        (other / "metadata.json").write_text(json.dumps(_labels()), encoding="utf-8")
        meta, found = resolve_metadata(tmp / "src", ds, "the_new_one", None)
        assert meta is not None, found
        assert Path(found).parent == other
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_rejected_candidate_says_which_way_it_failed():
    """`not found` on its own cannot tell a wrong path from a wrong file.

    Those are different problems with different fixes, and the user only sees
    the error text.
    """
    tmp = Path(tempfile.mkdtemp(prefix="meta_"))
    try:
        ds = tmp / "hdf5_data_512"
        ds.mkdir()
        present_but_bare = ds / "no_vocab"
        present_but_bare.mkdir()
        (present_but_bare / "metadata.json").write_text(
            json.dumps({"subject": "sub-01"}), encoding="utf-8")

        reasons = resolve_metadata(tmp / "src", ds, "new", tmp / "ghost")
        meta, rejected = reasons
        assert meta is None
        joined = "\n".join(rejected)
        assert "no such file" in joined, "a missing path must be reported as missing"
        assert "has no labels.phoneme_to_id" in joined, \
            "a present-but-empty file must be reported differently"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# rewrite_session_attr
# ---------------------------------------------------------------------------
def test_rewrite_stamps_every_trial_and_reports_how_many_changed():
    tmp = Path(tempfile.mkdtemp(prefix="attr_"))
    try:
        path = tmp / "data_train.hdf5"
        with h5py.File(path, "w") as f:
            for i in range(3):
                f.create_group(f"t{i}").attrs["session"] = "old"
        changed, total = rewrite_session_attr(path, "new")
        assert (changed, total) == (3, 3)
        with h5py.File(path, "r") as f:
            assert {f[k].attrs["session"] for k in f} == {"new"}
        # Idempotent: nothing left to change the second time.
        assert rewrite_session_attr(path, "new") == (0, 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def test_install_writes_the_builtin_vocabulary_when_no_metadata_exists():
    """End-to-end on a real split: a renamed, metadata-less source is rescued.

    The source has no `metadata.json` and the dataset dir is empty, so the
    installer has nothing to borrow from -- it must fall back to the built-in
    map and still produce a session that passes the audit.
    """
    if not (_REAL / "data_val.hdf5").exists():
        print("    (skipped: no shipped dataset on disk)")
        return

    tmp = Path(tempfile.mkdtemp(prefix="install_"))
    try:
        src = tmp / "matlab_out"
        src.mkdir()
        # One real split is enough to exercise the label contract, and copying
        # only `data_val` keeps the test cheap.
        shutil.copy2(_REAL / "data_val.hdf5", src / "data_train.hdf5")
        stale = "t15.2026.08.14.15-05-37_tc_sbp_512_prevblockcal"
        rewrite_session_attr(src / "data_train.hdf5", stale)

        ds = tmp / "hdf5_data_512"
        ds.mkdir()
        name = "t15.2026.08.14.15-05-37_tc_sbp_512"
        out = install(src, ds, name, None, overwrite=False)

        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert meta["session"] == name
        vocab = meta["labels"]["phoneme_to_id"]
        assert vocab["<blank>"] == 0 and vocab["<sil>"] == 34 and len(vocab) == 35

        with h5py.File(out / "data_train.hdf5", "r") as f:
            stamped = {f[k].attrs["session"] for k in f.keys()}
        assert stamped == {name}, f"output still stamped {stamped}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_install_refuses_without_overwrite_rather_than_clobbering():
    tmp = Path(tempfile.mkdtemp(prefix="install_"))
    try:
        src = tmp / "src"
        src.mkdir()
        with h5py.File(src / "data_train.hdf5", "w") as f:
            f.create_group("t0").attrs["session"] = "s"
        ds = tmp / "hdf5_data_512"
        (ds / "s").mkdir(parents=True)
        (ds / "s" / "keep_me.txt").write_text("x", encoding="utf-8")
        try:
            install(src, ds, "s", None, overwrite=False)
        except SystemExit as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("expected a refusal")
        assert (ds / "s" / "keep_me.txt").exists(), "the existing session was touched"
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
