"""Tests for the corrected mat -> hdf5 builder.

The important one is `test_encoder_reproduces_the_shipped_dataset`: it replays
the per-trial `sentence_label` through the builder's encoder and demands the
resulting `seq_class_ids` match the shipped HDF5 exactly, for every trial in
every split. That is the regression guard against the bug the ad-hoc builder
had -- a per-session re-derived vocabulary that reassigns 28 of 35 phoneme ids
while staying in range 0..34, so nothing raises and the corruption is invisible.

    cd Speech/model_training
    python alt_models/tests/test_build_hdf5_from_mat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ALT = _HERE.parent
_PARENT = _ALT.parent
for _p in (str(_ALT), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_hdf5_from_mat import (  # noqa: E402
    N_CLASSES,
    N_FEATURES,
    PHONEME_TO_ID,
    SIL,
    read_feature,
    sentence_to_ids,
)

_SESSION = "t15.2026.08.14.10-11-24_tc_sbp_512"
_DATASET = _PARENT.parent / "data" / "hdf5_data_512" / _SESSION


def _have_dataset() -> bool:
    return (_DATASET / "data_train.hdf5").exists()


# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------
def test_vocabulary_is_the_fixed_ground_truth():
    assert N_CLASSES == 35
    assert PHONEME_TO_ID["<blank>"] == 0
    assert SIL == 34
    assert sorted(PHONEME_TO_ID.values()) == list(range(35))
    # A re-derived vocabulary would put these elsewhere; the ground truth has
    # AE first and W last among the phonemes.
    assert PHONEME_TO_ID["AE"] == 1
    assert PHONEME_TO_ID["W"] == 33


def test_encoding_scheme_is_silence_delimited():
    # "My family is closer." -> sil, M AY, sil, F AE M AH L IY, sil, IH Z, sil, K L OW S ER, sil
    ids = sentence_to_ids("My family is closer.")
    assert ids == [34, 2, 27, 34, 17, 1, 2, 16, 10, 11, 34, 7, 25, 34, 9, 10, 13, 14, 15, 34]
    assert ids[0] == SIL and ids[-1] == SIL


def test_encoding_is_stable_when_the_day_changes():
    """The vocabulary must not depend on which sentences are in the session.

    This is the property the ad-hoc builder violated: its ids depended on the
    day's word set. Here the same sentence must yield the same ids regardless.
    """
    a = sentence_to_ids("My family is closer.")
    b = sentence_to_ids("Bring it closer.")
    assert a == sentence_to_ids("My family is closer.")
    # "closer" = K L OW S ER + closing sil, encoded identically in both sentences
    assert a[14:] == b[-6:]
    # sil + B R IH NG + sil + IH T + sil + K L OW S ER + sil
    assert b == [34, 5, 4, 7, 8, 34, 7, 18, 34, 9, 10, 13, 14, 15, 34]


def test_unknown_words_are_reported_not_silently_dropped():
    seen = []
    ids = sentence_to_ids("My zzzqqq family is closer.", warn=seen.append, strict=False)
    assert len(seen) == 1 and "zzzqqq" in seen[0]
    # the rest of the label survives
    assert ids == sentence_to_ids("My family is closer.")


def test_non_english_text_is_rejected_in_strict_mode():
    """The failure that produced a silently useless session.

    `_WORD_RE` is ASCII-only, so Chinese text matches zero words -- no OOV is
    ever reported and every label collapses to a bare <sil>. Strict mode must
    catch it on the empty word list, not on the per-word path.
    """
    for text in ("他没有药", "今天学校里人很多"):
        try:
            sentence_to_ids(text, strict=True)
        except ValueError as exc:
            assert "no ASCII words" in str(exc)
        else:
            raise AssertionError(f"{text!r} must be rejected in strict mode")

    # non-strict still degenerates silently -- that is exactly why the builder
    # runs strict, and why this is asserted rather than assumed
    assert sentence_to_ids("他没有药", warn=lambda m: None, strict=False) == [SIL]


def test_strict_mode_still_accepts_english():
    assert sentence_to_ids("My family is closer.", strict=True) == [
        34, 2, 27, 34, 17, 1, 2, 16, 10, 11, 34, 7, 25, 34, 9, 10, 13, 14, 15, 34,
    ]


# ---------------------------------------------------------------------------
# feature orientation
# ---------------------------------------------------------------------------
class _FakeRef:
    def __init__(self, arr):
        self.arr = arr


class _FakeFile(dict):
    def __getitem__(self, k):
        return self[k] if False else dict.__getitem__(self, k)


def _feature(arr):
    return read_feature(_FakeFile({"r": arr}), "r")


def test_orientation_handles_both_layouts():
    assert _feature(np.zeros((375, N_FEATURES), dtype=np.float32)).shape == (375, N_FEATURES)
    assert _feature(np.zeros((N_FEATURES, 375), dtype=np.float32)).shape == (375, N_FEATURES)


def test_square_feature_raises_instead_of_guessing():
    """(512, 512) transposes to the same shape, so guessing would corrupt silently."""
    try:
        _feature(np.zeros((N_FEATURES, N_FEATURES), dtype=np.float32))
    except RuntimeError as exc:
        assert "ambiguous" in str(exc)
        return
    raise AssertionError("a square feature array must raise, not be silently transposed")


def test_bad_shape_raises():
    try:
        _feature(np.zeros((300, 400), dtype=np.float32))
    except RuntimeError as exc:
        assert "bad feature shape" in str(exc)
        return
    raise AssertionError("expected a RuntimeError for a non-512-wide array")


# ---------------------------------------------------------------------------
# the regression guard
# ---------------------------------------------------------------------------
def test_encoder_reproduces_the_shipped_dataset():
    if not _have_dataset():
        print("    (skipped: no shipped dataset on disk)")
        return

    import h5py

    checked = 0
    for split in ("train", "val", "test"):
        path = _DATASET / f"data_{split}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                truth = list(g["seq_class_ids"][:])
                got = sentence_to_ids(g.attrs["sentence_label"], warn=lambda m: None)
                assert got == truth, (
                    f"{split}/{key} {g.attrs['sentence_label']!r}\n"
                    f"  truth={truth}\n  got  ={got}"
                )
                checked += 1
    assert checked > 0
    print(f"    verified {checked} real trials")


def test_shipped_dataset_satisfies_the_attribute_contract():
    """Pin exactly which attrs the trainer and evaluator read without a guard."""
    if not _have_dataset():
        print("    (skipped: no shipped dataset on disk)")
        return

    import h5py

    # dataset.py:164,167,168 and evaluate_model_helpers.py:178,183,190,191
    required = ("n_time_steps", "block_num", "trial_num", "session", "corpus")
    with h5py.File(_DATASET / "data_train.hdf5", "r") as f:
        for key in list(f.keys())[:5]:
            g = f[key]
            for attr in required:
                assert attr in g.attrs, f"{key} is missing {attr!r}"
            assert int(np.asarray(g.attrs["n_time_steps"]).reshape(-1)[0]) == g["input_features"].shape[0]


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
