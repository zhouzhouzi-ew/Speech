"""Tests for the silence trimmer.

Two things are load-bearing and both are pinned here:

1. **The frame <-> wall-clock mapping.** Everything downstream assumes feature
   row `k` is at `read_begin + k*20 ms`. The shipped day-1 session is the oracle:
   `test_bin_count_identity_holds_for_the_shipped_dataset` requires
   `n_time_steps == floor(read_duration_sec * 50)` for all 189 trials. If that
   ever stops holding, the mask is misaligned and the trimmer is cutting the
   wrong audio -- silently, since the output stays structurally valid.

2. **Trimming must not touch speech.** The tempting failure mode is a threshold
   that eats word onsets: the output still passes `audit_hdf5.py`, still trains,
   and just quietly performs worse. `test_..._removes_no_speech` demands that
   every frame the VAD called speech survives, on real data.

    cd Speech/model_training
    python alt_models/tests/test_trim_silence.py
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

from trim_silence import (  # noqa: E402
    FEATURE_HZ,
    frame_db,
    plan_keep,
    speech_mask,
)

_SESSION = "t15.2026.08.14.10-11-24_tc_sbp_512"
_DATASET = _PARENT.parent / "data" / "hdf5_data_512" / _SESSION
_AUDIO = (_PARENT.parent.parent / "sub-01" / "2026-08-14" / "session-10-11-24"
          / "EnglishSpeech" / "microphone_audio.wav")
_CSV = _AUDIO.with_name("data_2026-08-14_10-11-24.csv")

# 20 ms frames, so one bin == one frame in every test below.
HOP_MS = 20.0


def _mask(spec: str) -> np.ndarray:
    """`_'` = silence frame, `'#'` = speech frame."""
    return np.array([c == "#" for c in spec], dtype=bool)


# ---------------------------------------------------------------------------
# plan_keep
# ---------------------------------------------------------------------------
def test_edges_are_trimmed_to_the_margin():
    speech = _mask("____########____")           # 4 silent each side
    keep = plan_keep(speech, HOP_MS, max_silence_ms=1000, keep_silence_ms=200,
                     edge_keep_ms=40)
    # 40 ms = 2 frames of margin retained on each side
    assert keep.sum() == 8 + 2 + 2
    assert not keep[:2].any() and not keep[-2:].any()
    assert keep[2:14].all()


def test_short_internal_silences_are_left_alone():
    """A normal inter-word pause must survive -- it is what `<sil>` labels."""
    speech = _mask("##_##_##")                   # 1-frame gaps, well under 1000 ms
    keep = plan_keep(speech, HOP_MS, max_silence_ms=1000, keep_silence_ms=200,
                     edge_keep_ms=200)
    assert keep.all(), "gaps shorter than max_silence_ms must not be collapsed"


def test_a_long_internal_silence_is_collapsed_to_the_keep_length():
    speech = _mask("##" + "_" * 100 + "##")      # 2000 ms of silence
    keep = plan_keep(speech, HOP_MS, max_silence_ms=1000, keep_silence_ms=200,
                     edge_keep_ms=200)
    # 200 ms of the run survives (10 frames), the other 90 frames go
    assert keep.sum() == 4 + 10
    assert keep.nonzero()[0].tolist()[:2] == [0, 1]
    assert keep.nonzero()[0].tolist()[-2:] == [len(speech) - 2, len(speech) - 1]


def test_collapse_is_relative_not_absolute():
    """500 ms of silence is only collapsed when the threshold says so."""
    speech = _mask("##" + "_" * 25 + "##")       # 500 ms
    tight = plan_keep(speech, HOP_MS, max_silence_ms=300, keep_silence_ms=200,
                      edge_keep_ms=200)
    loose = plan_keep(speech, HOP_MS, max_silence_ms=1000, keep_silence_ms=200,
                      edge_keep_ms=200)
    assert tight.sum() == 4 + 10
    assert loose.sum() == len(speech)


def test_the_default_rule_only_touches_silences_over_the_threshold():
    """The instruction is "delete silence longer than 2 s", not "delete silence".

    So a 1.5 s pause anywhere in the trial -- including at the edges -- has to
    come through untouched, and only the 2.5 s one is cut.
    """
    # 20 ms per frame, so 1.5 s = 75 frames, 100 ms = 5, 2.5 s = 125.
    speech = _mask("_" * 75 + "#" * 4 + "_" * 5 + "#" * 4 + "_" * 125)
    keep = plan_keep(speech, HOP_MS, max_silence_ms=2000, keep_silence_ms=0,
                     edge_keep_ms=None)
    assert keep[:75].all(), "the 1.5 s lead is under the threshold -- it stays"
    assert not keep[88:].any(), "the 2.5 s tail is over the threshold -- it goes"
    assert keep[75:88].all(), "speech and the 100 ms gap both stay"


def test_no_speech_means_nothing_is_removed():
    """An all-silence trial must pass through, not be deleted."""
    keep = plan_keep(_mask("_" * 50), HOP_MS, max_silence_ms=1000,
                     keep_silence_ms=200, edge_keep_ms=200)
    assert keep.all()


def test_speech_frames_always_survive():
    """The invariant the whole script rests on.

    Note the direction: `keep` is *allowed* to retain silence (that is what the
    collapsed-run tail and any explicit `edge_keep_ms` margin are), so the claim
    is not `speech[keep].all()` -- it is that no speech frame is dropped.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        speech = rng.random(300) < 0.3
        if not speech.any():
            continue
        keep = plan_keep(speech, HOP_MS, max_silence_ms=2000, keep_silence_ms=0,
                         edge_keep_ms=None)
        assert speech[keep].sum() == speech.sum(), \
            "a frame the VAD called speech was dropped"


# ---------------------------------------------------------------------------
# speech_mask
# ---------------------------------------------------------------------------
def test_mask_separates_speech_from_room_tone():
    db = np.array([20.0] * 50 + [70.0] * 50)     # 50 dB of contrast
    mask, peak, floor = speech_mask(db, below_peak_db=22.0, above_floor_db=20.0)
    assert mask is not None
    assert mask[:50].sum() == 0 and mask[50:].all()
    assert peak > 60 and floor < 30


def test_mask_tracks_the_level_not_an_absolute_threshold():
    """The same shape at a much lower recording level must classify identically.

    Room tone on these sessions sits anywhere from 15 to 35 dB, so any fixed
    threshold either eats speech on quiet days or keeps noise on loud ones.
    """
    quiet = np.array([0.0] * 50 + [45.0] * 50)
    loud = np.array([25.0] * 50 + [75.0] * 50)
    for db in (quiet, loud):
        mask, _, _ = speech_mask(db, 22.0, 20.0)
        assert mask[:50].sum() == 0 and mask[50:].all()


def test_mask_refuses_a_trial_with_no_contrast():
    db = np.full(100, 30.0) + np.random.default_rng(1).normal(0, 1, 100)
    mask, peak, floor = speech_mask(db, 22.0, 20.0)
    assert mask is None, "uniform room tone must not be classified as speech"


# ---------------------------------------------------------------------------
# frame_db
# ---------------------------------------------------------------------------
def test_frame_db_is_exactly_n_bins_long_and_at_the_right_offset():
    fs = 44100
    hop = int(round(0.02 * fs))
    audio = np.zeros(fs * 10, dtype=np.float32)
    audio[fs * 2: fs * 3] = 1000.0              # loud from 2 s to 3 s
    n = 250                                      # 5 s
    db = frame_db(audio, fs, 0.0, n, hop)
    assert db.shape == (n,)
    loud = db > 20
    assert loud[100:150].all(), "frames 100-150 are 2.0-3.0 s, which is the loud part"
    assert not loud[:95].any() and not loud[155:].any()


def test_frame_db_pads_a_short_tail_instead_of_returning_fewer_bins():
    """A trial running past the end of the recording must still yield n bins."""
    fs = 1000
    hop = 20
    audio = np.ones(fs, dtype=np.float32)
    db = frame_db(audio, fs, 0.9, 20, hop)       # asks for 0.4 s of audio
    assert db.shape == (20,)


# ---------------------------------------------------------------------------
# the mapping oracle
# ---------------------------------------------------------------------------
def test_bin_count_identity_holds_for_the_shipped_dataset():
    """n_time_steps == floor(read_duration_sec * 50), on every real trial.

    This is the contract the mask depends on. It is checked against the shipped
    day-1 session because that is the only ground truth available for it.
    """
    if not (_DATASET / "data_train.hdf5").exists():
        print("    (skipped: no shipped dataset on disk)")
        return

    import h5py

    checked = 0
    for split in ("train", "val", "test"):
        with h5py.File(_DATASET / f"data_{split}.hdf5", "r") as f:
            for key in f.keys():
                g = f[key]
                rows = g["input_features"].shape[0]
                nts = int(np.asarray(g.attrs["n_time_steps"]).reshape(-1)[0])
                dur = float(np.asarray(g.attrs["read_duration_sec"]).reshape(-1)[0])
                assert nts == rows, f"{split}/{key}: n_time_steps {nts} != rows {rows}"
                assert int(np.floor(dur * FEATURE_HZ)) == rows, (
                    f"{split}/{key}: floor({dur}*50) != {rows} -- the frame<->audio "
                    "mapping has changed and the silence mask is now misaligned"
                )
                checked += 1
    assert checked > 0
    print(f"    verified {checked} real trials")


def test_real_trimming_removes_no_speech():
    """End-to-end on day 1: the loud frames all survive.

    The threshold has to be loose enough to keep word onsets, which are quiet
    relative to vowels. Checking the aggregate removal rate would not catch that;
    this checks the frames the VAD itself called speech.
    """
    if not _AUDIO.exists() or not (_DATASET / "data_train.hdf5").exists():
        print("    (skipped: no audio / dataset on disk)")
        return

    import h5py
    from trim_silence import load_audio, parse_task_csv

    mic_start, trials = parse_task_csv(_CSV)
    audio, fs = load_audio(_AUDIO)
    hop = int(round(0.02 * fs))

    checked = kept_frac = removed = 0
    for split in ("train", "val", "test"):
        with h5py.File(_DATASET / f"data_{split}.hdf5", "r") as f:
            for key in f.keys():
                g = f[key]
                n = g["input_features"].shape[0]
                gid = int(np.asarray(g.attrs["global_id"]).reshape(-1)[0])
                t0 = (trials[gid][0] - mic_start).total_seconds()
                db = frame_db(audio, fs, t0, n, hop)
                speech, _, _ = speech_mask(db, 22.0, 20.0)
                assert speech is not None, f"{split}/{key} gid {gid}: no contrast"
                keep = plan_keep(speech, 20.0, max_silence_ms=2000,
                                 keep_silence_ms=0, edge_keep_ms=None)
                assert speech[keep].sum() == speech.sum(), (
                    f"{split}/{key} gid {gid}: trimming dropped "
                    f"{int(speech.sum() - speech[keep].sum())} speech frames"
                )
                checked += 1
                kept_frac += keep.sum() / n
                removed += n - keep.sum()
    assert checked == 189, f"expected 189 trials, saw {checked}"
    print(f"    {checked} trials, {removed} silence bins removed, "
          f"{100 * kept_frac / checked:.0f}% of frames kept")


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
