#!/usr/bin/env python3
"""Trim over-long silent stretches out of an HDF5 session, guided by the mic audio.

Why this exists
---------------
`read_begin`/`read_end` comes from the task CSV, not from the subject's actual
voice. When a marker fires late (or the subject pauses for tens of seconds) the
resulting trial is mostly room tone: on the 2026-08-14 15-05-37 session the read
durations run 3.6 s to 57.4 s, median 6.8 s, with gid 105 at 50.0 s and gid 119
at 57.4 s. Those trials are ~100 % silence, cost a disproportionate amount of
memory and compute, and contribute gradients that teach the model nothing.

What it does
------------
For each trial it builds a 20 ms-frame voice-activity mask from the microphone
recording, then drops frames:

  * leading and trailing silence, beyond `--edge-keep-ms` each;
  * any *internal* silent run longer than `--max-silence-ms`, collapsed to
    `--keep-silence-ms` so the pause is still visible to the model as a pause.

`seq_class_ids` is deliberately **not** touched. CTC needs no frame-level
alignment -- it sums over every alignment path -- so removing frames from the
input cannot invalidate the transcript, as long as at least `L` frames survive.
The script enforces that and reports it per trial.

How the frames line up
----------------------
Verified against the shipped day-1 session: for all 189 trials,

    n_time_steps == floor(read_duration_sec * 50)

i.e. 20 ms per row, and row `k` covers wall time
`[read_begin + k*0.02, read_begin + (k+1)*0.02)` seconds. The audio's sample 0 is
the CSV's `audio_record`/`mic_start` event, so a wall clock time `t` is at
sample `(t - mic_start) * fs`. That holds to within 0.05 s of the file's own
duration on the session this was written against. The script re-checks the
`n_time_steps` identity per trial and refuses to touch a trial that violates it.

Usage
-----
    cd Speech/model_training
    python alt_models/trim_silence.py \
        --session-dir ../data/hdf5_data_512/t15.2026.08.14.15-05-37_tc_sbp_512 \
        --audio /mnt/d/wwl/.../session-15-05-37/EnglishSpeech/microphone_audio.wav \
        --out-dir ../data/hdf5_data_512/t15.2026.08.14.15-05-37_tc_sbp_512_trim

Add `--dry-run` first: it prints the same report and writes nothing. Then run
`alt_models/audit_hdf5.py` on the output before training on it.

Trim every day with the same flags -- a model trained on trimmed day 2 and
untrimmed day 1 is being asked to reconcile two different input distributions
on top of the day difference it is supposed to be learning.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import wave
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")
FEATURE_HZ = 50.0          # 20 ms bins
CARRY_OVER = ("metadata.json", "trial_manifest.csv", "b2txt_description.csv")


# ---------------------------------------------------------------------------
# audio <-> wall clock
# ---------------------------------------------------------------------------
def parse_task_csv(path: Path):
    """Return `(mic_start, {gid: (read_begin, read_end)})` from a task CSV.

    Timestamps carry a UTC offset, and `read_begin`/`read_end` live in `Data3`
    while the `mic_start` marker puts its label in `Data4`.
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    stamp = lambda r: datetime.fromisoformat(r["Timestamp"])

    mic_start = None
    for r in rows:
        if r["EventType"] == "audio_record" and r.get("Data4") == "mic_start":
            mic_start = stamp(r)
            break
    if mic_start is None:
        raise SystemExit(
            f"{path}: no `audio_record`/`mic_start` row, so the audio cannot be "
            "aligned to wall clock. Every trial's read_begin would be wrong."
        )

    trials, pending_begin = {}, {}
    for r in rows:
        ev = r["EventType"]
        if ev == "mark":
            if r["Data3"] == "read_begin":
                pending_begin[r["Data1"]] = stamp(r)
            elif r["Data3"] == "read_end" and r["Data1"] in pending_begin:
                trials[int(r["Data1"])] = (pending_begin.pop(r["Data1"]), stamp(r))
    if not trials:
        raise SystemExit(f"{path}: no read_begin/read_end pairs found")
    return mic_start, trials


def load_audio(path: Path):
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        if w.getnchannels() != 1:
            raise SystemExit(f"{path}: expected mono, got {w.getnchannels()} channels")
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM, got {w.getsampwidth()*8}-bit")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32), fs


def frame_db(audio: np.ndarray, fs: int, t0_sec: float, n_bins: int, hop: int):
    """Per-20 ms-frame RMS in dB, exactly `n_bins` long, starting at `t0_sec`.

    Reading exactly the frames the feature matrix has (rather than framing the
    whole file and slicing) is what makes row `k` of the matrix and element `k`
    of the mask the same instant by construction.
    """
    start = int(round(t0_sec * fs))
    seg = audio[start:start + n_bins * hop]
    usable = len(seg) // hop
    if usable < n_bins:
        seg = np.pad(seg, (0, n_bins * hop - len(seg)))
    frames = seg[:n_bins * hop].reshape(n_bins, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    return 20.0 * np.log10(rms + 1e-6)


# ---------------------------------------------------------------------------
# VAD + keep-planning
# ---------------------------------------------------------------------------
def speech_mask(db: np.ndarray, below_peak_db: float, above_floor_db: float):
    """Frames within `below_peak_db` of the trial's own speech level.

    Absolute thresholds do not survive a session change (room tone here sits at
    15-35 dB and speech at 60-78 dB); these two percentile rules do, and the
    `max` keeps a trial that is *entirely* room tone from classifying every
    frame as speech.
    """
    peak = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 20))
    if peak - floor < 15.0:
        return None, peak, floor          # no usable contrast in this trial
    return db >= max(peak - below_peak_db, floor + above_floor_db), peak, floor


def _runs(mask: np.ndarray):
    """Yield `(value, start, stop)` for each constant run, stop exclusive."""
    if mask.size == 0:
        return
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [mask.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield bool(mask[a]), int(a), int(b)


def plan_keep(speech: np.ndarray, hop_ms: float, edge_keep_ms: float,
              max_silence_ms: float, keep_silence_ms: float) -> np.ndarray:
    """Boolean `keep` over frames: edge silence capped, long internal silences collapsed."""
    n = speech.size
    keep = np.ones(n, dtype=bool)
    if not speech.any():
        return keep

    edge_bins = int(round(edge_keep_ms / hop_ms))
    max_sil_bins = max_silence_ms / hop_ms
    keep_sil_bins = int(round(keep_silence_ms / hop_ms))

    first = int(np.argmax(speech))
    last = int(n - 1 - np.argmax(speech[::-1]))

    keep[: max(0, first - edge_bins)] = False
    keep[min(n, last + 1 + edge_bins):] = False

    for is_speech, a, b in _runs(speech):
        if is_speech or a == 0 or b == n:
            continue                       # edges handled above
        if (b - a) > max_sil_bins:
            keep[a + keep_sil_bins: b] = False
    return keep


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def process_split(path: Path, out_path: Path, audio: np.ndarray, fs: int, hop: int,
                  mic_start, trials: dict, args, write: bool, session_name):
    rows = []
    dst = h5py.File(out_path, "w") if write else None
    with h5py.File(path, "r") as src:
        for key in sorted(src.keys()):
            g = src[key]
            feats = g["input_features"][:]
            n_bins = feats.shape[0]
            gid = int(np.asarray(g.attrs["global_id"]).reshape(-1)[0])
            n_ids = len(g["seq_class_ids"][:])

            row = {"split": path.stem.replace("data_", ""), "trial": key,
                   "global_id": gid, "bins_before": n_bins, "n_labels": n_ids,
                   "bins_after": n_bins, "removed": 0, "note": ""}

            keep = np.ones(n_bins, dtype=bool)
            db = None
            span = trials.get(gid)
            if span is None:
                row["note"] = "global_id not in task CSV -- left untouched"
            elif (span[1] - span[0]).total_seconds() * FEATURE_HZ < n_bins - 2:
                row["note"] = "read window shorter than feature rows -- left untouched"
            else:
                t0 = (span[0] - mic_start).total_seconds()
                db = frame_db(audio, fs, t0, n_bins, hop)
                speech, peak, floor = speech_mask(db, args.below_peak_db,
                                                  args.above_floor_db)
                if speech is None:
                    row["note"] = (f"no speech/room-tone contrast "
                                   f"({peak - floor:.1f} dB) -- left untouched")
                    db = None
                else:
                    keep = plan_keep(speech, 1000.0 * hop / fs, args.edge_keep_ms,
                                     args.max_silence_ms, args.keep_silence_ms)

            n_keep = int(keep.sum())
            if n_keep < n_ids and db is not None:
                # CTC needs at least one frame per label. Keeping the loudest
                # frames is a strictly better failure mode than writing a trial
                # the loss cannot align at all -- but it is still a compromise,
                # so it lands in the report rather than passing silently.
                row["note"] = (f"trim left {n_keep} < {n_ids} labels -- clamped "
                               "back to the loudest frames")
                keep = np.zeros(n_bins, dtype=bool)
                keep[np.sort(np.argsort(-db)[:n_ids])] = True
                n_keep = n_ids

            if write:
                out = dst.create_group(key)
                out.create_dataset("input_features", data=feats[keep],
                                   dtype=feats.dtype, compression=None)
                for name in g:
                    if name == "input_features":
                        continue
                    g.copy(name, out, name=name)
                for k, v in g.attrs.items():
                    out.attrs[k] = v
                out.attrs["n_time_steps"] = np.int32(n_keep)
                out.attrs["silence_trim"] = json.dumps({
                    "bins_before": int(n_bins), "bins_after": n_keep,
                    "max_silence_ms": args.max_silence_ms,
                    "keep_silence_ms": args.keep_silence_ms,
                    "edge_keep_ms": args.edge_keep_ms,
                })
                if session_name:
                    out.attrs["session"] = session_name

            row["bins_after"] = n_keep
            row["removed"] = n_bins - n_keep
            rows.append(row)
    if dst is not None:
        dst.close()
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", required=True,
                   help="session dir holding data_{train,val,test}.hdf5")
    p.add_argument("--audio", required=True, help="microphone_audio.wav")
    p.add_argument("--task-csv", default=None,
                   help="task CSV with read_begin/read_end; defaults to the only "
                        "data_*.csv next to --audio")
    p.add_argument("--out-dir", default=None,
                   help="where to write the trimmed session; defaults to "
                        "<session-dir>_trim")
    p.add_argument("--max-silence-ms", type=float, default=1000.0,
                   help="internal silent runs longer than this are collapsed (default 1000)")
    p.add_argument("--keep-silence-ms", type=float, default=200.0,
                   help="how much of a collapsed run to keep (default 200)")
    p.add_argument("--edge-keep-ms", type=float, default=200.0,
                   help="leading/trailing silence to keep (default 200)")
    p.add_argument("--below-peak-db", type=float, default=22.0,
                   help="a frame is speech if within this of the trial's p95 (default 22)")
    p.add_argument("--above-floor-db", type=float, default=20.0,
                   help="...and at least this far above the trial's p20 (default 20)")
    p.add_argument("--dry-run", action="store_true",
                   help="report only; write nothing")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        raise SystemExit(f"session dir not found: {session_dir}")

    audio_path = Path(args.audio)
    csv_path = Path(args.task_csv) if args.task_csv else None
    if csv_path is None:
        found = sorted(audio_path.parent.glob("data_*.csv"))
        if len(found) != 1:
            raise SystemExit(
                f"expected exactly one data_*.csv next to {audio_path}, found "
                f"{[f.name for f in found]} -- pass --task-csv"
            )
        csv_path = found[0]

    out_dir = Path(args.out_dir) if args.out_dir else session_dir.with_name(
        session_dir.name + "_trim")
    session_name = None
    if out_dir.name != session_dir.name:
        session_name = out_dir.name

    print(f"session : {session_dir.name}")
    print(f"audio   : {audio_path}")
    print(f"task csv: {csv_path}")
    print(f"output  : {'(dry run)' if args.dry_run else out_dir}")
    print()

    mic_start, trials = parse_task_csv(csv_path)
    audio, fs = load_audio(audio_path)
    hop = int(round(0.02 * fs))

    print(f"audio {len(audio)/fs:.1f} s @ {fs} Hz | mic_start {mic_start} "
          f"| {len(trials)} trials in CSV")
    print(f"frame = {1000.0 * hop / fs:.3f} ms (nominal 20)\n")

    if not args.dry_run:
        if out_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"{out_dir} exists; pass --overwrite")
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    all_rows = []
    for split in SPLITS:
        src = session_dir / f"data_{split}.hdf5"
        if not src.exists():
            continue
        rows = process_split(src, out_dir / src.name, audio, fs, hop, mic_start,
                             trials, args, write=not args.dry_run,
                             session_name=session_name)
        all_rows.extend(rows)
        removed = sum(r["removed"] for r in rows)
        before = sum(r["bins_before"] for r in rows)
        print(f"{split:>5}: {len(rows):>3} trials  {before:>7} -> {before - removed:>7} bins "
              f"({100.0 * removed / max(before, 1):5.1f}% removed)")

    if not args.dry_run:
        for name in CARRY_OVER:
            s = session_dir / name
            if s.exists():
                shutil.copy2(s, out_dir / name)
        if session_name:
            meta_path = out_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["session"] = session_name
                meta["silence_trim"] = {
                    "source_session": session_dir.name,
                    "max_silence_ms": args.max_silence_ms,
                    "keep_silence_ms": args.keep_silence_ms,
                    "edge_keep_ms": args.edge_keep_ms,
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = out_dir / "trim_report.csv" if not args.dry_run else None
    if report is not None:
        with open(report, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    noted = [r for r in all_rows if r["note"]]
    worst = sorted(all_rows, key=lambda r: -r["removed"])[:5]
    print(f"\nmost trimmed:")
    for r in worst:
        print(f"  gid {r['global_id']:>3} {r['split']:>5} {r['bins_before']:>5} -> "
              f"{r['bins_after']:>5} bins ({r['removed'] * 20 / 1000:.1f} s removed)")
    if noted:
        print(f"\n{len(noted)} trial(s) needed attention:")
        for r in noted[:20]:
            print(f"  gid {r['global_id']:>3} {r['split']:>5}: {r['note']}")
        if len(noted) > 20:
            print(f"  ... and {len(noted) - 20} more (see {report})")

    if args.dry_run:
        print("\ndry run -- nothing written")
    else:
        print(f"\nWrote {out_dir}")
        print(f"Next: python alt_models/audit_hdf5.py --dataset_dir {out_dir.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
