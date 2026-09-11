#!/usr/bin/env python3
"""Remove over-long silent stretches from an HDF5 session.

The rule is one line: **any silent run longer than `--max-silence-ms` (default
2 s) is cut back to `--keep-silence-ms` (default 0, i.e. removed).** Short
pauses -- the ones between words, which is what the `<sil>` labels in
`seq_class_ids` describe -- are left exactly as they are.

Why this is needed
------------------
`read_begin`/`read_end` comes from the task CSV, not from the subject's voice.
When a marker fires late or the subject pauses, the trial is mostly room tone:
on 2026-08-14 the read durations run 3.6 s to 57.4 s (median 6.8 s), with
gid 105 at 50.0 s and gid 119 at 57.4 s on the 15-05-37 session and gid 148 at
52.3 s on the 10-11-24 one.

Two-stage, because the audio is not always present
--------------------------------------------------
The VAD needs the microphone recording. On a training box that only has the
HDF5, use the plan:

    # where the audio lives -- build a plan once (no --session-dir == plan only)
    python alt_models/trim_silence.py \
        --audio .../session-10-11-24/EnglishSpeech/microphone_audio.wav \
        --out-plan alt_models/cut_plans/2026-08-14_10-11-24.json

    # anywhere -- apply it, no audio needed
    python alt_models/trim_silence.py --plan alt_models/cut_plans/2026-08-14_10-11-24.json \
        --session-dir ../data/hdf5_data_512/t15.2026.08.14.10-11-24_tc_sbp_512 \
        --out-dir     ../data/hdf5_data_512/t15.2026.08.14.10-11-24_tc_sbp_512_trim

A plan is keyed by `global_id` and records each trial's expected bin count, so
applying it to the wrong session is detected rather than silently misaligned.
`--audio` + `--session-dir` together still work in one shot when both are
present.

`seq_class_ids` is never modified. CTC sums over alignment paths, so it needs no
frame-level alignment: dropping input frames cannot invalidate the transcript as
long as at least `L` frames remain. That is enforced per trial and reported.

How frames line up
------------------
Verified against the shipped day-1 session: for all 189 trials
`n_time_steps == floor(read_duration_sec * 50)`, i.e. 20 ms per row, and row `k`
covers `[read_begin + k*0.02, read_begin + (k+1)*0.02)` seconds. Sample 0 of the
wav is the CSV's `audio_record`/`mic_start` event. The plan records the bin count
it was built for; applying to a trial whose row count differs is refused.
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

    trials, pending = {}, {}
    for r in rows:
        if r["EventType"] != "mark":
            continue
        if r["Data3"] == "read_begin":
            pending[r["Data1"]] = stamp(r)
        elif r["Data3"] == "read_end" and r["Data1"] in pending:
            trials[int(r["Data1"])] = (pending.pop(r["Data1"]), stamp(r))
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

    Framing exactly what the feature matrix has (rather than framing the whole
    file and slicing) is what makes row `k` of the matrix and element `k` of the
    mask the same instant by construction.
    """
    start = int(round(t0_sec * fs))
    seg = audio[start:start + n_bins * hop]
    if len(seg) < n_bins * hop:
        seg = np.pad(seg, (0, n_bins * hop - len(seg)))
    frames = seg[:n_bins * hop].reshape(n_bins, hop)
    return 20.0 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-6)


# ---------------------------------------------------------------------------
# VAD + keep-planning
# ---------------------------------------------------------------------------
def speech_mask(db: np.ndarray, below_peak_db: float, above_floor_db: float):
    """Frames within `below_peak_db` of the trial's own speech level.

    Absolute thresholds do not survive a session change -- room tone on these
    recordings spans 15-35 dB while speech sits at 60-78 dB -- but these two
    percentile rules do. The `max` keeps a trial that is *entirely* room tone
    from classifying every frame as speech. Returns `(None, peak, floor)` when
    the contrast is too small to decide, so the caller can leave it alone.
    """
    peak = float(np.percentile(db, 95))
    floor = float(np.percentile(db, 20))
    if peak - floor < 15.0:
        return None, peak, floor
    return db >= max(peak - below_peak_db, floor + above_floor_db), peak, floor


def _runs(mask: np.ndarray):
    """Yield `(value, start, stop)` for each constant run, stop exclusive."""
    if mask.size == 0:
        return
    edges = np.flatnonzero(np.diff(np.asarray(mask, dtype=np.int8))) + 1
    bounds = np.concatenate(([0], edges, [mask.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield bool(mask[a]), int(a), int(b)


def plan_keep(speech: np.ndarray, hop_ms: float, max_silence_ms: float,
              keep_silence_ms: float, edge_keep_ms: float | None = None) -> np.ndarray:
    """Boolean `keep`: drop silent runs longer than `max_silence_ms`.

    Each dropped run leaves `keep_silence_ms` behind, so a long pause is
    *shortened* rather than erased and the model still sees a boundary.
    `edge_keep_ms` additionally caps leading/trailing silence; leaving it `None`
    means the edges obey the same rule as everything else, which is what you
    want when the instruction is "only touch silences over 2 s".
    """
    n = speech.size
    keep = np.ones(n, dtype=bool)
    if not speech.any():
        return keep

    max_bins = max_silence_ms / hop_ms
    keep_bins = int(round(keep_silence_ms / hop_ms))

    for is_speech, a, b in _runs(speech):
        if is_speech or (b - a) <= max_bins:
            continue
        keep[a + keep_bins: b] = False

    if edge_keep_ms is not None:
        edge_bins = int(round(edge_keep_ms / hop_ms))
        first = int(np.argmax(speech))
        last = int(n - 1 - np.argmax(speech[::-1]))
        keep[: max(0, first - edge_bins)] = False
        keep[min(n, last + 1 + edge_bins):] = False
    return keep


def cut_ranges(keep: np.ndarray):
    """`[[start, stop), ...]` frame ranges to delete."""
    return [[a, b] for v, a, b in _runs(~np.asarray(keep, dtype=bool)) if v]


def _trim_one(db, n_ids, args):
    """Return `(keep, note)` for one trial's frame-level dB curve."""
    speech, peak, floor = speech_mask(db, args.below_peak_db, args.above_floor_db)
    if speech is None:
        return np.ones(db.size, dtype=bool), (
            f"no speech/room-tone contrast ({peak - floor:.1f} dB) -- left untouched")
    keep = plan_keep(speech, args.hop_ms, args.max_silence_ms, args.keep_silence_ms,
                     args.edge_keep_ms)
    n_keep = int(keep.sum())
    if n_keep < n_ids:
        # CTC needs at least one frame per label. Keeping the loudest frames is a
        # strictly better failure mode than writing a trial the loss cannot align
        # at all -- but it is still a compromise, so it is reported, not silent.
        keep = np.zeros(db.size, dtype=bool)
        keep[np.sort(np.argsort(-db)[:n_ids])] = True
        return keep, (f"trim left {n_keep} < {n_ids} labels -- clamped back to "
                      "the loudest frames")
    return keep, ""


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------
def build_plan(audio, fs, hop, mic_start, trials, args) -> dict:
    """VAD every trial straight from the CSV, with no HDF5 needed.

    This is why a plan can be built on a machine that has the audio even though
    the session it will be applied to does not exist yet.
    """
    args.hop_ms = 1000.0 * hop / fs
    entries, report = {}, []
    for gid, (t0, t1) in sorted(trials.items()):
        read_sec = (t1 - t0).total_seconds()
        n_bins = int(np.floor(read_sec * FEATURE_HZ))
        if n_bins <= 0:
            continue
        db = frame_db(audio, fs, (t0 - mic_start).total_seconds(), n_bins, hop)
        keep, note = _trim_one(db, 0, args)
        cuts = cut_ranges(keep)
        entries[str(gid)] = {
            "n_bins": n_bins,
            "read_duration_sec": round(read_sec, 4),
            "read_begin": t0.isoformat(),
            "cut": cuts,
        }
        report.append({"global_id": gid, "split": "", "trial": "",
                       "bins_before": n_bins, "n_labels": -1,
                       "bins_after": int(keep.sum()),
                       "removed": n_bins - int(keep.sum()), "note": note,
                       "read_duration_sec": read_sec})
    return {
        "source": {"audio": str(args.audio), "task_csv": str(args.task_csv),
                   "mic_start": mic_start.isoformat()},
        "params": {
            "max_silence_ms": args.max_silence_ms,
            "keep_silence_ms": args.keep_silence_ms,
            "edge_keep_ms": args.edge_keep_ms,
            "below_peak_db": args.below_peak_db,
            "above_floor_db": args.above_floor_db,
            "frame_ms": round(args.hop_ms, 4),
        },
        "trials": entries,
    }, report


def read_session_attr(session_dir: Path):
    """The name stamped on the source trials, read from the first group we see.

    `session` has to equal the directory name: `evaluate_model_extended.py:199`
    does `sessions.index(session)` on the per-trial attribute, while `rnn_trainer`
    builds `<dataset_dir>/<config entry>/data_train.hdf5` from the directory. A
    MATLAB output renamed by hand keeps the stamp it was written with, so the two
    silently diverge and only surface at evaluation time.
    """
    for split in SPLITS:
        path = session_dir / f"data_{split}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            for key in f.keys():
                value = f[key].attrs.get("session")
                if value is not None:
                    if isinstance(value, (bytes, bytearray)):
                        return value.decode("utf-8")
                    return str(value)
    return None


def load_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if "trials" not in plan:
        raise SystemExit(f"{path}: not a cut plan (no `trials` key)")
    return plan


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------
def process_split(path: Path, out_path: Path, plan: dict, args, write: bool,
                  session_name):
    """Apply a plan to one split. `plan` maps gid -> {n_bins, cut}."""
    rows = []
    dst = h5py.File(out_path, "w") if write else None
    with h5py.File(path, "r") as src:
        for key in sorted(src.keys()):
            g = src[key]
            feats = g["input_features"][:]
            n_bins = feats.shape[0]
            gid = int(np.asarray(g.attrs["global_id"]).reshape(-1)[0])
            n_ids = len(g["seq_class_ids"][:])

            # `refused` means "this trial could not be trimmed as planned". The
            # trial itself is still written, untouched, so the session stays
            # structurally complete -- but the run must not report success, or a
            # systematically misaligned plan would pass for a clean job.
            row = {"split": path.stem.replace("data_", ""), "trial": key,
                   "global_id": gid, "bins_before": n_bins, "n_labels": n_ids,
                   "bins_after": n_bins, "removed": 0, "note": "", "refused": False}

            entry = plan["trials"].get(str(gid))
            keep = np.ones(n_bins, dtype=bool)
            if entry is None:
                row["refused"] = True
                row["note"] = "global_id absent from the cut plan -- left untouched"
            elif entry["n_bins"] != n_bins:
                row["refused"] = True
                row["note"] = (f"plan expects {entry['n_bins']} bins but this trial "
                               f"has {n_bins} -- REFUSED (frames would misalign)")
            else:
                for a, b in entry["cut"]:
                    keep[a:b] = False
                if int(keep.sum()) < n_ids:
                    row["refused"] = True
                    row["note"] = (f"plan would leave {int(keep.sum())} < {n_ids} "
                                   "labels -- left untouched")

            n_keep = int(keep.sum())
            if write:
                out = dst.create_group(key)
                out.create_dataset("input_features", data=feats[keep],
                                   dtype=feats.dtype)
                for name in g:
                    if name != "input_features":
                        g.copy(name, out, name=name)
                for k, v in g.attrs.items():
                    out.attrs[k] = v
                out.attrs["n_time_steps"] = np.int32(n_keep)
                out.attrs["silence_trim"] = json.dumps({
                    "bins_before": int(n_bins), "bins_after": n_keep,
                    "max_silence_ms": plan["params"]["max_silence_ms"],
                    "keep_silence_ms": plan["params"]["keep_silence_ms"],
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
    p.add_argument("--session-dir", default=None,
                   help="session dir holding data_{train,val,test}.hdf5")
    p.add_argument("--audio", default=None, help="microphone_audio.wav")
    p.add_argument("--task-csv", default=None,
                   help="task CSV with read_begin/read_end; defaults to the only "
                        "data_*.csv next to --audio")
    p.add_argument("--out-dir", default=None,
                   help="where to write the trimmed session; defaults to "
                        "<session-dir>_trim")
    p.add_argument("--out-plan", default=None,
                   help="write the computed cut plan here (for use without audio)")
    p.add_argument("--plan", default=None,
                   help="apply a previously built cut plan instead of the audio")
    p.add_argument("--max-silence-ms", type=float, default=2000.0,
                   help="silent runs longer than this are cut (default 2000)")
    p.add_argument("--keep-silence-ms", type=float, default=0.0,
                   help="how much of a cut run survives (default 0 = removed)")
    p.add_argument("--edge-keep-ms", type=float, default=None,
                   help="also cap leading/trailing silence at this; default is to "
                        "apply the same >max-silence-ms rule to the edges")
    p.add_argument("--below-peak-db", type=float, default=22.0)
    p.add_argument("--above-floor-db", type=float, default=20.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.plan and (args.audio or args.out_plan):
        raise SystemExit("--plan cannot be combined with --audio/--out-plan")
    if not args.plan and not args.audio:
        raise SystemExit("pass either --audio (to measure) or --plan (to apply)")

    report = None
    if args.plan:
        plan = load_plan(Path(args.plan))
        print(f"plan    : {args.plan}")
        print(f"built   : max_silence={plan['params']['max_silence_ms']} ms, "
              f"keep={plan['params']['keep_silence_ms']} ms, "
              f"frame={plan['params']['frame_ms']} ms")
        print(f"source  : {plan['source']['task_csv']}")
        args.max_silence_ms = plan["params"]["max_silence_ms"]
        args.keep_silence_ms = plan["params"]["keep_silence_ms"]
    else:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise SystemExit(f"audio not found: {audio_path.resolve()}")
        csv_path = Path(args.task_csv) if args.task_csv else None
        if csv_path is None:
            found = sorted(audio_path.parent.glob("data_*.csv"))
            if len(found) != 1:
                raise SystemExit(
                    f"expected exactly one data_*.csv next to {audio_path}, found "
                    f"{[f.name for f in found]} -- pass --task-csv")
            csv_path = found[0]
        args.audio, args.task_csv = audio_path, csv_path
        mic_start, trials = parse_task_csv(csv_path)
        audio, fs = load_audio(audio_path)
        hop = int(round(0.02 * fs))
        print(f"audio   : {audio_path}")
        print(f"task csv: {csv_path}")
        print(f"audio {len(audio)/fs:.1f} s @ {fs} Hz | mic_start {mic_start} | "
              f"{len(trials)} trials in CSV | frame {1000.0*hop/fs:.3f} ms")
        plan, report = build_plan(audio, fs, hop, mic_start, trials, args)

    if args.out_plan:
        out_plan = Path(args.out_plan)
        out_plan.parent.mkdir(parents=True, exist_ok=True)
        out_plan.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        cut = sum(len(t["cut"]) for t in plan["trials"].values())
        saved = sum(sum(b - a for a, b in t["cut"]) for t in plan["trials"].values())
        print(f"\nplan written: {out_plan}  ({len(plan['trials'])} trials, "
              f"{cut} cut ranges, {saved} bins = {saved*0.02:.1f} s)")
        if report:
            _print_report(report)

    if not args.session_dir:
        if not args.out_plan:
            raise SystemExit("nothing to do: pass --session-dir, or --out-plan")
        return 0

    session_dir = Path(args.session_dir)
    if not session_dir.is_dir():
        raise SystemExit(f"session dir not found: {session_dir}")
    # Default to a *parallel dataset root*, not a `_trim` sibling. A sibling would
    # make `make_all_day_config.py` enumerate both the trimmed and untrimmed copy
    # as two separate days -- the model would get two day layers for one day of
    # data, and the duplicate would be the untrimmed one. Keeping the session name
    # identical also means the trial `session` attribute needs no rewrite, since it
    # already equals the directory name.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        root = session_dir.parent
        out_dir = root.with_name(root.name + "_trim") / session_dir.name

    # The output directory name IS the session name -- always stamp the trials to
    # match it, whatever the source said. Otherwise a hand-renamed MATLAB folder
    # produces a trimmed session that only fails later, at `sessions.index()`.
    session_name = out_dir.name
    stamped = read_session_attr(session_dir)
    if stamped is not None and stamped != session_name:
        print(f"\nWARNING: the source trials are stamped session={stamped!r} but "
              f"this session is {session_name!r}.\n"
              f"         The OUTPUT will be stamped {session_name!r} to match its "
              "directory.\n"
              "         The source still disagrees -- run install_matlab_session.py "
              "on it\n"
              "         so the config that points at the source stays usable.")

    print(f"\nsession : {session_dir.name}")
    print(f"output  : {'(dry run)' if args.dry_run else out_dir}")
    print()
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
        rows = process_split(src, out_dir / src.name, plan, args,
                             write=not args.dry_run, session_name=session_name)
        all_rows.extend(rows)
        removed = sum(r["removed"] for r in rows)
        before = sum(r["bins_before"] for r in rows)
        print(f"{split:>5}: {len(rows):>3} trials  {before:>7} -> {before-removed:>7} bins "
              f"({100.0*removed/max(before,1):5.1f}% removed)")

    if not args.dry_run:
        for name in CARRY_OVER:
            s = session_dir / name
            if s.exists():
                shutil.copy2(s, out_dir / name)

        # `metadata.json` is not optional downstream for anything but a 41-class
        # model: `evaluate_model_extended._load_session_bundle` falls back to the
        # official 41-class order when it is missing, and a 35-class model then
        # dies inside the LM path with "logits last dimension (35) does not match
        # source_order length (41)". Trimming is where the file gets dropped, and
        # this is the last point at which the source is still on hand, so say so
        # rather than producing a session that trains fine and cannot be
        # evaluated.
        if not (out_dir / "metadata.json").exists():
            print(f"\nWARNING: {session_dir.name} has no metadata.json to carry over, so "
                  f"{out_dir.name} has none either.\n"
                  "         Training will work; evaluation will not (the phoneme order "
                  "is unknown,\n"
                  "         and evaluation guesses 41 classes). Install the session "
                  "properly first --\n"
                  "         alt_models/install_matlab_session.py writes the vocabulary -- "
                  "or copy one in:\n"
                  f"           cp alt_models/session_metadata/{session_name}.json "
                  f"{out_dir / 'metadata.json'}\n")

        meta_path = out_dir / "metadata.json"
        if session_name and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["session"] = session_name
            meta["silence_trim"] = {"source_session": session_dir.name,
                                    **plan["params"]}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _print_report(all_rows, out_dir if not args.dry_run else None)
    if args.dry_run:
        print("\ndry run -- nothing written")
    else:
        print(f"\nNext: python alt_models/audit_hdf5.py --dataset_dir {out_dir.parent}")

    refused = [r for r in all_rows if r.get("refused")]
    if refused:
        print(f"\n{len(refused)} trial(s) were NOT trimmed as planned (see the "
              "notes above). The cut plan does not line up with this session -- "
              "do not train on this output.", file=sys.stderr)
        return 1
    return 0


def _print_report(rows, out_dir=None):
    if not rows:
        return
    noted = [r for r in rows if r["note"]]
    long_ones = [r for r in rows if r.get("read_duration_sec", 0) > 15.0]
    worst = sorted(rows, key=lambda r: -r["removed"])[:8]

    print("\nmost trimmed:")
    for r in worst:
        dur = f"{r['read_duration_sec']:5.1f}s read" if "read_duration_sec" in r else ""
        print(f"  gid {r['global_id']:>3} {r['split']:>5} {r['bins_before']:>5} -> "
              f"{r['bins_after']:>5} bins  ({r['removed']*0.02:5.1f} s cut)  {dur}")
    if long_ones:
        print(f"\ntrials with read > 15 s ({len(long_ones)}):")
        for r in sorted(long_ones, key=lambda r: -r["read_duration_sec"]):
            print(f"  gid {r['global_id']:>3}  read {r['read_duration_sec']:5.1f} s -> "
                  f"{r['bins_after']*0.02:5.1f} s kept ({r['removed']*0.02:.1f} s cut)")
    if noted:
        print(f"\n{len(noted)} trial(s) needed attention:")
        for r in noted[:20]:
            print(f"  gid {r['global_id']:>3} {r['split']:>5}: {r['note']}")
        if len(noted) > 20:
            print(f"  ... and {len(noted)-20} more")
    if out_dir is not None:
        with open(Path(out_dir) / "trim_report.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
