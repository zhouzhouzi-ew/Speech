"""Turn a finished local training log into a CSV + markdown results report.

Reads the trainer's stdout log (`_train_256_7day.log`) and the checkpoint
directory, and writes two files next to the checkpoint:

    <output_dir>/results_summary.csv      one row per validation step
    <output_dir>/results_summary.md       the same, as tables + per-day PER

The log is the only place the per-step val numbers survive in full: the trainer
keeps `val_metrics.pkl` for the *best* step only, and `train_val_trials.json`
records the split.  Parsing stdout is therefore the cheapest way to get the
whole curve without re-running anything.

    cd Speech/model_training
    python alt_models/report_local_256_run.py --log _train_256_7day.log \\
        --output_dir trained_models/baseline_rnn_256_spike_7day
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

TS = r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}: "

RE_VAL = re.compile(
    TS + r"Val batch (\d+): PER \(avg\): ([\d.]+) CTC Loss \(avg\): ([\d.]+) time: ([\d.]+)"
)
RE_TRAIN = re.compile(
    TS + r"Train batch (\d+): loss: ([\d.]+) grad norm: ([\d.]+) time: ([\d.]+)"
)
RE_DAY = re.compile(TS + r"(\S+) val PER: ([\d.]+)")
RE_BEST = re.compile(TS + r"New best test PER ([\d.]+|inf) --> ([\d.]+)")
RE_TOTAL = re.compile(TS + r"Total training time: ([\d.]+) minutes")


def parse(log_text: str) -> dict:
    """Return {'val': [...], 'train': [...], 'best': [...], 'total_min': float|None}.

    `val` rows carry the per-day PER of the *most recent* val step; the log
    prints `Val batch` first, then one line per day, so the day lines are
    attached to the last val row seen.
    """
    train, val, best = [], [], []
    total_min = None

    for line in log_text.splitlines():
        if (m := RE_TRAIN.match(line)):
            train.append({
                "batch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "grad_norm": float(m.group(3)),
                "step_sec": float(m.group(4)),
            })
        elif (m := RE_VAL.match(line)):
            val.append({
                "batch": int(m.group(1)),
                "avg_PER": float(m.group(2)),
                "val_ctc_loss": float(m.group(3)),
                "val_sec": float(m.group(4)),
                "day_PER": {},
            })
        elif (m := RE_DAY.match(line)) and val:
            val[-1]["day_PER"][m.group(1)] = float(m.group(2))
        elif (m := RE_BEST.match(line)):
            best.append({"batch": val[-1]["batch"] if val else None,
                         "from": m.group(1), "to": float(m.group(2))})
        elif (m := RE_TOTAL.match(line)):
            total_min = float(m.group(1))

    return {"val": val, "train": train, "best": best, "total_min": total_min}


def write_csv(rows: list, path: Path) -> None:
    days = sorted({d for r in rows for d in r["day_PER"]})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["batch", "avg_PER", "val_ctc_loss", "val_sec"] + [f"PER/{d}" for d in days])
        for r in rows:
            w.writerow([r["batch"], f"{r['avg_PER']:.4f}", f"{r['val_ctc_loss']:.4f}",
                        f"{r['val_sec']:.1f}"] + [f"{r['day_PER'].get(d, float('nan')):.4f}" for d in days])


def write_md(log_text: str, parsed: dict, out_dir: Path, args, path: Path) -> None:
    val, train, best = parsed["val"], parsed["train"], parsed["best"]
    days = sorted({d for r in val for d in r["day_PER"]})

    best_row = min(val, key=lambda r: r["avg_PER"]) if val else None
    step_times = [t["step_sec"] for t in train if t["batch"] > 0]
    mean_step = sum(step_times) / len(step_times) if step_times else float("nan")

    L = []
    L.append("# 本地 256 维 spike-only 训练结果\n")
    L.append(f"- 日志: `{args.log}`")
    L.append(f"- 输出目录: `{out_dir}`")
    L.append(f"- 训练步数: {args.num_batches if args.num_batches else '见 config'}"
             f" | batch_size: {args.batch_size}")
    L.append(f"- 实测平均每步: {mean_step:.2f} s ({len(step_times)} 步采样)")
    if parsed["total_min"] is not None:
        L.append(f"- 总训练时长: {parsed['total_min']:.1f} 分钟")
    L.append("")

    L.append("## 结论\n")
    if best_row:
        L.append(f"- **最佳 val PER = {best_row['avg_PER']:.4f}"
                 f" (batch {best_row['batch']})**")
        L.append(f"- 该步 val CTC Loss = {best_row['val_ctc_loss']:.4f}")
        L.append(f"- 首个 val PER = {val[0]['avg_PER']:.4f}；共 {len(val)} 个验证点")
    else:
        L.append("- 日志里没有解析到验证点，训练可能未跑到第一次验证。")
    L.append("")
    L.append("> PER = Σ 编辑距离 / Σ 真实 phoneme 数（在**整个验证集**上累加后相除，"
             "不是逐 trial 平均）。**不能**由 PER 直接换算成词准确率。\n")

    if days:
        L.append("## 分天 val PER（最后一次验证）\n")
        L.append("| session | val PER |")
        L.append("|---|---|")
        for d in days:
            L.append(f"| `{d}` | {val[-1]['day_PER'].get(d, float('nan')):.4f} |")
        L.append("")

    L.append("## 逐验证点曲线\n")
    L.append("| batch | avg PER | val CTC loss | val 耗时(s) |")
    L.append("|---|---|---|---|")
    for r in val:
        L.append(f"| {r['batch']} | {r['avg_PER']:.4f} | {r['val_ctc_loss']:.4f} | {r['val_sec']:.1f} |")
    L.append("")

    if best:
        L.append("## best checkpoint 更新记录\n")
        L.append("| batch | 旧 best | 新 best |")
        L.append("|---|---|---|")
        for b in best:
            L.append(f"| {b['batch']} | {b['from']} | {b['to']:.4f} |")
        L.append("")

    ckpt = out_dir / "checkpoint" / "best_checkpoint"
    L.append("## 产物\n")
    L.append(f"- `{ckpt}` — {'已存在' if ckpt.exists() else '**缺失**'}")
    L.append(f"- `{out_dir / 'checkpoint' / 'args.yaml'}` — "
             f"{'已存在' if (out_dir / 'checkpoint' / 'args.yaml').exists() else '**缺失**'}")
    L.append(f"- `{out_dir / 'checkpoint' / 'val_metrics.pkl'}` — 最佳步的完整验证指标")
    L.append("")
    L.append("## 拿到 GPU 机器上评估\n")
    L.append("```bash")
    L.append(f"python evaluate_model.py --model_path {out_dir.as_posix()} \\")
    L.append(f"    --data_dir ../{args.dataset_dir} --eval_type test --gpu_number 0 \\")
    L.append(f"    --skip_lm --output_prefix baseline_256_7day")
    L.append("```")
    L.append("")
    L.append("去掉 `--skip_lm` 并启动 3-gram WFST LM server 即得 LM-on WER"
             "（见 `alt_models/CUT_AND_TRAIN_RUNBOOK.md` §7）。")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, default=Path("_train_256_7day.log"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("trained_models/baseline_rnn_256_spike_7day"))
    p.add_argument("--dataset_dir", default="data/hdf5_final")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_batches", type=int, default=1200)
    args = p.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}")
        return 2
    text = args.log.read_text(encoding="utf-8", errors="replace")
    parsed = parse(text)

    if not parsed["val"]:
        print("no validation rows parsed from the log -- had training reached its "
              "first val step?")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results_summary.csv"
    md_path = args.output_dir / "results_summary.md"
    write_csv(parsed["val"], csv_path)
    write_md(text, parsed, args.output_dir, args, md_path)

    best = min(r["avg_PER"] for r in parsed["val"])
    print(f"{len(parsed['val'])} val step(s) parsed")
    print(f"best avg val PER: {best:.4f}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
