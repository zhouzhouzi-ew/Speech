from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from rnn_model import GRUDecoder
from evaluate_model_helpers import (
    LOGIT_TO_PHONEME,
    expand_logits_to_official_order,
    finalize_remote_lm,
    get_current_redis_time_ms,
    load_h5py_file,
    load_phoneme_word_lexicon,
    load_session_metadata,
    load_session_phoneme_order,
    phonemes_to_words,
    rearrange_speech_logits_pt,
    remove_punctuation,
    reset_remote_language_model,
    runSingleDecodingStep,
    send_logits_to_remote_lm,
    sequence_edit_distance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEXICON_PATH = REPO_ROOT / "language_model" / "examples" / "speech" / "s0" / "t15_2026_08_01_lm_inputs" / "dict"


def load_model_args(path):
    try:
        return OmegaConf.load(path)
    except UnicodeDecodeError:
        for encoding in ("utf-8-sig", "gbk", "cp936", "latin1"):
            try:
                with open(path, "r", encoding=encoding) as handle:
                    return OmegaConf.create(handle.read())
            except UnicodeDecodeError:
                continue
        raise


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        cleaned[new_key] = value
    return cleaned


def _resolve_output_dir(model_path: str, output_dir: Optional[str]) -> Path:
    if output_dir is None:
        return Path(model_path) / "eval_outputs"
    return Path(output_dir)


def _load_csv_if_available(csv_path: str) -> Optional[pd.DataFrame]:
    path = Path(csv_path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _decode_trial_logits(
    logits: np.ndarray,
    source_order: Sequence[str],
    *,
    n_time_steps: Optional[int] = None,
    patch_size: Optional[int] = None,
    patch_stride: Optional[int] = None,
) -> List[str]:
    """Mirror the trainer's greedy CTC decode.

    The training loop first truncates logits to the valid time range,
    then applies `unique_consecutive`, and only after that removes the
    blank symbol (index 0). Keeping that order here avoids PER drift
    between trainer-side validation and standalone evaluation.
    """
    valid_len = logits.shape[0]
    if n_time_steps is not None and patch_size and patch_stride:
        adjusted_len = int((int(n_time_steps) - int(patch_size)) / int(patch_stride) + 1)
        valid_len = min(valid_len, max(0, adjusted_len))

    pred_seq_ids = torch.argmax(torch.as_tensor(logits[:valid_len]), dim=-1)
    pred_seq_ids = torch.unique_consecutive(pred_seq_ids, dim=-1)
    pred_seq_ids = pred_seq_ids.cpu().numpy()
    pred_seq_ids = np.array([int(item) for item in pred_seq_ids if int(item) != 0])
    return [source_order[idx] for idx in pred_seq_ids]


def _maybe_words_from_phonemes(pred_seq, phoneme_word_lexicon):
    if phoneme_word_lexicon is None:
        return None
    return phonemes_to_words(pred_seq, phoneme_word_lexicon)


def _flatten_true_phonemes(seq_class_ids, seq_len, source_order):
    if seq_class_ids is None or seq_len is None:
        return None
    true_seq = seq_class_ids[: int(seq_len)]
    return [source_order[int(item)] for item in true_seq]


def _write_trial_details(detail_rows: List[Dict[str, object]], csv_path: Path) -> None:
    pd.DataFrame(detail_rows).to_csv(csv_path, index=False)


def _markdown_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_report(summary: Dict[str, object], detail_rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append(f"# {summary['output_prefix']} {summary['eval_type']} report")
    lines.append("")
    lines.append(f"Subject: {summary.get('subject', '')}")
    lines.append(f"Dates: {' '.join(summary.get('dates', []))}")
    lines.append(f"Sessions: {' '.join(summary.get('sessions', []))}")
    lines.append(f"Timestamp: {summary.get('timestamp', '')}")
    lines.append(f"Model path: {summary.get('model_path', '')}")
    lines.append(f"Checkpoint: {summary.get('checkpoint_path', '')}")
    lines.append(f"Lexicon path: {summary.get('lexicon_path', '')}")
    lines.append(f"Skip LM: {summary.get('skip_lm', False)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Test trials loaded: {summary['split'].get('test_trials_loaded', 0)}")
    lines.append(f"- PER: {_markdown_value(summary['test_metrics'].get('per'))}")
    lines.append(f"- WER LM OFF: {_markdown_value(summary['test_metrics'].get('wer_lm_off'))}")
    lines.append(f"- WER LM ON: {_markdown_value(summary['test_metrics'].get('wer_lm_on'))}")
    lines.append(f"- Word Accuracy LM OFF: {_markdown_value(summary['test_metrics'].get('word_accuracy_lm_off'))}")
    lines.append(f"- Word Accuracy LM ON: {_markdown_value(summary['test_metrics'].get('word_accuracy_lm_on'))}")
    lines.append("")
    lines.append("## Per-trial details")
    lines.append("")
    for idx, row in enumerate(detail_rows, start=1):
        lines.append(f"### Trial {idx:03d}")
        lines.append(f"- Date: {_markdown_value(row.get('date'))}")
        lines.append(f"- Session: {_markdown_value(row.get('session'))}")
        lines.append(f"- Raw session: {_markdown_value(row.get('raw_session'))}")
        lines.append(f"- Diagnostic session: {_markdown_value(row.get('paired_diagnostic_session'))}")
        lines.append(f"- Block: {_markdown_value(row.get('block'))}")
        lines.append(f"- Trial: {_markdown_value(row.get('trial'))}")
        lines.append(f"- Target sentence: {_markdown_value(row.get('target_sentence'))}")
        lines.append(f"- Target phonemes: {_markdown_value(row.get('target_phonemes'))}")
        lines.append(f"- Decoded phonemes: {_markdown_value(row.get('predicted_phonemes'))}")
        lines.append(f"- Trial PER: {_markdown_value(row.get('trial_PER'))}")
        lines.append(f"- LM OFF sentence: {_markdown_value(row.get('predicted_sentence_lm_off'))}")
        lines.append(f"- LM OFF WER: {_markdown_value(row.get('wer_lm_off'))}")
        lines.append(f"- LM ON sentence: {_markdown_value(row.get('predicted_sentence_lm_on'))}")
        lines.append(f"- LM ON WER: {_markdown_value(row.get('wer_lm_on'))}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_session_bundle(session: str, data_dir: Path, eval_type: str, csv_df: Optional[pd.DataFrame]):
    session_dir = data_dir / session
    eval_file = session_dir / f"data_{eval_type}.hdf5"
    if not eval_file.exists():
        return None

    data = load_h5py_file(str(eval_file), csv_df)
    metadata_path = session_dir / "metadata.json"
    metadata = load_session_metadata(metadata_path) if metadata_path.exists() else {}
    if metadata_path.exists():
        source_order = load_session_phoneme_order(metadata_path)
    else:
        n_classes = int(metadata.get("labels", {}).get("n_classes", 0))
        if n_classes and n_classes != len(LOGIT_TO_PHONEME):
            raise ValueError(f"Missing metadata.json for {session}; cannot map {n_classes}-class logits")
        source_order = LOGIT_TO_PHONEME

    return {
        "session": session,
        "data": data,
        "metadata": metadata,
        "source_order": source_order,
        "eval_file": str(eval_file),
    }


def _predict_logits(bundle, model, model_args, device):
    session = bundle["session"]
    data = bundle["data"]
    input_layer = model_args["dataset"]["sessions"].index(session)
    data["logits"] = []
    with tqdm(total=len(data["neural_features"]), desc=f"Predicting {session}", unit="trial") as pbar:
        for trial_idx in range(len(data["neural_features"])):
            neural_input = np.expand_dims(data["neural_features"][trial_idx], axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data["logits"].append(logits)
            pbar.update(1)


def _run_remote_lm(bundles, session_logit_orders, args):
    import redis

    r = redis.Redis(host="localhost", port=6379, db=0)
    r.flushall()

    remote_lm_input_stream = "remote_lm_input"
    remote_lm_output_partial_stream = "remote_lm_output_partial"
    remote_lm_output_final_stream = "remote_lm_output_final"

    remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
    remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
    remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)

    lm_sentences: List[str] = []
    with tqdm(total=sum(len(bundle["data"]["logits"]) for bundle in bundles), desc="Running remote language model", unit="trial") as pbar:
        for bundle in bundles:
            session = bundle["session"]
            for trial_idx in range(len(bundle["data"]["logits"])):
                official_logits = expand_logits_to_official_order(
                    bundle["data"]["logits"][trial_idx][0],
                    session_logit_orders[session],
                )
                logits = rearrange_speech_logits_pt(official_logits[None, ...] if official_logits.ndim == 2 else official_logits)[0]
                remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(
                    r, remote_lm_done_resetting_lastEntrySeen
                )
                remote_lm_output_partial_lastEntrySeen, _ = send_logits_to_remote_lm(
                    r,
                    remote_lm_input_stream,
                    remote_lm_output_partial_stream,
                    remote_lm_output_partial_lastEntrySeen,
                    logits,
                )
                remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                    r,
                    remote_lm_output_final_stream,
                    remote_lm_output_final_lastEntrySeen,
                )
                lm_sentences.append(lm_out["candidate_sentences"][0])
                pbar.update(1)
    return lm_sentences


def _build_summary(
    *,
    args,
    model_path: str,
    checkpoint_path: str,
    output_dir: Path,
    bundles,
    total_test_trials: int,
    split_totals: Dict[str, int],
    subject: str,
    dates: List[str],
    loaded_sessions: List[str],
    lm_sentences: Optional[List[str]],
    metrics: Dict[str, Optional[float]],
    session_summaries: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "subject": subject,
        "dates": dates,
        "sessions": loaded_sessions,
        "model": "electrode_aggregation",
        "eval_type": args.eval_type,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
        "data_dir": args.data_dir,
        "csv_path": args.csv_path,
        "lexicon_path": args.lexicon_path,
        "skip_lm": bool(args.skip_lm),
        "output_dir": str(output_dir),
        "output_prefix": args.output_prefix,
        "split": {
            "train_trials": split_totals.get("train", 0),
            "val_trials": split_totals.get("val", 0),
            "test_trials_configured": split_totals.get("test_configured", 0),
            "test_trials_loaded": total_test_trials,
        },
        "test_metrics": metrics,
        "session_summaries": session_summaries,
        "lm_trials_loaded": len(lm_sentences) if lm_sentences is not None else 0,
    }


def _parse_args():
    default_output_prefix = "baseline_rnn"
    parser = argparse.ArgumentParser(description="Evaluate a pretrained RNN model on the copy task dataset.")
    parser.add_argument("--model_path", type=str, default="../data/t15_pretrained_rnn_baseline")
    parser.add_argument("--data_dir", type=str, default="../data/hdf5_data_512")
    parser.add_argument("--eval_type", type=str, default="test", choices=["val", "test","train"])
    parser.add_argument("--csv_path", type=str, default="../data/t15_copyTaskData_description.csv")
    parser.add_argument("--gpu_number", type=int, default=1)
    parser.add_argument(
        "--lexicon_path",
        type=str,
        default=str(DEFAULT_LEXICON_PATH),
        help="50-word CMUdict-style lexicon used to report pre-LM greedy word accuracy.",
    )
    parser.add_argument("--skip_lm", action="store_true", help="Only run neural decoding; skip the remote LM.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for JSON/CSV/Markdown outputs.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=default_output_prefix,
        help="Prefix for output files, e.g. sub01_electrode.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = args.model_path
    data_dir = args.data_dir
    eval_type = args.eval_type
    output_dir = _resolve_output_dir(model_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_df = _load_csv_if_available(args.csv_path)
    model_args = load_model_args(Path(model_path) / "checkpoint" / "args.yaml")

    gpu_number = args.gpu_number
    if torch.cuda.is_available() and gpu_number >= 0:
        if gpu_number >= torch.cuda.device_count():
            raise ValueError(f"GPU number {gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}")
        device = torch.device(f"cuda:{gpu_number}")
        print(f"Using {device} for model inference.")
    else:
        if gpu_number >= 0:
            print(f"GPU number {gpu_number} requested but not available.")
        print("Using CPU for model inference.")
        device = torch.device("cpu")

    model = GRUDecoder(
        neural_dim=model_args["model"]["n_input_features"],
        n_units=model_args["model"]["n_units"],
        n_days=len(model_args["dataset"]["sessions"]),
        n_classes=model_args["dataset"]["n_classes"],
        rnn_dropout=model_args["model"]["rnn_dropout"],
        input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
        n_layers=model_args["model"]["n_layers"],
        patch_size=model_args["model"]["patch_size"],
        patch_stride=model_args["model"]["patch_stride"],
    )

    checkpoint_path = Path(model_path) / "checkpoint" / "best_checkpoint"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["model_state_dict"] = _clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    bundles = []
    loaded_sessions = []
    session_logit_orders = {}
    session_summaries = []

    for session in model_args["dataset"]["sessions"]:
        bundle = _load_session_bundle(session, Path(data_dir), eval_type, csv_df)
        if bundle is None:
            print(f"Skipping {session}: missing data_{eval_type}.hdf5")
            continue
        bundles.append(bundle)
        loaded_sessions.append(session)
        session_logit_orders[session] = bundle["source_order"]
        meta = bundle["metadata"]
        data = bundle["data"]
        session_summaries.append(
            {
                "session": session,
                "date": meta.get("date", data["date"][0] if data["date"] else None),
                "raw_session": meta.get("raw_session", data["raw_session"][0] if data["raw_session"] else None),
                "paired_diagnostic_session": meta.get("paired_diagnostic_session"),
                "n_train": meta.get("n_train", 0),
                "n_val": meta.get("n_val", 0),
                "n_test": meta.get("n_test", 0),
                "loaded_trials": len(data["neural_features"]),
                "eval_file": bundle["eval_file"],
            }
        )
        print(f"Loaded {len(data['neural_features'])} {eval_type} trials for session {session}.")

    if not bundles:
        raise SystemExit(f"No {eval_type} splits were found under {data_dir}.")

    total_test_trials = sum(len(bundle["data"]["neural_features"]) for bundle in bundles)
    print(f"Total number of {eval_type} trials: {total_test_trials}")

    phoneme_word_lexicon = None
    if args.lexicon_path is not None:
        lexicon_path = Path(args.lexicon_path)
        if lexicon_path.exists():
            phoneme_word_lexicon = load_phoneme_word_lexicon(str(lexicon_path))
            print(f"Loaded {len(phoneme_word_lexicon)} phoneme->word lexicon entries from {lexicon_path}")
        else:
            print(f"Lexicon not found at {lexicon_path}; LM-off word metrics will be skipped.")

    for bundle in bundles:
        _predict_logits(bundle, model, model_args, device)

    ordered_trials: List[Tuple[str, int]] = []
    for bundle in bundles:
        session = bundle["session"]
        data = bundle["data"]
        source_order = session_logit_orders[session]
        data["pred_phonemes"] = []
        data["pred_words_off"] = []
        data["true_phonemes"] = []
        data["true_words"] = []
        for trial_idx in range(len(data["logits"])):
            logits = data["logits"][trial_idx][0]
            pred_seq = _decode_trial_logits(
                logits,
                source_order,
                n_time_steps=data["n_time_steps"][trial_idx],
                patch_size=model_args["model"]["patch_size"],
                patch_stride=model_args["model"]["patch_stride"],
            )
            data["pred_phonemes"].append(pred_seq)
            data["pred_words_off"].append(_maybe_words_from_phonemes(pred_seq, phoneme_word_lexicon))
            true_seq = _flatten_true_phonemes(data["seq_class_ids"][trial_idx], data["seq_len"][trial_idx], source_order)
            data["true_phonemes"].append(true_seq)
            sentence_label = data["sentence_label"][trial_idx]
            data["true_words"].append(remove_punctuation(sentence_label).split() if sentence_label is not None else None)
            ordered_trials.append((session, trial_idx))
            block_num = data["block_num"][trial_idx]
            trial_num = data["trial_num"][trial_idx]
            print(f"Session: {session}, Block: {block_num}, Trial: {trial_num}")
            if true_seq is not None:
                print(f"Sentence label:      {sentence_label}")
                print(f"True sequence:       {' '.join(true_seq)}")
            print(f"Predicted Sequence:  {' '.join(pred_seq)}")
            if data["pred_words_off"][trial_idx] is not None:
                print(f"Pre-LM words:        {' '.join(data['pred_words_off'][trial_idx])}")
            print()

    lm_sentences = None
    if args.skip_lm:
        print("Skipping remote language model (--skip_lm).")
    else:
        lm_sentences = _run_remote_lm(bundles, session_logit_orders, args)

    detail_rows: List[Dict[str, object]] = []
    raw_phoneme_ed = 0
    raw_phoneme_total = 0
    pre_lm_word_ed = 0
    pre_lm_word_total = 0
    post_lm_word_ed = 0
    post_lm_word_total = 0

    lm_idx = 0
    for bundle in bundles:
        session = bundle["session"]
        data = bundle["data"]
        meta = bundle["metadata"]
        for trial_idx in range(len(data["logits"])):
            true_sentence = data["sentence_label"][trial_idx]
            true_seq = data["true_phonemes"][trial_idx]
            true_words = data["true_words"][trial_idx]
            pred_seq = data["pred_phonemes"][trial_idx]
            pred_words_off = data["pred_words_off"][trial_idx]
            lm_sentence = lm_sentences[lm_idx] if lm_sentences is not None else None
            lm_idx += 1
            lm_words_on = remove_punctuation(lm_sentence).split() if lm_sentence else None

            trial_per = None
            if true_seq is not None:
                trial_per = sequence_edit_distance(true_seq, pred_seq) / max(1, len(true_seq))
                raw_phoneme_ed += sequence_edit_distance(true_seq, pred_seq)
                raw_phoneme_total += len(true_seq)

            trial_wer_off = None
            if true_words is not None and pred_words_off is not None:
                trial_wer_off = sequence_edit_distance(true_words, pred_words_off) / max(1, len(true_words))
                pre_lm_word_ed += sequence_edit_distance(true_words, pred_words_off)
                pre_lm_word_total += len(true_words)

            trial_wer_on = None
            if true_words is not None and lm_words_on is not None:
                trial_wer_on = sequence_edit_distance(true_words, lm_words_on) / max(1, len(true_words))
                post_lm_word_ed += sequence_edit_distance(true_words, lm_words_on)
                post_lm_word_total += len(true_words)

            detail_rows.append(
                {
                    "subject": data["subject"][trial_idx],
                    "date": data["date"][trial_idx],
                    "session": session,
                    "raw_session": data["raw_session"][trial_idx],
                    "paired_diagnostic_session": data["paired_diagnostic_session"][trial_idx],
                    "paired_diagnostic_block_num": data["paired_diagnostic_block_num"][trial_idx],
                    "split": data["split"][trial_idx],
                    "corpus": data["corpus"][trial_idx],
                    "block": data["block_num"][trial_idx],
                    "trial": data["trial_num"][trial_idx],
                    "target_sentence": true_sentence,
                    "target_phonemes": " ".join(true_seq) if true_seq is not None else None,
                    "predicted_phonemes": " ".join(pred_seq),
                    "trial_PER": trial_per,
                    "predicted_sentence_lm_off": " ".join(pred_words_off) if pred_words_off is not None else None,
                    "wer_lm_off": trial_wer_off,
                    "predicted_sentence_lm_on": lm_sentence,
                    "wer_lm_on": trial_wer_on,
                    "num_target_phonemes": len(true_seq) if true_seq is not None else None,
                    "phoneme_edit_distance": sequence_edit_distance(true_seq, pred_seq) if true_seq is not None else None,
                    "num_target_words": len(true_words) if true_words is not None else None,
                    "word_edit_distance_lm_off": sequence_edit_distance(true_words, pred_words_off)
                    if true_words is not None and pred_words_off is not None
                    else None,
                    "word_edit_distance_lm_on": sequence_edit_distance(true_words, lm_words_on)
                    if true_words is not None and lm_words_on is not None
                    else None,
                    "n_time_steps": data["n_time_steps"][trial_idx],
                    "seq_len": data["seq_len"][trial_idx],
                }
            )

    metrics = {
        "per": raw_phoneme_ed / raw_phoneme_total if raw_phoneme_total > 0 else None,
        "wer_lm_off": pre_lm_word_ed / pre_lm_word_total if pre_lm_word_total > 0 else None,
        "wer_lm_on": post_lm_word_ed / post_lm_word_total if post_lm_word_total > 0 else None,
        "word_accuracy_lm_off": 1 - (pre_lm_word_ed / pre_lm_word_total) if pre_lm_word_total > 0 else None,
        "word_accuracy_lm_on": 1 - (post_lm_word_ed / post_lm_word_total) if post_lm_word_total > 0 else None,
    }

    print()
    if metrics["per"] is not None:
        print(f"Raw phoneme PER: {metrics['per']:.4f} ({raw_phoneme_ed}/{raw_phoneme_total})")
    if metrics["wer_lm_off"] is not None:
        print(f"LM OFF WER: {metrics['wer_lm_off']:.4f} ({pre_lm_word_ed}/{pre_lm_word_total})")
    if metrics["wer_lm_on"] is not None:
        print(f"LM ON WER: {metrics['wer_lm_on']:.4f} ({post_lm_word_ed}/{post_lm_word_total})")

    split_totals = {
        "train": sum(int(bundle["metadata"].get("n_train", 0)) for bundle in bundles),
        "val": sum(int(bundle["metadata"].get("n_val", 0)) for bundle in bundles),
        "test_configured": sum(int(bundle["metadata"].get("n_test", 0)) for bundle in bundles),
    }
    dates = sorted({row["date"] for row in detail_rows if row.get("date")})
    subject = next((row["subject"] for row in detail_rows if row.get("subject")), "sub-01")

    summary = _build_summary(
        args=args,
        model_path=model_path,
        checkpoint_path=str(checkpoint_path),
        output_dir=output_dir,
        bundles=bundles,
        total_test_trials=total_test_trials,
        split_totals=split_totals,
        subject=subject,
        dates=[date for date in dates if date],
        loaded_sessions=loaded_sessions,
        lm_sentences=lm_sentences,
        metrics=metrics,
        session_summaries=session_summaries,
    )

    metrics_path = output_dir / f"{args.output_prefix}_{eval_type}_metrics.json"
    detail_path = output_dir / f"{args.output_prefix}_{eval_type}_trial_details.csv"
    report_path = output_dir / f"{args.output_prefix}_{eval_type}_report.md"

    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_trial_details(detail_rows, detail_path)
    report_path.write_text(_build_report(summary, detail_rows), encoding="utf-8")

    print(f"Wrote summary metrics to {metrics_path}")
    print(f"Wrote trial details to {detail_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
