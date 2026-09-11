"""`DiphoneTrainer` -- the NEJM trainer with diphone CTC targets.

Inherits everything from `rnn_trainer.BrainToTextDecoder_Trainer`: dataset and
dataloader construction, the three-way AdamW param groups (bias / day layer /
everything else), the cosine LR schedule with warmup, gradient clipping, the
checkpoint-on-best-val-PER rule, `val_metrics.pkl`, early stopping and logging.
None of that is re-implemented here.

Two things change, and only two:

1. The model is built as a `DiphoneGRUDecoder` instead of a `GRUDecoder`.
   `rnn_trainer` binds the class at import time (`from rnn_model import
   GRUDecoder`), so the swap is a scoped replacement of that module attribute
   for the duration of `super().__init__()` and nothing else.

2. CTC targets are diphone ids rather than phoneme ids, so `train()` is
   overridden.  See the block comment on `train` for the exact diff against
   `BrainToTextDecoder_Trainer.train`.

`validation()` is **not** overridden.  It calls `self.model(...)`, which for a
`DiphoneGRUDecoder` already returns marginalised phoneme logits with the
baseline class ordering, so the inherited greedy decode, the PER accounting and
the `metrics['logits']` payload all keep their original meaning.  That is why
`save_val_logits: true` still produces logits the language model can consume.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import rnn_trainer  # noqa: E402  (needs the sys.path tweak above)
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

from diphone import batch_phonemes_to_diphones, n_symbols_for  # noqa: E402
from diphone_model import DiphoneGRUDecoder  # noqa: E402


@contextmanager
def _gru_class_swapped(cls):
    """Temporarily bind `rnn_trainer.GRUDecoder` to `cls`.

    `rnn_trainer` does `from rnn_model import GRUDecoder`, so the name lives in
    that module's namespace and this is the one lever needed to make the
    inherited `__init__` build a different class -- without editing a single
    line of the original file.  The original binding is restored on the way out.
    """
    original = rnn_trainer.GRUDecoder
    rnn_trainer.GRUDecoder = cls
    try:
        yield
    finally:
        rnn_trainer.GRUDecoder = original


def resolve_diphone_options(args) -> dict:
    """Read the optional `diphone:` block from the config, with safe defaults."""
    block = args.get("diphone", None) or {}
    close_with_sil = bool(block.get("close_with_sil", False))
    sil_phoneme_id = block.get("sil_phoneme_id", None)
    return {
        "close_with_sil": close_with_sil,
        "sil_phoneme_id": None if sil_phoneme_id is None else int(sil_phoneme_id),
    }


class DiphoneTrainer(BrainToTextDecoder_Trainer):
    """Trains a `DiphoneGRUDecoder` on diphone CTC targets."""

    def __init__(self, args):
        options = resolve_diphone_options(args)

        with _gru_class_swapped(DiphoneGRUDecoder):
            super().__init__(args)

        if not isinstance(self.model, DiphoneGRUDecoder):
            # torch.compile wraps the module, so unwrap before the type check.
            inner = getattr(self.model, "_orig_mod", self.model)
            if not isinstance(inner, DiphoneGRUDecoder):
                raise TypeError(
                    "DiphoneTrainer expected a DiphoneGRUDecoder but the trainer built "
                    f"{type(inner).__name__}. Check that `alt_models` is importable and "
                    "that nothing else rebinds rnn_trainer.GRUDecoder."
                )

        inner = getattr(self.model, "_orig_mod", self.model)
        self.n_symbols = inner.n_symbols
        self.close_with_sil = options["close_with_sil"]
        # `<sil>` is the last class in the 35-class English set (id 34), which
        # equals n_symbols. Only used when close_with_sil is enabled.
        self.sil_phoneme_id = (
            options["sil_phoneme_id"]
            if options["sil_phoneme_id"] is not None
            else self.n_symbols
        )

        self.logger.info(
            f"Diphone CTC targets: {inner.n_phoneme_classes} phoneme classes -> "
            f"{inner.n_diphone_classes} diphone classes "
            f"(n_symbols={inner.n_symbols}, close_with_sil={self.close_with_sil})"
        )

    # ------------------------------------------------------------------
    def to_diphone_targets(self, labels, phone_seq_lens):
        """`(B, Lmax)` phoneme ids -> `(B, Dmax)` diphone ids (0-padded)."""
        return batch_phonemes_to_diphones(
            labels=labels,
            lengths=phone_seq_lens,
            n_symbols=self.n_symbols,
            close_with_sil=self.close_with_sil,
            sil_phoneme_id=self.sil_phoneme_id,
        )

    # ------------------------------------------------------------------
    def train(self):
        """Identical to `BrainToTextDecoder_Trainer.train` except for the loss.

        Diff against the parent method -- two lines:

            logits = self.model(features, day_indicies)
        ->  logits = self.model.forward_diphone(features, day_indicies)

            targets = labels, target_lengths = phone_seq_lens
        ->  targets = diphone_labels, target_lengths = diphone_seq_lens

        The diphone label tensors are built in place from the same batch fields
        the parent reads.  Every other statement -- augmentation, adjusted_lens,
        grad clipping, the LR scheduler step, the val/checkpoint/early-stop
        block -- is copied verbatim and must stay in sync with the parent.
        """
        self.model.train()

        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []

        val_steps_since_improvement = 0

        save_best_checkpoint = self.args.get("save_best_checkpoint", True)
        early_stopping = self.args.get("early_stopping", True)
        early_stopping_val_steps = self.args["early_stopping_val_steps"]

        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):

            self.model.train()
            self.optimizer.zero_grad()

            start_time = time.time()

            features = batch["input_features"].to(self.device)
            labels = batch["seq_class_ids"].to(self.device)
            n_time_steps = batch["n_time_steps"].to(self.device)
            phone_seq_lens = batch["phone_seq_lens"].to(self.device)
            day_indicies = batch["day_indicies"].to(self.device)

            # ---- the diphone target swap -------------------------------
            diphone_labels, diphone_seq_lens = self.to_diphone_targets(
                labels, phone_seq_lens
            )
            # ------------------------------------------------------------

            with torch.autocast(
                device_type="cuda", enabled=self.args["use_amp"], dtype=torch.bfloat16
            ):

                features, n_time_steps = self.transform_data(
                    features, n_time_steps, "train"
                )

                adjusted_lens = (
                    (n_time_steps - self.args["model"]["patch_size"])
                    / self.args["model"]["patch_stride"]
                    + 1
                ).to(torch.int32)

                # raw diphone logits, NOT the marginalised phoneme logits
                logits = self.model.forward_diphone(features, day_indicies)

                loss = self.ctc_loss(
                    log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets=diphone_labels,
                    input_lengths=adjusted_lens,
                    target_lengths=diphone_seq_lens,
                )

                loss = torch.mean(loss)

            loss.backward()

            if self.args["grad_norm_clip_value"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.args["grad_norm_clip_value"],
                    error_if_nonfinite=True,
                    foreach=True,
                )

            self.optimizer.step()
            self.learning_rate_scheduler.step()

            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            if i % self.args["batches_per_train_log"] == 0:
                self.logger.info(
                    f"Train batch {i}: "
                    + f"loss: {(loss.detach().item()):.2f} "
                    + f"grad norm: {grad_norm:.2f} "
                    f"time: {train_step_duration:.3f}"
                )

            if i % self.args["batches_per_val_step"] == 0 or i == (
                self.args["num_training_batches"] - 1
            ):
                self.logger.info(f"Running test after training batch: {i}")

                start_time = time.time()
                # Inherited: DiphoneGRUDecoder.forward returns marginalised
                # phoneme logits, so PER here is comparable to the baseline's.
                val_metrics = self.validation(
                    loader=self.val_loader,
                    return_logits=self.args["save_val_logits"],
                    return_data=self.args["save_val_data"],
                )
                val_step_duration = time.time() - start_time

                self.logger.info(
                    f"Val batch {i}: "
                    + f"PER (avg): {val_metrics['avg_PER']:.4f} "
                    + f"CTC Loss (avg): {val_metrics['avg_loss']:.4f} "
                    f"time: {val_step_duration:.3f}"
                )

                if self.args["log_individual_day_val_PER"]:
                    for day in val_metrics["day_PERs"].keys():
                        self.logger.info(
                            f"{self.args['dataset']['sessions'][day]} val PER: "
                            f"{val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}"
                        )

                val_PERs.append(val_metrics["avg_PER"])
                val_losses.append(val_metrics["avg_loss"])
                val_results.append(val_metrics)

                new_best = False
                if val_metrics["avg_PER"] < self.best_val_PER:
                    self.logger.info(
                        f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}"
                    )
                    self.best_val_PER = val_metrics["avg_PER"]
                    self.best_val_loss = val_metrics["avg_loss"]
                    new_best = True
                elif val_metrics["avg_PER"] == self.best_val_PER and (
                    val_metrics["avg_loss"] < self.best_val_loss
                ):
                    self.logger.info(
                        f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}"
                    )
                    self.best_val_loss = val_metrics["avg_loss"]
                    new_best = True

                if new_best:

                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(
                            f'{self.args["checkpoint_dir"]}/best_checkpoint',
                            self.best_val_PER,
                            self.best_val_loss,
                        )

                    if self.args["save_val_metrics"]:
                        import pickle

                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', "wb") as f:
                            pickle.dump(val_metrics, f)

                    val_steps_since_improvement = 0

                else:
                    val_steps_since_improvement += 1

                if self.args["save_all_val_steps"]:
                    self.save_model_checkpoint(
                        f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}',
                        val_metrics["avg_PER"],
                    )

                if early_stopping and (
                    val_steps_since_improvement >= early_stopping_val_steps
                ):
                    self.logger.info(
                        f"Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}"
                    )
                    break

        training_duration = time.time() - train_start_time

        self.logger.info(f"Best avg val PER achieved: {self.best_val_PER:.5f}")
        self.logger.info(f"Total training time: {(training_duration / 60):.2f} minutes")

        if self.args["save_final_model"]:
            self.save_model_checkpoint(
                f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1]
            )

        train_stats = {}
        train_stats["train_losses"] = train_losses
        train_stats["val_losses"] = val_losses
        train_stats["val_PERs"] = val_PERs
        train_stats["val_metrics"] = val_results

        return train_stats
