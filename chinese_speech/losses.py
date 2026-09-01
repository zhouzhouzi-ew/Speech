from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def _ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    return loss_fn(log_probs, targets, input_lengths, target_lengths)


def dual_stream_ctc_loss(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    tone_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    syllable_loss = _ctc_loss(
        model_output["syllable_logits"],
        batch["seq_syllable_ids"],
        batch["n_time_steps"],
        batch["syllable_seq_lens"],
    )
    tone_loss = _ctc_loss(
        model_output["tone_logits"],
        batch["seq_tone_ids"],
        batch["n_time_steps"],
        batch["tone_seq_lens"],
    )
    loss = syllable_loss + tone_weight * tone_loss
    return loss, {
        "syllable_loss": float(syllable_loss.detach().cpu()),
        "tone_loss": float(tone_loss.detach().cpu()),
        "tone_weight": float(tone_weight),
    }
