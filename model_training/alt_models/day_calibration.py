"""Stronger day-specific input calibration: a "hammer" and a "scalpel" with a learned gate.

After the NHS day-calibration module (Olak et al., ICLR 2026).  The baseline's
day layer is already the "hammer" -- one global affine map per day, initialised
to the identity:

    X_h = X W_d + 1 b_d                    (global affine, day-specific)

The addition is a second, elementwise branch -- the "scalpel" -- which can
rescale and shift individual channels, something a full affine map *can*
express but is hard to learn when the identity initialisation dominates:

    X_s = X . gamma_d + beta_d             (FiLM-style, day-specific)

The two are blended per day by a learned gate, and softsign is applied last,
exactly as in the baseline:

    g_d   = sigmoid(w_g . e_d)             (w_g is one logit per day)
    X_out = softsign( g_d * X_s + (1 - g_d) * X_h )

Starting point is the baseline
------------------------------
`W_d = I`, `b_d = 0`, `gamma_d = 1`, `beta_d = 0` means `X_h == X_s == X`, so
whichever way the gate is initialised the whole block collapses to
`softsign(X)` -- precisely what `GRUDecoder` does at step 0.  The gate logits
are initialised to 0 (g = 0.5, a balanced blend).  So a day-calibrated model
starts out functionally identical to the baseline and can only move away from it
if the data says so.

Reported ablation deltas (their setting, not ours): about +2.7 PER / +2.6 WER
over a plain linear day transform, and +3.6 PER / +4.4 WER over no day
calibration at all.

Caveat
------
With a single day in `dataset.sessions` there is nothing to calibrate *between*:
`n_days == 1` makes the gate and all four parameter sets constant, and the block
becomes a strictly weaker version of the baseline's day layer (it can still
learn to leave the input alone, but it has no cross-day signal to exploit).  The
benefit is realised only when `sessions` lists several days -- which is what
`make_all_day_config.py` is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from rnn_model import GRUDecoder  # noqa: E402  (needs the sys.path tweak above)


class DayCalibratedGRUDecoder(GRUDecoder):
    """`GRUDecoder` with a hammer + scalpel + gate day layer.

    Subclasses the baseline rather than replacing it, so the GRU stack, patch
    embedding, `h0`, initialisers, dropout and the `day_layer_activation` /
    `day_layer_dropout` modules are all inherited unchanged.  Only the day block
    inside `forward` and the constructor differ.
    """

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=5,
        patch_size=0,
        patch_stride=0,
    ):
        super().__init__(
            neural_dim=neural_dim,
            n_units=n_units,
            n_days=n_days,
            n_classes=n_classes,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )

        # The scalpel branch. gamma starts at 1 and beta at 0, so it begins as
        # the identity -- matching the hammer's identity initialisation.
        self.day_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_offsets = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        # One gate logit per day; 0 means sigmoid(0) = 0.5, a balanced blend.
        self.day_gate_logits = nn.Parameter(torch.zeros(self.n_days))

    # ------------------------------------------------------------------
    def _apply_day_calibration(self, x, day_idx):
        """Blend the hammer and scalpel branches, then softsign and dropout.

        Parameter names all start with `day_`, so `rnn_trainer.create_optimizer`
        routes every one of them into the day-layer param group -- which is what
        gives them `lr_max_day` and `weight_decay_day` instead of the default
        learning rate.
        """
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        day_scales = torch.stack([self.day_scales[i] for i in day_idx], dim=0)
        day_offsets = torch.cat([self.day_offsets[i] for i in day_idx], dim=0).unsqueeze(1)

        # hammer: global affine, exactly the baseline's day layer
        hammer = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases

        # scalpel: elementwise rescale + shift
        scalpel = x * day_scales + day_offsets

        # per-day gate. view(-1, 1, 1) broadcasts over time and channels.
        gate = torch.sigmoid(self.day_gate_logits[day_idx]).view(-1, 1, 1)
        x = gate * scalpel + (1.0 - gate) * hammer

        # Same order as the baseline day layer: activate, then drop out.
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)
        return x

    # ------------------------------------------------------------------
    def forward(self, x, day_idx, states=None, return_state=False):
        """Identical to `GRUDecoder.forward` except for the day block.

        Diff against the parent: the four lines that build and apply the
        identity-initialised day affine are replaced by a single call to
        `_apply_day_calibration`.  The patch embedding, hidden-state
        initialisation, GRU and head are copied verbatim.
        """
        x = self._apply_day_calibration(x, day_idx)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0:

            x = x.unsqueeze(1)                      # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)               # [batches, feature_dim, 1, timesteps]

            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)  # [batches, feature_dim, 1, num_patches, patch_size]

            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(2)           # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(0, 2, 3, 1)  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(self.n_layers, x.shape[0], self.n_units).contiguous()

        # Pass input through RNN
        output, hidden_states = self.gru(x, states)

        # Compute logits
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits
