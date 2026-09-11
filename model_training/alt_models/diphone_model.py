"""`DiphoneGRUDecoder` -- the NEJM GRU with a diphone CTC head.

This is a subclass of `rnn_model.GRUDecoder`, so it inherits -- verbatim -- the
day-specific identity-initialised input layers, the patch embedding, the
5-layer unidirectional GRU, `h0`, and every weight initialiser.  The only
difference is the width of the output head, and a marginalisation step that
folds the diphone distribution back down to a phoneme distribution.

Output contract
---------------
`forward(...)` returns exactly what `GRUDecoder.forward(...)` returns: a
`(batch, time, n_classes)` tensor of **phoneme** logits using the baseline class
ordering.  That is deliberate -- it is what `runSingleDecodingStep`, the greedy
CTC decode in `BrainToTextDecoder_Trainer.validation`, and the language-model
bridge all expect, so none of them need to know this model exists.

`forward_diphone(...)` returns the raw `(batch, time, 1 + n_symbols**2)` diphone
logits.  Only the training loop calls it.

Parameter-name contract
-----------------------
Because `super().__init__` builds everything, `named_parameters()` yields the
same names as the baseline (`day_weights.*`, `day_biases.*`, `gru.*`, `out.*`,
`h0`).  `rnn_trainer.create_optimizer` splits parameters by *substring* match on
those names (`'day_'`, `'gru.bias'`, `'out.bias'`), and the cosine scheduler
branches on the resulting param-group count.  Those all keep working untouched:
the diphone model still produces the same three param groups, including the
dedicated day-layer learning rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from rnn_model import GRUDecoder  # noqa: E402  (needs the sys.path tweak above)

from diphone import marginalize_diphone_logits, n_diphone_classes, n_symbols_for  # noqa: E402


class DiphoneGRUDecoder(GRUDecoder):
    """Baseline GRU whose head predicts phoneme transitions instead of phonemes.

    Constructed with the *baseline* class count (35 for the English copy task);
    the wider diphone head is derived internally.  Keeping the constructor
    signature identical is what lets `evaluate_model_extended.py` instantiate
    this class without a single edit.
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
        self.n_phoneme_classes = int(n_classes)
        self.n_symbols = n_symbols_for(self.n_phoneme_classes)
        self.n_diphone_classes = n_diphone_classes(self.n_phoneme_classes)

        super().__init__(
            neural_dim=neural_dim,
            n_units=n_units,
            n_days=n_days,
            n_classes=self.n_diphone_classes,
            rnn_dropout=rnn_dropout,
            input_dropout=input_dropout,
            n_layers=n_layers,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )

    # ------------------------------------------------------------------
    # raw diphone head -- training only
    # ------------------------------------------------------------------
    def forward_diphone(self, x, day_idx, states=None, return_state=False):
        """Return the un-marginalised `(B, T, n_diphone_classes)` diphone logits."""
        return super().forward(x, day_idx, states=states, return_state=return_state)

    # ------------------------------------------------------------------
    # marginalised phoneme head -- the baseline-compatible interface
    # ------------------------------------------------------------------
    def forward(self, x, day_idx, states=None, return_state=False):
        """Return `(B, T, n_classes)` *phoneme* logits, exactly like the baseline.

        Same signature, same shapes, same semantics as `GRUDecoder.forward`, so
        this class is usable anywhere the baseline is -- including through a
        `torch.compile` wrapper and through the language-model decode path.
        """
        logits, hidden_states = super().forward(
            x, day_idx, states=states, return_state=True
        )
        phoneme_logits = marginalize_diphone_logits(logits, self.n_phoneme_classes)

        if return_state:
            return phoneme_logits, hidden_states
        return phoneme_logits
