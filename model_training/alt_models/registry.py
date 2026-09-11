"""Pick a model class from a config or checkpoint `args.yaml`.

Both pieces of this folder are optional and orthogonal, so four combinations
exist:

    diphone targets x day calibration
    --------------------------------
    no  x baseline          -> rnn_model.GRUDecoder        (the original)
    yes x baseline          -> DiphoneGRUDecoder
    no  x hammer_scalpel    -> DayCalibratedGRUDecoder
    yes x hammer_scalpel    -> DiphoneDayCalibratedGRUDecoder

The selection is driven by two keys in the `model:` block:

    model:
      diphone_targets: true          # default: true   (this package is diphone-first)
      day_calibration: baseline      # default: baseline

Both `DiphoneTrainer` and `evaluate_diphone.py` resolve the class through here,
and the evaluation path reads the values back out of the checkpoint's
`args.yaml`. That is what keeps a trained checkpoint's architecture recoverable
at evaluation time without a `--arch` flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parents[1]
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from rnn_model import GRUDecoder  # noqa: E402

from day_calibration import DayCalibratedGRUDecoder  # noqa: E402
from diphone_model import DiphoneDayCalibratedGRUDecoder, DiphoneGRUDecoder  # noqa: E402

DAY_CALIBRATION_MODES = ("baseline", "hammer_scalpel")


def _model_block(model_args):
    """Accept either a full config or an already-extracted `model:` block."""
    if model_args is None:
        return {}
    if "model" in model_args:
        return model_args["model"]
    return model_args


def resolve_day_calibration(model_args) -> str:
    block = _model_block(model_args)
    mode = block.get("day_calibration", "baseline")
    mode = "baseline" if mode is None else str(mode).strip().lower()
    if mode not in DAY_CALIBRATION_MODES:
        raise ValueError(
            f"Unknown model.day_calibration {mode!r}; expected one of {list(DAY_CALIBRATION_MODES)}"
        )
    return mode


def resolve_use_diphone(model_args) -> bool:
    block = _model_block(model_args)
    value = block.get("diphone_targets", True)
    return True if value is None else bool(value)


def resolve_model_class(model_args):
    """Return the `nn.Module` subclass to build for this configuration."""
    use_diphone = resolve_use_diphone(model_args)
    day_calibration = resolve_day_calibration(model_args)

    if use_diphone and day_calibration == "hammer_scalpel":
        return DiphoneDayCalibratedGRUDecoder
    if use_diphone:
        return DiphoneGRUDecoder
    if day_calibration == "hammer_scalpel":
        return DayCalibratedGRUDecoder
    return GRUDecoder


def describe_model_class(model_args) -> str:
    """Human-readable one-liner for logs."""
    cls = resolve_model_class(model_args)
    return f"{cls.__name__} (diphone_targets={resolve_use_diphone(model_args)}, day_calibration={resolve_day_calibration(model_args)})"
