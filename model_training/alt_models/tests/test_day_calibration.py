"""Tests for the hammer+scalpel day calibrator and the model registry.

The load-bearing claim is `test_identity_at_init`: because `W_d = I`, `b_d = 0`,
`gamma_d = 1`, `beta_d = 0`, the whole calibrator collapses to `softsign(X)` --
which is exactly what the baseline day layer does at step 0.  A day-calibrated
model therefore starts out functionally identical to the baseline and can only
diverge if the data pulls it there.  If that ever stops being true, this variant
is no longer a safe drop-in and the test should stop you.

    cd Speech/model_training
    python alt_models/tests/test_day_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ALT = _HERE.parent
_PARENT = _ALT.parent
for _p in (str(_ALT), str(_PARENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from day_calibration import DayCalibratedGRUDecoder  # noqa: E402
from diphone import marginalize_diphone_logits  # noqa: E402
from diphone_model import (  # noqa: E402
    DiphoneDayCalibratedGRUDecoder,
    DiphoneGRUDecoder,
)
from registry import (  # noqa: E402
    resolve_day_calibration,
    resolve_model_class,
    resolve_use_diphone,
)
from rnn_model import GRUDecoder  # noqa: E402

N_CLASSES = 35
N_DIPHONES = 1157
N_DAYS = 3


def _build(cls, n_days=N_DAYS, n_classes=N_CLASSES):
    torch.manual_seed(0)
    return cls(
        neural_dim=512,
        n_units=24,
        n_days=n_days,
        n_classes=n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=2,
        patch_size=14,
        patch_stride=4,
    )


def _adopt_baseline_weights(calibrated, baseline):
    """Copy every baseline parameter into the calibrated model.

    Leaves the calibrator-only parameters (`day_scales`, `day_offsets`,
    `day_gate_logits`) at their initialisation, which is precisely the state the
    identity-at-init claim is about.
    """
    missing, unexpected = calibrated.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected, f"calibrated model has no slot for: {unexpected}"
    assert set(missing) == {"day_scales", "day_offsets", "day_gate_logits"} or set(
        missing
    ) == {
        "day_scales.%d" % i for i in range(N_DAYS)
    } | {
        "day_offsets.%d" % i for i in range(N_DAYS)
    } | {
        "day_gate_logits"
    }, f"unexpected parameters left at init: {missing}"


# ---------------------------------------------------------------------------
# identity at initialisation
# ---------------------------------------------------------------------------
def test_identity_at_init():
    base = _build(GRUDecoder)
    cal = _build(DayCalibratedGRUDecoder)
    _adopt_baseline_weights(cal, base)
    base.eval()
    cal.eval()

    x = torch.randn(4, 60, 512)
    day = torch.tensor([0, 1, 2, 0])

    with torch.no_grad():
        out_base = base(x, day)
        out_cal = cal(x, day)

    assert torch.allclose(out_base, out_cal, atol=1e-5), (
        "day calibration is not a no-op at init; "
        f"max abs diff {(out_base - out_cal).abs().max().item():.3e}"
    )


def test_identity_at_init_for_the_diphone_variant():
    """Same claim, one level up.

    The diphone head is 1157-wide while the baseline is 35-wide, so the
    state-dict copy used by `test_identity_at_init` cannot work here.  The claim
    is instead asserted where it actually lives: the calibrator must collapse to
    plain `softsign(X)` at init, which is exactly what the baseline day layer
    computes.
    """
    cal = _build(DiphoneDayCalibratedGRUDecoder)
    cal.eval()

    x = torch.randn(2, 60, 512)
    day = torch.tensor([0, 1])

    with torch.no_grad():
        transformed = cal._apply_day_calibration(x, day)
        expected = cal.day_layer_activation(x)
        raw = cal.forward_diphone(x, day)
        manual = marginalize_diphone_logits(raw, N_CLASSES)

    assert torch.allclose(transformed, expected, atol=1e-5), (
        "the calibrator is not a no-op at init: "
        f"max abs diff {(transformed - expected).abs().max().item():.3e}"
    )
    assert manual.shape[-1] == N_CLASSES


# ---------------------------------------------------------------------------
# the gate actually does something
# ---------------------------------------------------------------------------
def test_gate_closed_reproduces_the_baseline_and_open_does_not():
    base = _build(GRUDecoder)
    cal = _build(DayCalibratedGRUDecoder)
    _adopt_baseline_weights(cal, base)
    base.eval()
    cal.eval()

    x = torch.randn(4, 60, 512)
    day = torch.tensor([0, 1, 2, 0])

    with torch.no_grad():
        out_base = base(x, day)

        # gate ~ 0 -> hammer only, i.e. the baseline day layer
        cal.day_gate_logits.fill_(-20.0)
        out_hammer = cal(x, day)

        # gate ~ 1 and a scalpel that is no longer the identity -> not the baseline
        cal.day_gate_logits.fill_(20.0)
        for scale in cal.day_scales:
            scale.fill_(2.0)
        out_scalpel = cal(x, day)

    assert torch.allclose(out_hammer, out_base, atol=1e-5), (
        "with the gate closed the calibrator must reproduce the baseline day layer"
    )
    assert not torch.allclose(out_scalpel, out_base, atol=1e-3), (
        "with the gate open and gamma=2 the output must differ from the baseline"
    )


def test_gate_receives_gradient():
    cal = _build(DayCalibratedGRUDecoder)
    cal.train()
    x = torch.randn(2, 40, 512)
    day = torch.tensor([0, 1])

    cal(x, day).sum().backward()
    assert cal.day_gate_logits.grad is not None
    assert torch.isfinite(cal.day_gate_logits.grad).all()
    assert cal.day_scales[0].grad is not None
    assert cal.day_offsets[0].grad is not None


# ---------------------------------------------------------------------------
# naming / optimizer contract
# ---------------------------------------------------------------------------
def test_every_new_parameter_is_in_the_day_param_group():
    """rnn_trainer splits on `'day_' in name`; all four sets plus the gate must match."""
    model = _build(DayCalibratedGRUDecoder)
    named = dict(model.named_parameters())

    for expected in (
        "day_weights.0",
        "day_biases.0",
        "day_scales.0",
        "day_offsets.0",
        "day_gate_logits",
    ):
        assert expected in named, f"{expected} missing from {sorted(named)}"
        assert "day_" in expected

    day_params = [n for n, _ in model.named_parameters() if "day_" in n]
    # 3 days x (weights, biases, scales, offsets) + the gate vector
    assert len(day_params) == N_DAYS * 4 + 1

    other = [
        n
        for n, _ in model.named_parameters()
        if "day_" not in n and "gru.bias" not in n and "out.bias" not in n
    ]
    assert "day_gate_logits" not in other, "the gate must not fall into the 'other' group"


def test_baseline_parameter_names_are_unchanged():
    """The calibrator only *adds* names; it must not rename anything."""
    base = {n for n, _ in _build(GRUDecoder).named_parameters()}
    cal = {n for n, _ in _build(DayCalibratedGRUDecoder).named_parameters()}
    assert base <= cal, f"calibrator dropped baseline parameters: {sorted(base - cal)}"
    assert cal - base == {
        "day_scales.0",
        "day_scales.1",
        "day_scales.2",
        "day_offsets.0",
        "day_offsets.1",
        "day_offsets.2",
        "day_gate_logits",
    }


def test_diphone_day_calibrated_keeps_three_param_groups():
    model = _build(DiphoneDayCalibratedGRUDecoder)
    named = list(model.named_parameters())

    bias = [p for n, p in named if "gru.bias" in n or "out.bias" in n]
    day = [p for n, p in named if "day_" in n]
    other = [p for n, p in named if "day_" not in n and "gru.bias" not in n and "out.bias" not in n]

    assert len(day) == N_DAYS * 4 + 1
    assert len(bias) == 1 + 2 + 2
    assert len(bias) + len(day) + len(other) == len(named)


# ---------------------------------------------------------------------------
# head contract for the combined model
# ---------------------------------------------------------------------------
def test_combined_model_keeps_both_contracts():
    model = _build(DiphoneDayCalibratedGRUDecoder)
    model.eval()
    x = torch.randn(2, 60, 512)
    day = torch.tensor([0, 1])

    assert model.out.out_features == N_DIPHONES
    assert model.n_phoneme_classes == N_CLASSES

    with torch.no_grad():
        phoneme_logits, hidden = model(x, day, return_state=True)
        raw = model.forward_diphone(x, day)
        manual = marginalize_diphone_logits(raw, N_CLASSES)

    assert phoneme_logits.shape == (2, (60 - 14) // 4 + 1, N_CLASSES)
    assert raw.shape == (2, (60 - 14) // 4 + 1, N_DIPHONES)
    assert torch.allclose(phoneme_logits, manual, atol=1e-5)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def _cfg(**model_block):
    return {"model": model_block}


def test_registry_defaults_to_diphone_with_baseline_day_layer():
    assert resolve_model_class(_cfg()) is DiphoneGRUDecoder
    assert resolve_use_diphone(_cfg()) is True
    assert resolve_day_calibration(_cfg()) == "baseline"


def test_registry_all_four_combinations():
    assert (
        resolve_model_class(_cfg(diphone_targets=False, day_calibration="baseline"))
        is GRUDecoder
    )
    assert (
        resolve_model_class(_cfg(diphone_targets=True, day_calibration="baseline"))
        is DiphoneGRUDecoder
    )
    assert (
        resolve_model_class(_cfg(diphone_targets=False, day_calibration="hammer_scalpel"))
        is DayCalibratedGRUDecoder
    )
    assert (
        resolve_model_class(_cfg(diphone_targets=True, day_calibration="hammer_scalpel"))
        is DiphoneDayCalibratedGRUDecoder
    )


def test_registry_rejects_an_unknown_mode():
    try:
        resolve_model_class(_cfg(day_calibration="magic"))
    except ValueError as exc:
        assert "day_calibration" in str(exc)
        return
    raise AssertionError("expected a ValueError for an unknown day_calibration mode")


def test_registry_accepts_an_already_extracted_model_block():
    """`evaluate_diphone.py` passes `args['model']`, the trainer passes the whole config."""
    assert resolve_model_class({"day_calibration": "hammer_scalpel"}) is DiphoneDayCalibratedGRUDecoder
    assert resolve_model_class(None) is DiphoneGRUDecoder


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
