"""Correctness tests for the diphone CTC variant.

Run from `Speech/model_training`:

    python -m pytest alt_models/tests/test_diphone.py -v
    # or, without pytest:
    python alt_models/tests/test_diphone.py

The tests that matter most:

* `test_marginalisation_matches_brute_force` -- the whole approach rests on
  `softmax(marginalize(D))` being exactly `0.5*(sum_j P[k,j] + sum_i P[i,k])`.
  Verified against a materialised brute-force computation.
* `test_parameter_names_match_baseline` -- `rnn_trainer.create_optimizer`
  splits parameters by *substring* match on `named_parameters()` (`'day_'`,
  `'gru.bias'`, `'out.bias'`) and the cosine scheduler branches on the
  resulting param-group count. If the diphone model renames anything, training
  silently loses its dedicated day-layer LR. This test pins that contract.
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

from diphone import (  # noqa: E402
    batch_phonemes_to_diphones,
    diphone_id,
    encode_phoneme_sequence,
    marginalize_diphone_logits,
    n_diphone_classes,
    n_symbols_for,
)
from diphone_model import DiphoneGRUDecoder  # noqa: E402
from rnn_model import GRUDecoder  # noqa: E402

N_CLASSES = 35
N_SYMBOLS = 34
N_DIPHONES = 1 + N_SYMBOLS * N_SYMBOLS  # 1157


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def test_class_counts():
    assert n_symbols_for(N_CLASSES) == N_SYMBOLS
    assert n_diphone_classes(N_CLASSES) == N_DIPHONES == 1157


def test_diphone_id_layout():
    # id 0 is reserved for the CTC blank.
    assert diphone_id(0, 0, N_SYMBOLS) == 1
    assert diphone_id(0, 1, N_SYMBOLS) == 2
    assert diphone_id(1, 0, N_SYMBOLS) == 1 + N_SYMBOLS
    # the very last diphone class
    assert diphone_id(33, 33, N_SYMBOLS) == N_DIPHONES - 1


def test_encode_matches_reference_definition():
    seq = [1, 2, 3, 5]
    assert encode_phoneme_sequence(seq, N_SYMBOLS) == [
        diphone_id(0, 1, N_SYMBOLS),
        diphone_id(1, 2, N_SYMBOLS),
        diphone_id(2, 4, N_SYMBOLS),
    ]
    assert len(encode_phoneme_sequence(seq, N_SYMBOLS)) == len(seq) - 1


def test_encode_close_with_sil_uses_baseline_target_length():
    seq = [1, 2, 3]
    closed = encode_phoneme_sequence(seq, N_SYMBOLS, close_with_sil=True)
    assert len(closed) == len(seq)  # same target length the monophone run needed
    assert closed[-1] == diphone_id(2, N_SYMBOLS - 1, N_SYMBOLS)  # (last phoneme, <sil>)


def test_batch_conversion_matches_scalar_helper():
    labels = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],
            [7, 7, 0, 0, 0, 0],
            [5, 6, 7, 8, 9, 10],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 2, 6], dtype=torch.long)

    got, got_lens = batch_phonemes_to_diphones(labels, lengths, N_SYMBOLS)
    assert got_lens.tolist() == [3, 1, 5]
    assert got.shape == (3, 5)

    for row, (lab, ln) in enumerate(zip(labels, lengths)):
        expected = encode_phoneme_sequence(lab[: int(ln)].tolist(), N_SYMBOLS)
        assert got[row, : len(expected)].tolist() == expected, f"row {row}"
        # everything past the true length must be the CTC blank
        assert (got[row, len(expected) :] == 0).all(), f"row {row} padding"


def test_batch_conversion_close_with_sil_target_length_equals_input():
    labels = torch.tensor([[1, 2, 3, 4, 0], [9, 5, 0, 0, 0]], dtype=torch.long)
    lengths = torch.tensor([4, 2], dtype=torch.long)

    got, got_lens = batch_phonemes_to_diphones(
        labels, lengths, N_SYMBOLS, close_with_sil=True
    )
    assert got_lens.tolist() == [4, 2]
    assert got.shape[1] == labels.shape[1]
    for row, (lab, ln) in enumerate(zip(labels, lengths)):
        expected = encode_phoneme_sequence(
            lab[: int(ln)].tolist(), N_SYMBOLS, close_with_sil=True
        )
        assert got[row, : len(expected)].tolist() == expected, f"row {row}"
        assert (got[row, len(expected) :] == 0).all(), f"row {row} padding"


def test_no_diphone_id_is_ever_blank():
    labels = torch.randint(1, N_CLASSES, (4, 6), dtype=torch.long)
    lengths = torch.tensor([6, 5, 3, 2], dtype=torch.long)
    got, got_lens = batch_phonemes_to_diphones(labels, lengths, N_SYMBOLS)
    for row, ln in enumerate(got_lens.tolist()):
        assert ln >= 1
        assert (got[row, :ln] >= 1).all(), f"row {row} produced a 0 diphone id"
        assert (got[row, :ln] < N_DIPHONES).all(), f"row {row} produced an out-of-range id"


# ---------------------------------------------------------------------------
# marginalisation -- the load-bearing piece
# ---------------------------------------------------------------------------
def test_marginalisation_matches_brute_force():
    torch.manual_seed(0)
    d = torch.randn(2, 3, N_DIPHONES, dtype=torch.float64)

    probs = torch.softmax(d, dim=-1)
    block = probs[..., 1:].reshape(*d.shape[:-1], N_SYMBOLS, N_SYMBOLS)
    expected_phonemes = 0.5 * (block.sum(dim=-1) + block.sum(dim=-2))

    out = torch.softmax(marginalize_diphone_logits(d, N_CLASSES), dim=-1)

    assert torch.allclose(out[..., 0], probs[..., 0], atol=1e-12)
    assert torch.allclose(out[..., 1:], expected_phonemes, atol=1e-12)
    assert torch.allclose(out.sum(dim=-1), torch.ones(d.shape[:-1], dtype=torch.float64), atol=1e-12)


def test_marginalisation_shape_and_dtype():
    d = torch.randn(8, 17, N_DIPHONES)
    out = marginalize_diphone_logits(d, N_CLASSES)
    assert out.shape == (8, 17, N_CLASSES)
    assert out.dtype == d.dtype


def test_marginalisation_is_differentiable():
    d = torch.randn(2, 5, N_DIPHONES, requires_grad=True)
    marginalize_diphone_logits(d, N_CLASSES).sum().backward()
    assert d.grad is not None
    assert torch.isfinite(d.grad).all()
    # blank logits only reach the blank output, so they must get gradient too
    assert d.grad[..., 0].abs().sum() > 0


def test_marginalisation_rejects_wrong_width():
    try:
        marginalize_diphone_logits(torch.randn(1, 4, N_CLASSES), N_CLASSES)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a phoneme-width input")


# ---------------------------------------------------------------------------
# model contract
# ---------------------------------------------------------------------------
def _build(cls, torch_compile=False):
    model = cls(
        neural_dim=512,
        n_units=32,
        n_days=2,
        n_classes=N_CLASSES,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=2,
        patch_size=14,
        patch_stride=4,
    )
    return model


def test_forward_returns_baseline_shaped_phoneme_logits():
    model = _build(DiphoneGRUDecoder)
    model.eval()
    x = torch.randn(3, 40, 512)
    day = torch.tensor([0, 1, 0])

    logits = model(x, day)
    assert logits.shape == (3, (40 - 14) // 4 + 1, N_CLASSES)

    logits_with_state, hidden = model(x, day, return_state=True)
    assert logits_with_state.shape == logits.shape
    assert hidden.shape[1] == 3


def test_forward_diphone_returns_full_diphone_width():
    model = _build(DiphoneGRUDecoder)
    model.eval()
    x = torch.randn(2, 40, 512)
    logits = model.forward_diphone(x, torch.tensor([0, 1]))
    assert logits.shape == (2, (40 - 14) // 4 + 1, N_DIPHONES)


def test_marginalised_forward_equals_manual_marginalisation():
    """`forward` must be exactly `marginalize(forward_diphone)`, no more, no less."""
    model = _build(DiphoneGRUDecoder)
    model.eval()
    x = torch.randn(2, 60, 512)
    day = torch.tensor([0, 1])

    with torch.no_grad():
        via_forward = model(x, day)
        manual = marginalize_diphone_logits(
            model.forward_diphone(x, day), N_CLASSES
        )

    assert torch.allclose(via_forward, manual, atol=1e-5)


def test_parameter_names_match_baseline():
    """The optimizer/scheduler contract in rnn_trainer.py depends on these names."""
    baseline = {name for name, _ in _build(GRUDecoder).named_parameters()}
    diphone = {name for name, _ in _build(DiphoneGRUDecoder).named_parameters()}
    assert diphone == baseline, (
        f"diphone-only: {sorted(diphone - baseline)}, "
        f"baseline-only: {sorted(baseline - diphone)}"
    )


def test_optimizer_param_groups_are_still_three():
    """Mirrors rnn_trainer.create_optimizer's substring split."""
    model = _build(DiphoneGRUDecoder)
    named = list(model.named_parameters())

    bias_params = [p for n, p in named if "gru.bias" in n or "out.bias" in n]
    day_params = [p for n, p in named if "day_" in n]
    other_params = [
        p for n, p in named if "day_" not in n and "gru.bias" not in n and "out.bias" not in n
    ]

    assert len(day_params) == 4, "2 day layers x (day_weights, day_biases)"
    assert len(bias_params) == 1 + 2 + 2, "out.bias + 2 GRU layers x (bias_ih, bias_hh)"
    assert len(other_params) > 0
    assert len(bias_params) + len(day_params) + len(other_params) == len(named)


def test_day_layer_is_untouched_by_the_diphone_change():
    """Day layers keep the baseline's identity init and their original shape."""
    model = _build(DiphoneGRUDecoder)
    assert model.day_weights[0].shape == (512, 512)
    assert torch.allclose(model.day_weights[0], torch.eye(512))
    assert model.day_biases[0].shape == (1, 512)
    assert model.gru.input_size == 512 * 14


def test_gradients_reach_the_day_layer_through_a_diphone_ctc_loss():
    """End-to-end smoke test: diphone targets -> CTC -> backward -> day grads."""
    model = _build(DiphoneGRUDecoder)
    model.train()

    torch.manual_seed(1)
    batch, time = 4, 40
    x = torch.randn(batch, time, 512, requires_grad=False)
    day = torch.tensor([0, 1, 0, 1])

    phoneme_labels = torch.tensor(
        [
            [1, 2, 3, 4, 5, 0, 0, 0],
            [6, 7, 8, 0, 0, 0, 0, 0],
            [9, 10, 11, 12, 13, 14, 0, 0],
            [15, 16, 17, 18, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    phoneme_lens = torch.tensor([5, 3, 6, 4], dtype=torch.long)

    diphone_labels, diphone_lens = batch_phonemes_to_diphones(
        phoneme_labels, phoneme_lens, N_SYMBOLS
    )

    adjusted_lens = torch.tensor(
        [(time - 14) // 4 + 1] * batch, dtype=torch.int32
    )
    assert (adjusted_lens >= diphone_lens).all(), "CTC needs input_length >= target_length"

    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
    logits = model.forward_diphone(x, day)
    loss = ctc(
        torch.permute(logits.log_softmax(2), [1, 0, 2]),
        diphone_labels,
        adjusted_lens,
        diphone_lens,
    )
    loss = loss.mean()

    assert torch.isfinite(loss), f"CTC loss was {loss.item()}"
    loss.backward()
    assert model.day_weights[0].grad is not None
    assert torch.isfinite(model.day_weights[0].grad).all()
    assert model.out.weight.grad is not None
    assert model.out.weight.grad.shape == (N_DIPHONES, 32)


def test_diphone_head_shape_is_derived_not_passed():
    model = _build(DiphoneGRUDecoder)
    assert model.out.out_features == N_DIPHONES
    assert model.n_phoneme_classes == N_CLASSES
    assert model.n_diphone_classes == N_DIPHONES


def test_output_width_matches_the_session_class_order():
    """The LM bridge indexes logits by the session's own phoneme order.

    `evaluate_model_helpers.load_session_phoneme_order` reads `phoneme_to_id`
    from the session's `metadata.json` and `expand_logits_to_official_order`
    requires the model's last dimension to equal that length. If the diphone
    model emitted diphone-width logits here, the language model would either
    raise or silently mis-map classes.
    """
    import json

    metadata = (
        _PARENT.parent
        / "data"
        / "hdf5_data_512"
        / "t15.2026.08.14.10-11-24_tc_sbp_512"
        / "metadata.json"
    )
    if not metadata.exists():
        print(f"  (skipped: {metadata} not present)")
        return

    labels = json.loads(metadata.read_text(encoding="utf-8")).get("labels", {})
    session_classes = len(labels["phoneme_to_id"])
    assert session_classes == N_CLASSES

    model = DiphoneGRUDecoder(
        neural_dim=512,
        n_units=16,
        n_days=1,
        n_classes=session_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=1,
        patch_size=14,
        patch_stride=4,
    )
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.randn(1, 40, 512), torch.tensor([0]), return_state=True)
    assert logits.shape[-1] == session_classes, (
        f"model emitted {logits.shape[-1]} classes but the session order has {session_classes}"
    )


def test_state_dict_head_shape():
    model = _build(DiphoneGRUDecoder)
    assert model.state_dict()["out.weight"].shape == (N_DIPHONES, 32)
    assert model.state_dict()["out.bias"].shape == (N_DIPHONES,)


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------
def test_real_trial_is_ctc_feasible_as_diphones():
    """Target length must not exceed the post-patch time resolution."""
    h5py = __import__("h5py")
    path = (
        _PARENT.parent
        / "data"
        / "hdf5_data_512"
        / "t15.2026.08.14.10-11-24_tc_sbp_512"
        / "data_train.hdf5"
    )
    if not path.exists():
        print(f"  (skipped: {path} not present)")
        return

    patch_size, patch_stride = 14, 4
    checked = 0
    with h5py.File(path, "r") as f:
        for key in list(f.keys()):
            g = f[key]
            seq = torch.as_tensor(g["seq_class_ids"][:]).reshape(1, -1).long()
            length = torch.tensor(
                [int(g.attrs["seq_len"]) if "seq_len" in g.attrs else seq.shape[1]]
            )
            n_time_steps = int(torch.as_tensor(g.attrs["n_time_steps"]).reshape(-1)[0])

            diphones, diphone_lens = batch_phonemes_to_diphones(
                seq, length, N_SYMBOLS
            )
            adjusted = (n_time_steps - patch_size) // patch_stride + 1
            assert diphone_lens[0].item() <= adjusted, (
                f"{key}: {diphone_lens[0].item()} diphones > {adjusted} frames"
            )
            assert int(diphones.max()) < N_DIPHONES
            checked += 1

    print(f"  checked {checked} real trials")


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
