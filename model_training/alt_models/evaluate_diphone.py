"""Evaluate a diphone-target GRU.  Drop-in sibling of `model_training/evaluate_model.py`.

    cd Speech/model_training
    python alt_models/evaluate_diphone.py \
        --model_path trained_models/diphone_rnn_electrode_512 \
        --data_dir ../data/hdf5_data_512 \
        --eval_type test \
        --output_prefix diphone_rnn

The whole evaluation stack -- greedy phoneme decode, PER, the phoneme->word
lexicon, the `rearrange_speech_logits_pt` reordering, the Redis WFST language
model -- is reused verbatim from `evaluate_model_extended.py`.  The only thing
this file does is hand that module the right model class; because the diphone
model's `forward` returns marginalised *phoneme* logits with the baseline class
ordering, nothing downstream can tell the difference.

Which class that is comes from the checkpoint's own `checkpoint/args.yaml`, so
the architecture is recoverable without a `--arch` flag and a diphone run and a
diphone+day-calibration run use the same command.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (str(_PARENT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py_compat  # noqa: E402
import evaluate_model_extended as _eme  # noqa: E402
from registry import describe_model_class, resolve_model_class  # noqa: E402


def _peek_model_path(argv):
    """Read `--model_path` without consuming the rest of the command line.

    `evaluate_model_extended.main()` does its own full argparse pass; this only
    needs the one value, early enough to choose a model class before `main()`
    constructs it.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model_path", type=str, default="../data/t15_pretrained_rnn_baseline"
    )
    known, _ = parser.parse_known_args(argv)
    return known.model_path


def main() -> None:
    # Must run before any HDF5 file is opened. See h5py_compat for why the
    # upstream evaluation path needs this (numpy 2.x rejects int(array([8]))).
    h5py_compat.install()

    model_path = _peek_model_path(sys.argv[1:])
    args_path = Path(model_path) / "checkpoint" / "args.yaml"
    if not args_path.exists():
        raise SystemExit(
            f"No checkpoint/args.yaml under {model_path}. Point --model_path at a "
            "trained model directory."
        )

    model_args = _eme.load_model_args(args_path)
    # `evaluate_model_extended` does `from rnn_model import GRUDecoder`, so the
    # name it instantiates lives in that module's namespace. Rebinding it here is
    # the entire integration -- the 583-line evaluation file is not modified.
    _eme.GRUDecoder = resolve_model_class(model_args)
    print(f"Evaluating with {describe_model_class(model_args)}")

    _eme.main()


if __name__ == "__main__":
    main()
