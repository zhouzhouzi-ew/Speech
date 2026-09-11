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
this file does is hand that module a `DiphoneGRUDecoder` instead of a
`GRUDecoder`; because the diphone model's `forward` returns marginalised
*phoneme* logits with the baseline class ordering, nothing downstream can tell
the difference.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
for _p in (str(_PARENT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h5py_compat  # noqa: E402
import evaluate_model_extended as _eme  # noqa: E402
from diphone_model import DiphoneGRUDecoder  # noqa: E402

# `evaluate_model_extended` does `from rnn_model import GRUDecoder`, so the name
# it instantiates lives in that module's namespace. Rebinding it here is the
# entire integration -- the 583-line evaluation file is not modified.
_eme.GRUDecoder = DiphoneGRUDecoder


def main() -> None:
    # Must run before any HDF5 file is opened. See h5py_compat for why the
    # upstream evaluation path needs this (numpy 2.x rejects int(array([8]))).
    h5py_compat.install()
    _eme.main()


if __name__ == "__main__":
    main()
