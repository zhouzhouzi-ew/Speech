# `alt_models/` — alternative decoders (diphone CTC)

Additive only. **Nothing under `model_training/` is modified**, so the original
commands keep working byte-for-byte:

```bash
python train_model.py rnn_args.yaml          # unchanged
python evaluate_model.py --model_path ...    # unchanged
```

The diphone work lives entirely in this folder and is wired in by *rebinding a
name in an already-imported module's namespace*, not by editing it.

---

## What "diphone" means here

After **DCoND** (arXiv:2411.10657). A monophone is acoustically ambiguous — its
realisation depends on its neighbours. So instead of predicting the isolated
phoneme, the head predicts the *transition* (the diphone), and the monophone
posterior is recovered by **exact marginalisation**:

```
P(p_k) = 0.5 * ( sum_j P(p_k, p_j)  +  sum_i P(p_i, p_k) )
```

That marginalisation is the whole trick. Because the model ends up emitting an
ordinary `n_classes`-way **phoneme** posterior, everything downstream — the
greedy CTC decode, PER, `save_val_logits`, `rearrange_speech_logits_pt`, the
Redis WFST language model — is completely untouched and unaware.

**No architecture change.** Same day layer, same patch embedding, same 5-layer
unidirectional GRU, same `h0`, same initialisers, same augmentation, same
optimizer with its three param groups and dedicated day-layer LR.

| | baseline | diphone |
|---|---|---|
| output classes | 35 | **1157** (`1 + 34²`) |
| CTC target | phonemes | phoneme transitions |
| CTC target length | `L` | `L-1` (or `L` with `close_with_sil`) |
| logits seen by decode / LM | 35 | **35** (identical) |

Head grows from 768×35 to 768×1157 (≈ +0.86 M params).

---

## Files

| file | role |
|---|---|
| `diphone.py` | encoding + marginalisation. Pure functions, no state. |
| `diphone_model.py` | `DiphoneGRUDecoder(GRUDecoder)` — derived head width, marginalised `forward`. |
| `diphone_trainer.py` | `DiphoneTrainer(BrainToTextDecoder_Trainer)` — swap the model class, swap the targets. |
| `train_diphone.py` / `evaluate_diphone.py` | entry points, siblings of `train_model.py` / `evaluate_model.py`. |
| `rnn_args_diphone.yaml` | config (English 512D, copy task). |
| `make_all_day_config.py` | generates the multi-day `sessions:` / `dataset_probability_val:` block. |
| `h5py_compat.py` | fixes a pre-existing numpy-2 bug in the *shared* eval helpers (see below). |
| `tests/test_diphone.py` | 22 correctness tests, incl. real-trial CTC feasibility. |

### `DiphoneGRUDecoder` keeps two contracts that the rest of the repo relies on

1. **Shapes.** `forward()` returns `(B, T, n_classes)` phoneme logits — the same
   as `GRUDecoder`. `forward_diphone()` returns the raw `(B, T, 1157)` diphone
   logits and is called only by the training loop.
2. **Parameter names.** `named_parameters()` is *identical* to the baseline's
   (`day_weights.*`, `day_biases.*`, `gru.*`, `out.*`, `h0`). `rnn_trainer`
   splits params by substring (`'day_'`, `'gru.bias'`, `'out.bias'`) and the
   cosine scheduler branches on the resulting group count — so training would
   silently lose its dedicated day-layer learning rate if any of these were
   renamed. `test_parameter_names_match_baseline` pins it.

The constructor takes the **baseline** class count (35) and derives 1157
internally, which is why `evaluate_model_extended.py` can instantiate this class
without an edit.

---

## Running

### Single day

```bash
cd Speech/model_training
python alt_models/train_diphone.py alt_models/rnn_args_diphone.yaml

python alt_models/evaluate_diphone.py \
    --model_path trained_models/diphone_rnn_electrode_512 \
    --data_dir ../data/hdf5_data_512 \
    --eval_type test --skip_lm \
    --output_prefix diphone_rnn
```

### All days

Generate the day list from whatever HDF5 is on disk, then train:

```bash
cd Speech/model_training
python alt_models/make_all_day_config.py \
    --dataset_dir ../data/hdf5_data_512 \
    --output alt_models/rnn_args_diphone_alldays.yaml \
    --output_dir trained_models/diphone_rnn_alldays

python alt_models/train_diphone.py alt_models/rnn_args_diphone_alldays.yaml
```

Day order matters: `sessions[i]` **is** the day index the day-specific input
layer uses, so regenerate the config rather than hand-editing a trained one.

> Multi-day training needs `data_train.hdf5` under `dataset_dir` for every day.
> Those come out of the MATLAB 512 pipeline
> (`sub-01/run_20260814_nejm_512_hdf5_v6_prevblock_calibration.m`), not out of
> `data_preprocessing/preprocess_sub01_english_electrode.py`, which is 256-dim
> TC-only.

---

## Config knobs that actually matter

* **`num_training_batches` / `lr_decay_steps`.** The diphone head is 33× wider,
  so it needs more steps than the monophone baseline to reach the same loss.
  The config ships 20,000 (and matches `lr_decay_steps` to it). If val PER is
  still falling at the last logged step, raise both.
* **`diphone.close_with_sil`.** `false` (default) → a length-`L` phoneme
  sequence becomes `L-1` consecutive diphones, the textbook DCoND form and
  strictly easier for CTC than the baseline, which needed `L` frames. `true` →
  close with `<sil>`, giving `L` diphones and the baseline's exact target
  length. Try both if the first is disappointing.
* **`diphone.sil_phoneme_id`.** `null` means `n_classes - 1` = 34. Verified
  against `metadata.json`, where `<sil>` is indeed the last label.

### One caveat worth knowing

The marginalised logits are `logsumexp` over the contributing raw diphone
logits, so they carry roughly a `+log(34)` offset relative to the baseline's raw
linear output. **Greedy PER is argmax-based and therefore unaffected**, but the
language model's `acoustic_scale` / `blank_penalty` (tuned against the baseline's
logit scale) may want a small retune. Neural metrics need no adjustment.

---

## Known pre-existing bug this folder works around

`evaluate_model_helpers.py:186` does `int(g.attrs['paired_diagnostic_block_num'])`.
The MATLAB pipeline stores that attribute as a **1-element array**, and

```python
int(np.array([8]))   # DeprecationWarning in numpy 1.x, TypeError in numpy 2.x
```

So **`evaluate_model.py` is broken for every MATLAB-generated dataset** on
numpy ≥ 2 (reproduced on numpy 2.4.3 / Python 3.14) — this is not caused by the
diphone change; the diphone model just surfaced it. `dataset.py` already handles
the same situation with its `_hdf5_scalar` helper, which is why *training* works
and evaluation does not.

`h5py_compat.py` fixes it by making size-1 attributes read back as Python
scalars, and `evaluate_diphone.py` installs it. Attributes of size > 1 are
returned untouched.

**If you want the original `evaluate_model.py` working again**, apply the same
three-line shim there, or change line 186 to reuse `dataset._hdf5_scalar`. That
edit touches a shared file, so it was left out of this folder on purpose.

---

## Tests

```bash
cd Speech/model_training
python alt_models/tests/test_diphone.py          # or: python -m pytest alt_models/tests -v
```

Covers the encoding (against a scalar reference implementation), the
marginalisation (against a materialised brute-force computation, plus row sums
and differentiability), the model contracts (shapes, parameter names, param-group
split, day-layer identity init, gradients reaching the day layer through a real
CTC loss), and CTC feasibility on all 153 real training trials.
