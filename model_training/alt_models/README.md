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
| `day_calibration.py` | `DayCalibratedGRUDecoder(GRUDecoder)` — hammer + scalpel day layer (see below). |
| `registry.py` | `resolve_model_class(config)` — the 2×2 diphone × day-calibration dispatch. |
| `diphone_trainer.py` | `DiphoneTrainer(BrainToTextDecoder_Trainer)` — swap the model class, swap the targets. |
| `train_diphone.py` / `evaluate_diphone.py` | entry points, siblings of `train_model.py` / `evaluate_model.py`. |
| `rnn_args_diphone.yaml` | config (English 512D, copy task). |
| `make_all_day_config.py` | generates the multi-day `sessions:` / `dataset_probability_val:` block. |
| `check_config.py` | pre-flight for the `sessions:` block — catches the silent one-day failure. |
| `install_matlab_session.py` | installs a MATLAB `data_*.hdf5` folder as a session; reconciles the dir-name/attr split and supplies `metadata.json`. |
| `trim_silence.py` | drops over-long silent stretches from a session, using the session's microphone audio. |
| `h5py_compat.py` | fixes a pre-existing numpy-2 bug in the *shared* eval helpers (see below). |
| `tests/test_diphone.py` | 22 correctness tests, incl. real-trial CTC feasibility. |
| `tests/test_day_calibration.py` | 12 tests: identity-at-init, gate behaviour, param groups, registry. |
| `tests/test_trim_silence.py` | 13 tests: keep-planning, adaptive VAD, real-data speech retention. |

Two config keys in the `model:` block select among four model classes:

```yaml
model:
  diphone_targets: true        # default true   — this package is diphone-first
  day_calibration: baseline    # baseline | hammer_scalpel
```

| `diphone_targets` | `day_calibration` | class |
|---|---|---|
| `false` | `baseline` | `rnn_model.GRUDecoder` (the original) |
| `true` | `baseline` | `DiphoneGRUDecoder` |
| `false` | `hammer_scalpel` | `DayCalibratedGRUDecoder` |
| `true` | `hammer_scalpel` | `DiphoneDayCalibratedGRUDecoder` |

Both knobs are written into the checkpoint's `args.yaml`, so
`evaluate_diphone.py` picks the right class back up with no `--arch` flag.

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

### Always run the pre-flight first

A wrong `sessions:` list fails **silently**. Nothing scans `dataset_dir` —
`rnn_trainer.py:163` builds `<dataset_dir>/<session>/data_train.hdf5` straight
from the list — so a forgotten day, a typo'd directory, or a
`dataset_probability_val:` of the wrong length all produce a run that starts
fine, trains fine, and reports a PER for the wrong dataset. There is no error to
read afterwards.

```bash
python alt_models/check_config.py alt_models/rnn_args_diphone_alldays.yaml
```

It verifies every session directory exists with `data_train.hdf5` and
`data_val.hdf5`, that `dataset_probability_val` matches `sessions` in length,
that the order is chronological (list position *is* the day index), and exits
non-zero on any problem, so it chains:

```bash
python alt_models/check_config.py rnn_args.yaml && python train_model.py rnn_args.yaml
```

It works on any config in this repo, not just the diphone ones.

### All days

Generate the day list from whatever HDF5 is on disk, then train:

```bash
cd Speech/model_training
python alt_models/make_all_day_config.py \
    --dataset_dir ../data/hdf5_data_512 \
    --output alt_models/rnn_args_diphone_alldays.yaml \
    --day_calibration hammer_scalpel

python alt_models/train_diphone.py alt_models/rnn_args_diphone_alldays.yaml
```

Day order matters: `sessions[i]` **is** the day index the day-specific input
layer uses, so regenerate the config rather than hand-editing a trained one.
The script sorts lexically, which for `<name>.<YYYY>.<MM>.<DD>.<HH-MM-SS>_<sfx>`
is also chronologically. `dataset_probability_val` gets one `1` per day, meaning
"validate on this day" — it gates reporting, not the train/val trial split
(which is `dataset.test_percentage`).

With more than one day, `--output_dir` defaults to the base `output_dir` suffixed
with the day count (`..._512_2day`), so a multi-day run can't overwrite a
single-day checkpoint sitting at the same path. Pass `--output_dir` explicitly
to override.

> Multi-day training needs `data_train.hdf5` under `dataset_dir` for every day.
> Those come out of the MATLAB 512 pipeline
> (`sub-01/run_20260814_nejm_512_hdf5_v6_prevblock_calibration.m`), not out of
> `data_preprocessing/preprocess_sub01_english_electrode.py`, which is 256-dim
> TC-only.

### Installing a new day

The MATLAB pipeline writes its HDF5s straight into a folder named after the
**raw recording** (`.../self_mat/hdf5_data_512/20260814-144544/`) while stamping
every trial with the **session** name
(`t15.2026.08.14.15-05-37_tc_sbp_512_prevblockcal`). Both strings are
load-bearing and they are not the same string:

* `rnn_trainer.py:163` builds `<dataset_dir>/<config entry>/data_train.hdf5`, so
  the config entry must equal the **directory** name;
* `evaluate_model_extended.py:199` does `sessions.index(session)` on the
  per-trial **attribute**.

Three strings have to agree exactly — directory, config entry, attribute.
MATLAB guarantees only the third, and it emits no `metadata.json` either.
`install_matlab_session.py` closes both gaps, then runs `audit_hdf5.py` on the
result and exits non-zero if it is not clean:

```bash
python alt_models/install_matlab_session.py \
    --source-dir /mnt/d/wwl/data/self_mat/hdf5_data_512/20260814-144544 \
    --dataset-dir ../data/hdf5_data_512 \
    --session-name t15.2026.08.14.15-05-37_tc_sbp_512
```

Omit `--session-name` to keep the name MATLAB stamped. `--metadata-from` points
at a session to borrow `labels.phoneme_to_id` from (day 1 is fine — the
vocabulary is a fixed property of the corpus, which the audit re-checks
afterwards by re-encoding every label).

The labels themselves need no attention: the MATLAB script copies
`seq_class_ids` and `transcription` directly out of the reference day's HDF5,
so a new day cannot drift onto a different phoneme ordering. That is why
`build_hdf5_from_mat.py` — which re-derives the encoding — is **not** the tool
for MATLAB 512D output. It exists for `.mat` sources, and it runs strict
precisely so that pointing it at the wrong corpus fails loudly instead of
writing a session whose every label is a lone `<sil>`.

### Trimming over-long silence

`read_begin`/`read_end` come from the task CSV, not from the subject's voice,
so a late marker or a long pause yields a trial that is almost entirely room
tone. The 2026-08-14 15-05-37 session runs 3.6 s to 57.4 s per trial (median
6.8 s), with gid 105 at 50.0 s and gid 119 at 57.4 s; day 1 has the same shape
(gid 148 at 52.3 s). Those trials cost a disproportionate amount of memory and
compute and teach the model nothing.

```bash
python alt_models/trim_silence.py \
    --session-dir ../data/hdf5_data_512/t15.2026.08.14.15-05-37_tc_sbp_512 \
    --audio /mnt/d/wwl/.../session-15-05-37/EnglishSpeech/microphone_audio.wav \
    --out-dir ../data/hdf5_data_512/t15.2026.08.14.15-05-37_tc_sbp_512_trim
```

Run with `--dry-run` first; it prints the full report and writes nothing. Then
`audit_hdf5.py` the output before training. `--task-csv` defaults to the only
`data_*.csv` sitting next to the wav.

It uses a 20 ms-frame VAD over the trial's own audio. Feature row `k` is at
`read_begin + k*20 ms` — checked against all 189 day-1 trials, where
`n_time_steps == floor(read_duration_sec * 50)` exactly — and sample 0 of the
wav is the CSV's `mic_start` event. The threshold is set from the trial's own
percentiles (`p95 - 22 dB`, floored at `p20 + 20 dB`) rather than an absolute
level, because room tone here ranges over 15–35 dB between sessions. A trial
with under 15 dB of speech-to-noise contrast is left untouched and reported.

Leading and trailing silence is capped at `--edge-keep-ms`; silent runs longer
than `--max-silence-ms` are collapsed to `--keep-silence-ms`, so the pause stays
visible as a pause instead of vanishing.

**`seq_class_ids` is never modified.** CTC sums over alignment paths, so it
needs no frame-level alignment — removing input frames cannot invalidate the
transcript provided at least `L` frames remain. The script enforces that and
reports any trial where it had to clamp. On day 1 it removes 39 % of all frames
while retaining **100 % of the frames the VAD called speech**, on all 189
trials.

> Trim every day with the same flags. A model trained on trimmed day 2 and
> untrimmed day 1 is being asked to reconcile two input distributions on top of
> the day difference it is meant to be learning.

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

## Day calibration (`hammer_scalpel`)

After **NHS** (Olak et al., ICLR 2026). The baseline day layer already gives each
day its own affine input map `X_h = X W_d + 1 b_dᵀ` — the paper's *hammer*. It
adds a second, multiplicative branch, the *scalpel*:

```
X_s = X ⊙ γ_d + β_d                  # FiLM
g_d = σ(w_gᵀ e_d)                    # learned per-day gate
X   = g_d · X_s + (1 - g_d) · X_h
```

so a day can be corrected by a full affine map, by a cheap diagonal + shift, or
by any mix the data chooses.

**It starts as an exact no-op.** With `W_d = I`, `b_d = 0`, `γ_d = 1`, `β_d = 0`
both branches equal `X`, and `σ(0) = 0.5` splits between two identical things —
so at step 0 the model computes `softsign(X)`, byte-identical to the baseline.
It can only diverge if the data pulls it there. `test_identity_at_init` and
`test_gate_closed_reproduces_the_baseline_and_open_does_not` pin both halves of
that claim; `test_identity_at_init_for_the_diphone_variant` pins it one level up,
where the head is 1157-wide and a state-dict comparison isn't possible.

**Only worth enabling with more than one day.** With a single session there is
nothing to calibrate *between*, and the gate has nothing to learn. That is why
the shipping config defaults to `baseline`.

**Optimizer contract.** Every new parameter is named `day_*` (`day_scales.i`,
`day_offsets.i`, `day_gate_logits`), so `rnn_trainer.create_optimizer` puts them
in the day-layer group and they get `lr_max_day` / `weight_decay_day` rather than
the general LR. Baseline parameter names are untouched — the calibrator only
*adds* names. Both pinned by `test_every_new_parameter_is_in_the_day_param_group`
and `test_baseline_parameter_names_are_unchanged`.

Enabling it:

```bash
python alt_models/make_all_day_config.py \
    --dataset_dir ../data/hdf5_data_512 \
    --output alt_models/rnn_args_diphone_alldays.yaml \
    --day_calibration hammer_scalpel
```

or set `model.day_calibration: hammer_scalpel` by hand.

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
python alt_models/tests/test_diphone.py          # 22 tests
python alt_models/tests/test_day_calibration.py  # 12 tests
# or: python -m pytest alt_models/tests -v
```

`test_diphone.py` covers the encoding (against a scalar reference
implementation), the marginalisation (against a materialised brute-force
computation, plus row sums and differentiability), the model contracts (shapes,
parameter names, param-group split, day-layer identity init, gradients reaching
the day layer through a real CTC loss), and CTC feasibility on all 153 real
training trials.

`test_day_calibration.py` covers identity-at-init for both the monophone and
diphone variants, that the gate actually modulates (closed → baseline, open with
`γ=2` → not baseline), that gradients reach all three new parameter sets, the
optimizer param-group contract, and all four registry combinations.
