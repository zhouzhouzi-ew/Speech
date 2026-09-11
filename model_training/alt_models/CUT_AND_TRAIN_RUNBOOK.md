# Runbook — trim the 814 sessions, then train and evaluate

For the GPU box (WSL, `~/nejm-brain-to-text-en`). Every command is run from
`~/nejm-brain-to-text-en/model_training` unless stated otherwise.

Context: the MATLAB script `run_20260814_144544_512_hdf5_prevblock_calibration.m`
writes the 2026-08-14 sentence session to

```
/mnt/d/wwl/data/self_mat/hdf5_data_512/20260814-144544/
```

with per-trial `session` = `t15.2026.08.14.15-05-37_tc_sbp_512_prevblockcal`.
The directory it writes into is named after the *recording* (`20260814-144544`),
not the session, and there is no `metadata.json` — both are handled in step 2.

The audio needed to detect silence exists only on the machine where the
recordings live. The cut plans were therefore computed there and committed to
`alt_models/cut_plans/`. **Nothing below needs the wav files.**

---

## 1. Pull

```bash
cd ~/nejm-brain-to-text-en
git pull
cd model_training
```

Confirm the plans arrived:

```bash
ls -la alt_models/cut_plans/
#   2026-08-14_10-11-24.json    day 1 (session-10-11-24)
#   2026-08-14_15-05-37.json    day 2 (session-15-05-37)
```

## 2. Install the MATLAB day-2 output as a session

```bash
python alt_models/install_matlab_session.py \
    --source-dir /mnt/d/wwl/data/self_mat/hdf5_data_512/20260814-144544 \
    --dataset-dir ../data/hdf5_data_512 \
    --session-name t15.2026.08.14.15-05-37_tc_sbp_512 \
    --overwrite
```

**Always pass `--session-name`.** The `_prevblockcal` suffix comes from the
MATLAB script, which names the session after the block it calibrated against
(`run_20260814_144544_512_hdf5_prevblock_calibration.m`) and stamps that string
into all 189 trials. The installer's default is to reuse whatever MATLAB
stamped, so leaving the flag off is what produces `..._tc_sbp_512_prevblockcal`.
Nothing in this repo requires the suffix — pass the name you want and both the
directory and every trial attribute take it.

It reconciles the directory name against the per-trial `session` attribute
(three strings have to agree exactly — see `README.md`), borrows
`labels.phoneme_to_id` from day 1 if it can find one, and runs the audit itself.
**It exits non-zero if the audit is not clean — stop there if it does.**

Remember the name it prints; step 3 needs it:

```bash
cd ~/nejm-brain-to-text-en/model_training
ls -d ../data/hdf5_data_512/*15-05-37*
```

### If you renamed the session directory by hand

Renaming the folder fixes the config, because `make_all_day_config.py` enumerates
from directory names — but it does **not** touch the per-trial `session`
attribute MATLAB wrote. `evaluate_model_extended.py:199` does
`sessions.index(session)` on that attribute, so a renamed-but-stale session
audits with 189 problems (`session attr '…_prevblockcal' != directory name '…'`)
and fails at evaluation, after training.

Two ways out; either is fine.

**Preferred — reinstall with the new name.** The original MATLAB output is still
on disk and the installer rebuilds the session from it:

```bash
python alt_models/install_matlab_session.py \
    --source-dir /mnt/d/wwl/data/self_mat/hdf5_data_512/20260814-144544 \
    --dataset-dir ../data/hdf5_data_512 \
    --metadata-from ../data/hdf5_data_512/t15.2026.08.14.10-11-24_tc_sbp_512 \
    --session-name t15.2026.08.14.15-05-37_tc_sbp_512 \
    --overwrite
```

If that box has no `metadata.json` anywhere, the installer says so and writes
the built-in 35-class vocabulary instead — the audit that follows re-encodes
every label against exactly that map, so the session is still verified. Read the
warning rather than skipping past it.

**Or just proceed to step 3.** The trimmer stamps its output with the name of
the directory it writes into, so `hdf5_data_512_trim/` is clean whichever way
you got there — training and evaluation read only that root. The stale source is
then harmless; it only shows up if you audit the raw `hdf5_data_512/`.

## 3. Trim both days

The plans are replayed, not recomputed — no audio required. Each trial's frame
count is checked against the plan before cutting, so a plan applied to the wrong
session is **refused per trial** rather than cutting at the wrong offset.

Start with `--dry-run`, on both days, and read the output:

```bash
python alt_models/trim_silence.py --plan alt_models/cut_plans/2026-08-14_10-11-24.json \
    --session-dir ../data/hdf5_data_512/t15.2026.08.14.10-11-24_tc_sbp_512 \
    --out-dir     ../data/hdf5_data_512_trim/t15.2026.08.14.10-11-24_tc_sbp_512 \
    --dry-run
```

For day 1 expect `train 153 trials 55745 -> 48495 bins (13.0% removed)`,
`val 8812 -> 6385 (27.5%)`, `test 6001 -> 5084 (15.3%)`, exit 0. If those numbers
differ, the day-1 HDF5 is not the one the plan was built against — stop and ask
before training.

For day 2, substitute the session name from step 2 and the `15-05-37` plan. Day
2 has no ground truth to compare against yet, so the check that matters there is
simply **exit 0 and no `REFUSED` lines**. If trials are refused, the MATLAB bin
counts disagree with `floor(read_duration_sec * 50)` and the plan does not fit —
stop and report it rather than training on a partial trim.

Then drop `--dry-run` on both. Writing to the parallel `hdf5_data_512_trim/` root
is deliberate: a `_trim` sibling inside `hdf5_data_512/` would be enumerated as a
second day for the same recordings.

## 4. Audit the trimmed sessions

```bash
python alt_models/audit_hdf5.py --dataset_dir ../data/hdf5_data_512_trim
```

Must print `OK -- 2 session(s) pass the structural + label contract.`

This also checks that each session has a `metadata.json`. That file is what maps
the model's 35 classes onto the LM's official 41-phoneme order — the session's
order is not the official one (its id 1 is `AE`, the official id 1 is `AA`), so
without it the LM reads the wrong phonemes. Confirm both are there:

```bash
ls -la ../data/hdf5_data_512_trim/*/metadata.json
```

## 5. Build the two-day config

```bash
python alt_models/make_all_day_config.py \
    --dataset_dir ../data/hdf5_data_512_trim \
    --config alt_models/rnn_args_diphone.yaml \
    --output alt_models/rnn_args_diphone_2day_trim.yaml \
    --day_calibration hammer_scalpel

python alt_models/check_config.py alt_models/rnn_args_diphone_2day_trim.yaml
```

Check the printed day list is **exactly the two 814 sessions in chronological
order** — `sessions[i]` IS the day index, so a stray directory silently becomes a
day layer. `check_config.py` must pass before training.

`hammer_scalpel` is the FiLM + learned-gate day layer; it is an exact no-op at
init, so it can only help. With two days there is finally something for it to
calibrate between. Use `--day_calibration baseline` for the A/B.

`output_dir` becomes `trained_models/diphone_rnn_electrode_512_2day`.

### 5b. The baseline (monophone) config, for the A/B

Steps 5–7 produce **diphone numbers only**. Without a baseline on the same two
trimmed sessions there is nothing to compare them against, so build the
monophone config the same way:

```bash
python alt_models/make_all_day_config.py \
    --dataset_dir ../data/hdf5_data_512_trim \
    --config alt_models/rnn_args_baseline.yaml \
    --output alt_models/rnn_args_baseline_2day_trim.yaml

python alt_models/check_config.py alt_models/rnn_args_baseline_2day_trim.yaml
```

`alt_models/rnn_args_baseline.yaml` is the stock `rnn_args.yaml` plus two
deliberate changes. The file is separate so `rnn_args.yaml` itself stays
untouched.

1. `diphone_targets: false`, which has to be explicit: `registry.py` defaults it
   to **true**, so a config without the key resolves to `DiphoneGRUDecoder` and
   the checkpoint gets evaluated with the wrong head.
2. **`num_training_batches: 20000` and the matching decay / val-step values**,
   where the stock config says 2,000. The two arms have to train for the same
   number of steps or the comparison measures the schedule as much as the model.
   The baseline file also sets `batches_per_val_step: 1000`, so both arms report
   20 val points and the PER curves are directly comparable.

Confirm they line up before starting the run — this prints nothing when they do:

```bash
python - <<'EOF'
from omegaconf import OmegaConf
b = OmegaConf.load("alt_models/rnn_args_baseline_2day_trim.yaml")
d = OmegaConf.load("alt_models/rnn_args_diphone_2day_trim.yaml")
# batch_size lives under `dataset:`, the rest at the top level.
for path in ("num_training_batches", "lr_decay_steps", "lr_decay_steps_day",
             "batches_per_val_step", "seed", "dataset.batch_size",
             "dataset.days_per_batch"):
    get = lambda c: c.get(path, c.dataset.get(path.split(".", 1)[1])
                          if path.startswith("dataset.") else None)
    assert get(b) == get(d), f"{path}: baseline={get(b)} diphone={get(d)}"
print("schedules match")
EOF
```

`output_dir` becomes `trained_models/baseline_rnn_512_2day`.

## 6. Train

```bash
python alt_models/train_diphone.py alt_models/rnn_args_diphone_2day_trim.yaml
```

Watch the per-day val PER (`log_individual_day_val_PER: true`) — with two days it
is the only quick read on whether the day layer is doing anything. `num_training
_batches` is 20 000; if val PER is still falling at the last logged step, raise
it and `lr_decay_steps` together and rerun.

## 7. Evaluate — PER and WER

`--skip_lm` gives PER + lexicon WER cheaply and needs no server:

```bash
python alt_models/evaluate_diphone.py \
    --model_path trained_models/diphone_rnn_electrode_512_2day \
    --data_dir ../data/hdf5_data_512_trim \
    --eval_type test \
    --gpu_number 0 \
    --skip_lm \
    --output_prefix diphone_2day_trim
```

Then the same again **without** `--skip_lm` for the LM-on WER. That needs the
Redis WFST LM running (3-gram, not 5-gram — 5-gram needs `--rescore` and a lot of
RAM):

```bash
# terminal A
redis-server --port 6379 --save "" --appendonly no

# terminal B
conda activate b2txt25_lm
cd ~/nejm-brain-to-text-en
python language_model/language-model-standalone.py \
  --lm_path language_model/pretrained_language_models/openwebtext_3gram_lm_sil \
  --redis_ip localhost --redis_port 6379 --gpu_number 0 \
  --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55
```

Check that asset exists first — it is gitignored and lives only on the box that
has it: `ls language_model/pretrained_language_models/`. Use `localhost`, not the
server default.

```bash
# terminal C
conda activate b2txt25
cd ~/nejm-brain-to-text-en/model_training
python alt_models/evaluate_diphone.py \
    --model_path trained_models/diphone_rnn_electrode_512_2day \
    --data_dir ../data/hdf5_data_512_trim \
    --eval_type test \
    --gpu_number 0 \
    --output_prefix diphone_2day_trim
```

Reports `PER`, `WER LM OFF`, `WER LM ON`, word accuracy, and a per-trial markdown
+ JSON + CSV under `<model_path>/eval_outputs/`. `--eval_type val` for the val
split instead.

## 8. Evaluate the baseline — the numbers to compare against

Same two sessions, same splits, same trimmed data, so the only difference is the
model. Train it with the **original, unmodified** entry points:

```bash
python train_model.py alt_models/rnn_args_baseline_2day_trim.yaml

python evaluate_model.py \
    --model_path trained_models/baseline_rnn_512_2day \
    --data_dir ../data/hdf5_data_512_trim \
    --eval_type test \
    --gpu_number 0 \
    --skip_lm \
    --output_prefix baseline_2day_trim
```

Then the same again without `--skip_lm`, with the LM server from step 7 running.

`train_model.py` and `evaluate_model.py` are the stock ones and stay that way —
`evaluate_model.py` is a two-line shim over `evaluate_model_extended.main()`.

Two things worth knowing before you run it:

- **`evaluate_model.py` does the same everything else does, but no registry.**
  It instantiates `rnn_model.GRUDecoder` directly, so it evaluates a baseline
  checkpoint that has no `diphone_targets` key at all. That is *not* true of
  `evaluate_diphone.py`, which reads the key from the checkpoint and defaults it
  to true — so point step 7's script only at diphone checkpoints.
- **The numpy-2 landmine is fixed, but only as of this commit.** Sizes-1
  attributes written by MATLAB (`int(array([8]))`) used to raise `TypeError` on
  numpy 2 and kill `evaluate_model.py` on day 1's first trial. If you pulled
  before the fix and see that error, pull again. `evaluate_diphone.py` was never
  affected — it patches `h5py` instead.

For an apples-to-apples A/B, keep `--skip_lm` **off** on both or **on** on both.
Greedy WER is unaffected by the LM, but the reported `WER LM ON` is a different
quantity, and the diphone logits carry a `+log(34)` offset that `acoustic_scale`
and `blank_penalty` were not tuned against (see the caveat in step 7). If the
diphone model's LM-on WER looks clearly worse than its LM-off WER while PER is
fine, that is the scale mismatch, not the model.

### What the three arms are

Step 5 builds the diphone config with `--day_calibration hammer_scalpel`, so you
end up with three readable numbers rather than two:

| arm | how | what it isolates |
|---|---|---|
| **baseline** (原始) | `train_model.py` + `evaluate_model.py` on step 5b's config | the published architecture, reproduced on this subject |
| **diphone** | step 5's config with `--day_calibration baseline` | the effect of predicting phoneme transitions |
| **diphone + day-cal** | step 5's config as written (`hammer_scalpel`) | what the day layer adds on two days |

Build the middle one by rerunning step 5 with `--day_calibration baseline` and a
distinct `--output` / `--output_dir` — `make_all_day_config.py` only appends
`_2day`, so leaving the defaults means it overwrites the `hammer_scalpel`
checkpoint's config. To compare **only** the diphone change against the original,
make the middle one and leave the third out; the day layer is a separate question
and mixing it in makes a worse result impossible to attribute.

All three share `dataset_probability_val`, the trim, and the split, so the
numbers are comparable line by line in the `eval_outputs/` markdown.

### One caveat specific to the diphone model

The marginalised diphone logits carry roughly a `+log(34)` offset relative to the
baseline's raw linear output, because they are a `logsumexp` over 34 raw logits.
**PER and greedy WER are argmax-based and unaffected**, but `acoustic_scale` and
`blank_penalty` were tuned against the baseline's logit scale. If LM-on WER comes
out clearly worse than LM-off while PER looks fine, that is the knob to look at
before concluding the model is bad — see the note in `README.md`.

---

## If something goes wrong

| symptom | meaning |
|---|---|
| `install_matlab_session.py` exits non-zero | the MATLAB output failed the audit; do not train on it |
| trim prints `REFUSED` / exits 1 | the plan does not line up with that session's frame counts — the numbers above will say by how much |
| `check_config.py` fails | a session dir is missing `data_train.hdf5`, or `dataset_probability_val` is the wrong length |
| `make_all_day_config.py` refuses | a `name` / `name_trim` pair sits in the same root; point `--dataset_dir` at one or the other |
| `make_all_day_config.py` says a session "is not usable as a day" | that directory's trials are stamped with a different name, or carry no `session` attribute at all — a leftover build. It is still enumerable, so it would shift every later day index. Point `--dataset_dir` at `hdf5_data_512_trim`, not the raw root |
| `--dataset_dir ../data/hdf5_data_512` reports 3 days | the stale `t15.2026.08.14.14-45-44_tc_sbp_512` is sitting in there; use the `_trim` root |
| train reports a PER for the wrong day count | the `sessions:` list — regenerate it, do not hand-edit |
| `logits last dimension (35) does not match source_order length (41)` in the LM step | a session under `--data_dir` has **no `metadata.json`**, so evaluation fell back to guessing the official 41-class order. Only the LM path needs it, which is why `--skip_lm` works and PER/WER look fine. `ls ../data/hdf5_data_512_trim/*/metadata.json`, then copy the missing one in from `alt_models/session_metadata/` and re-run |

## Cleanup between LM runs

```bash
rm -rf trained_models/diphone_rnn_electrode_512_2day/eval_outputs
rm -rf trained_models/baseline_rnn_512_2day/eval_outputs
redis-cli -h localhost -p 6379 shutdown nosave
```

Do **not** delete anything under `language_model/pretrained_language_models/`.
