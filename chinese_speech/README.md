# Chinese Syllable/Tone Pipeline

This folder is a Mandarin-specific side path for the copied `Speech` repo. It does not modify the English 50-word preprocessing, model training, or language-model runtime.

## Current Scope

- The preprocessing builder reads session folders from `../sub-01/speech`, not `speech.zip`.
- Uses `trial_data.mat` sorted-spike data and aggregates valid units to 256 physical electrodes.
- Uses `state_bin == 2` as the read/intended-speech epoch.
- Builds Mandarin labels as two aligned CTC streams:
  - `seq_syllable_ids`: pinyin base syllables, no tone number.
  - `seq_tone_ids`: tone numbers `1..5`.
- Keeps `seq_class_ids` as an alias of `seq_syllable_ids` so older single-head utilities can still inspect the file.
- Skips blank/no-action trials by default and records them in `trial_manifest.csv`.
- The training script reads finished HDF5 files from `data/hdf5_chinese` and does not require raw `sub-01/speech` on the GPU machine.

## Build

Dry-run the available sessions:

```bash
python -m chinese_speech.builder --dry-run
```

Build one session:

```bash
python -m chinese_speech.builder --session 2026-08-31-S2 --overwrite
```

Build all discovered sessions:

```bash
python -m chinese_speech.builder --overwrite
```

The default `--output-root` is `data/hdf5_chinese`.

Final handoff output for training:

```text
Speech/data/hdf5_chinese/
```

Each session contains:

```text
data_train.hdf5
data_val.hdf5
data_test.hdf5
trial_manifest.csv
metadata.json
```

## HDF5 Trial Fields

Each written trial contains:

```text
input_features      T x 256 float32
seq_class_ids       syllable stream alias
seq_syllable_ids    syllable CTC targets
seq_tone_ids        tone CTC targets
transcription       UTF-8 bytes with zero terminator
```

Important attrs:

```text
n_time_steps
seq_len
tone_seq_len
sentence_label
pronunciation
target_syllables
target_tones
```

## Training

Train the independent Chinese dual-stream decoder from the repo root:

```bash
python -m chinese_speech.train_dual_stream --config chinese_speech/train_config.yaml
```

For a short smoke run:

```bash
python -m chinese_speech.train_dual_stream --config chinese_speech/train_config.yaml --num-batches 1 --output-dir chinese_speech/trained_models/smoke
```

The training script builds one global syllable/tone label map across all configured sessions and remaps each session's local HDF5 ids before computing CTC loss. This is required because the per-session HDF5 metadata can assign different local ids to the same syllable.

Internally it uses `DualStreamGRUDecoder` for a shared recurrent encoder with two output heads:

```python
from chinese_speech.dual_stream_model import DualStreamGRUDecoder
from chinese_speech.losses import dual_stream_ctc_loss

model = DualStreamGRUDecoder(
    neural_dim=256,
    n_units=768,
    n_days=n_days,
    n_syllable_classes=n_syllable_classes,
    n_tone_classes=n_tone_classes,
)

out = model(features, day_idx)
loss, parts = dual_stream_ctc_loss(out, batch)
```

## Open Points

- Pronunciation is currently a built-in lexicon covering the present `sub-01/speech` corpus, with phrase overrides for cases such as `好学`, `还书`, and `还给`.
- Tone sandhi is not globally applied. Only explicit phrase overrides are used.
- Neural features currently reuse the existing sorted-spike 256D electrode aggregation path. The raw NS6 512D TC+SBP path is a separate later step.
- Chinese LM integration is intentionally not wired yet; start with LM-off syllable/tone decoding and add a Chinese lexicon/LM only after labels and model behavior are stable.
