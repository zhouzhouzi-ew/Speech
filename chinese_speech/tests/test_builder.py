from pathlib import Path

import h5py
import numpy as np

from chinese_speech.builder import (
    ChineseSpeechBuilder,
    compute_normalization_stats_by_date,
    default_output_root,
    discover_chinese_sessions,
)


def _write_minimal_trial_data(path, trial_values):
    rows = []
    trial_mask = []
    for trial_num, value in trial_values:
        rows.append(np.full((40, 1), value, dtype=np.float32))
        trial_mask.append(np.full(40, trial_num, dtype=np.int32))

    with h5py.File(path, "w") as h5:
        array_channel_unit = np.zeros((4, 1), dtype=np.float32)
        array_channel_unit[3, 0] = 1
        h5.create_dataset("array_channel_unit", data=array_channel_unit)
        h5.create_dataset("neuron_mask", data=np.array([1], dtype=np.int32))
        h5.create_dataset("spike_bin", data=np.concatenate(rows, axis=0))
        h5.create_dataset("trial_mask", data=np.concatenate(trial_mask))
        h5.create_dataset("state_bin", data=np.full(40 * len(trial_values), 2, dtype=np.int32))


def _write_trial_start_csv(path, labels):
    lines = ["EventType,Data1,Data2,Data3"]
    for trial_num, label in enumerate(labels, start=1):
        lines.append(f"trial_start,{trial_num},{trial_num},{label}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_default_output_root_is_data_hdf5_chinese():
    project_root = Path(__file__).resolve().parents[2]

    assert default_output_root(project_root) == project_root / "data" / "hdf5_chinese"


def test_discover_chinese_sessions_reads_folder_tree_not_zip():
    root = Path(__file__).resolve().parents[3] / "sub-01" / "speech"
    sessions = discover_chinese_sessions(root)

    names = [session.session_name for session in sessions]

    assert "2026-08-31-S1" in names
    assert "2026-08-31-S2" in names
    assert all(session.csv_path.exists() for session in sessions)
    assert all(session.trial_data_path.exists() for session in sessions)


def test_builder_writes_dual_stream_labels_and_english_compatible_fields(tmp_path):
    root = Path(__file__).resolve().parents[3] / "sub-01" / "speech" / "2026-08-31-S2"
    builder = ChineseSpeechBuilder(
        session_dir=root,
        output_root=tmp_path,
        split_seed=7,
        include_diagnostic_trials=True,
        overwrite=True,
    )

    result = builder.build()
    session_dir = Path(result["output_dir"])

    assert session_dir.name.startswith("t15.2026.08.31.")
    with h5py.File(session_dir / "data_train.hdf5", "r") as f:
        trial = f["trial_0000"]
        assert "input_features" in trial
        assert "seq_class_ids" in trial
        assert "seq_syllable_ids" in trial
        assert "seq_tone_ids" in trial
        assert "transcription" in trial
        assert trial.attrs["feature_type"].startswith("syllable_tone")
        assert "fixed_day_first20" in trial.attrs["feature_type"]
        assert "prev20_non_test" not in trial.attrs["feature_type"]
        assert trial.attrs["seq_len"] == len(trial["seq_syllable_ids"][:])
        assert trial.attrs["seq_len"] == len(trial["seq_tone_ids"][:])
        assert trial.attrs["n_time_steps"] == trial["input_features"].shape[0]

    with open(session_dir / "metadata.json", encoding="utf-8") as f:
        metadata = f.read()
    assert "syllable_to_id" in metadata
    assert "tone_to_id" in metadata
    assert "english_compatible" in metadata


def test_builder_marks_no_action_trials_without_labels(tmp_path):
    root = Path(__file__).resolve().parents[3] / "sub-01" / "speech" / "2026-08-31-S1"
    builder = ChineseSpeechBuilder(
        session_dir=root,
        output_root=tmp_path,
        split_seed=7,
        include_diagnostic_trials=True,
        overwrite=True,
    )

    result = builder.build()
    manifest_path = Path(result["output_dir"]) / "trial_manifest.csv"
    manifest = manifest_path.read_text(encoding="utf-8")

    assert "no_action" in manifest
    assert "blank" in manifest


def test_builder_excludes_single_character_diagnostic_trials_by_default(tmp_path):
    session_dir = tmp_path / "2026-09-01-S1"
    session_dir.mkdir()
    _write_trial_start_csv(
        session_dir / "data_synthetic.csv",
        ["\u725b", "\u6211\u6ca1\u6709\u836f"],
    )
    _write_minimal_trial_data(session_dir / "trial_data.mat", [(1, 1), (2, 3)])

    result = ChineseSpeechBuilder(
        session_dir=session_dir,
        output_root=tmp_path / "out",
        overwrite=True,
        n_electrodes=1,
        val_fraction=0.0,
        test_fraction=0.0,
    ).build()

    output_dir = Path(result["output_dir"])
    with h5py.File(output_dir / "data_train.hdf5", "r") as h5:
        assert list(h5.keys()) == ["trial_0000"]
        assert h5["trial_0000"].attrs["trial_num"] == 2

    manifest = (output_dir / "trial_manifest.csv").read_text(encoding="utf-8")
    assert "excluded_diagnostic" in manifest
    assert "\u725b" in manifest


def test_builder_uses_fixed_first_trials_zscore_for_train_val_and_test(tmp_path, monkeypatch):
    session_dir = tmp_path / "2026-09-01-S1"
    session_dir.mkdir()
    _write_trial_start_csv(session_dir / "data_synthetic.csv", ["\u6211\u6ca1\u6709\u836f"] * 3)
    _write_minimal_trial_data(session_dir / "trial_data.mat", [(1, 1), (2, 10), (3, 20)])

    monkeypatch.setattr(
        "chinese_speech.builder._split_trials",
        lambda *args, **kwargs: {1: "test", 2: "train", 3: "val"},
    )
    builder = ChineseSpeechBuilder(
        session_dir=session_dir,
        output_root=tmp_path / "out",
        overwrite=True,
        n_electrodes=1,
    )

    result = builder.build()

    with h5py.File(Path(result["output_dir"]) / "data_train.hdf5", "r") as h5:
        train_features = h5["trial_0000"]["input_features"][:]
    with h5py.File(Path(result["output_dir"]) / "data_test.hdf5", "r") as h5:
        test_features = h5["trial_0000"]["input_features"][:]

    calibration = np.asarray([20.0, 20.0, 200.0, 200.0, 400.0, 400.0])
    expected_train = (200.0 - calibration.mean()) / calibration.std()
    expected_test = (20.0 - calibration.mean()) / calibration.std()
    assert np.allclose(train_features, [[expected_train], [expected_train]])
    assert np.allclose(test_features, [[expected_test], [expected_test]])


def test_daily_zscore_stats_use_first_twenty_included_trials_across_sessions(tmp_path):
    first = tmp_path / "2026-09-01-S2"
    first.mkdir()
    _write_trial_start_csv(first / "data_synthetic.csv", ["\u6211\u6ca1\u6709\u836f"] * 20)
    _write_minimal_trial_data(first / "trial_data.mat", [(idx, 1) for idx in range(1, 21)])

    second = tmp_path / "2026-09-01-S4"
    second.mkdir()
    _write_trial_start_csv(second / "data_synthetic.csv", ["\u4f60\u60f3\u5403\u4ec0\u4e48"])
    _write_minimal_trial_data(second / "trial_data.mat", [(1, 10)])

    sessions = discover_chinese_sessions(tmp_path)
    stats_by_date = compute_normalization_stats_by_date(sessions, n_electrodes=1)

    assert stats_by_date["2026-09-01"].n_trials == 20

    result = ChineseSpeechBuilder(
        session_dir=second,
        output_root=tmp_path / "out",
        normalization_stats=stats_by_date["2026-09-01"],
        overwrite=True,
        n_electrodes=1,
        val_fraction=0.0,
        test_fraction=0.0,
    ).build()

    with h5py.File(Path(result["output_dir"]) / "data_train.hdf5", "r") as h5:
        features = h5["trial_0000"]["input_features"][:]
    assert features.tolist() == [[20.0], [20.0]]
