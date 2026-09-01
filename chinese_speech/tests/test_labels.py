import json
from pathlib import Path

from chinese_speech.labels import (
    LabelSchema,
    load_default_pronunciation_lexicon,
    text_to_pronunciation,
)


def test_pronunciation_uses_phrase_overrides_for_polyphonic_words():
    lexicon = load_default_pronunciation_lexicon()

    assert text_to_pronunciation("好学还书还给", lexicon) == [
        ("hao", 4),
        ("xue", 2),
        ("huan", 2),
        ("shu", 1),
        ("huan", 2),
        ("gei", 3),
    ]


def test_label_schema_builds_equal_length_syllable_and_tone_targets():
    lexicon = load_default_pronunciation_lexicon()
    schema = LabelSchema.from_texts(["牛去女", "好学还书"], lexicon)

    encoded = schema.encode_text("牛去女", lexicon)

    assert schema.syllable_to_id["<blank>"] == 0
    assert schema.tone_to_id["<blank>"] == 0
    assert schema.syllable_to_id["<sil>"] == encoded.syllable_ids[0]
    assert schema.tone_to_id["<sil>"] == encoded.tone_ids[0]
    assert len(encoded.syllable_ids) == len(encoded.tone_ids)
    assert encoded.pronunciation == [
        ("niu", 2),
        ("qu", 4),
        ("nv", 3),
    ]


def test_default_lexicon_covers_current_speech_folder_targets():
    speech_root = Path(__file__).resolve().parents[3] / "sub-01" / "speech"
    lexicon = load_default_pronunciation_lexicon()
    missing = set()

    for csv_path in sorted(speech_root.glob("*/data_*.csv")):
        for line in csv_path.read_text(encoding="utf-8-sig").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 4 or parts[0] != "trial_start":
                continue
            text = parts[3].strip()
            if not text:
                continue
            try:
                text_to_pronunciation(text, lexicon)
            except KeyError as exc:
                missing.add(str(exc))

    assert missing == set()


def test_schema_json_roundtrip_preserves_id_order():
    lexicon = load_default_pronunciation_lexicon()
    schema = LabelSchema.from_texts(["牛去女", "好学还书"], lexicon)

    restored = LabelSchema.from_json(json.loads(json.dumps(schema.to_json())))

    assert restored.syllable_to_id == schema.syllable_to_id
    assert restored.tone_to_id == schema.tone_to_id
    assert restored.encode_text("好学还书", lexicon).syllable_ids == schema.encode_text(
        "好学还书", lexicon
    ).syllable_ids
