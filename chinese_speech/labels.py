from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Pronunciation = List[Tuple[str, int]]
PronunciationLexicon = Dict[str, Pronunciation]

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
BLANK_TOKEN = "<blank>"
SIL_TOKEN = "<sil>"
NEUTRAL_TONE = 5


@dataclass(frozen=True)
class EncodedText:
    text: str
    pronunciation: Pronunciation
    syllable_ids: List[int]
    tone_ids: List[int]


@dataclass(frozen=True)
class LabelSchema:
    syllable_to_id: Dict[str, int]
    tone_to_id: Dict[str, int]

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        lexicon: Mapping[str, Pronunciation],
    ) -> "LabelSchema":
        syllable_to_id = {BLANK_TOKEN: 0}
        for text in texts:
            for syllable, _tone in text_to_pronunciation(text, lexicon):
                if syllable not in syllable_to_id:
                    syllable_to_id[syllable] = len(syllable_to_id)
        syllable_to_id[SIL_TOKEN] = len(syllable_to_id)

        tone_to_id = {BLANK_TOKEN: 0}
        for tone in (1, 2, 3, 4, NEUTRAL_TONE):
            tone_to_id[str(tone)] = len(tone_to_id)
        tone_to_id[SIL_TOKEN] = len(tone_to_id)

        return cls(syllable_to_id=syllable_to_id, tone_to_id=tone_to_id)

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "LabelSchema":
        labels = data.get("labels", data)
        if not isinstance(labels, Mapping):
            raise ValueError("Label schema JSON must contain a mapping.")
        return cls(
            syllable_to_id={str(k): int(v) for k, v in labels["syllable_to_id"].items()},
            tone_to_id={str(k): int(v) for k, v in labels["tone_to_id"].items()},
        )

    def to_json(self) -> Dict[str, Dict[str, int]]:
        return {
            "syllable_to_id": dict(self.syllable_to_id),
            "tone_to_id": dict(self.tone_to_id),
            "id_to_syllable": {str(v): k for k, v in self.syllable_to_id.items()},
            "id_to_tone": {str(v): k for k, v in self.tone_to_id.items()},
        }

    def encode_text(
        self,
        text: str,
        lexicon: Mapping[str, Pronunciation],
    ) -> EncodedText:
        pronunciation = text_to_pronunciation(text, lexicon)
        syllable_ids = [self.syllable_to_id[SIL_TOKEN]]
        tone_ids = [self.tone_to_id[SIL_TOKEN]]
        for syllable, tone in pronunciation:
            syllable_ids.append(self.syllable_to_id[syllable])
            tone_ids.append(self.tone_to_id[str(tone)])
        syllable_ids.append(self.syllable_to_id[SIL_TOKEN])
        tone_ids.append(self.tone_to_id[SIL_TOKEN])
        return EncodedText(
            text=text,
            pronunciation=pronunciation,
            syllable_ids=syllable_ids,
            tone_ids=tone_ids,
        )


def normalize_chinese_text(text: str) -> str:
    return "".join(CJK_RE.findall(unicodedata.normalize("NFKC", str(text)).strip()))


def _normalize_syllable(syllable: str) -> str:
    out = unicodedata.normalize("NFKC", str(syllable)).strip().lower()
    out = out.replace("u:", "v").replace("ü", "v")
    return out


def _parse_pronunciation(value: Sequence[object]) -> Pronunciation:
    pronunciation: Pronunciation = []
    for item in value:
        if isinstance(item, str):
            match = re.fullmatch(r"([a-züv:]+)([1-5])", item.strip().lower())
            if not match:
                raise ValueError(f"Invalid pronunciation item: {item!r}")
            pronunciation.append((_normalize_syllable(match.group(1)), int(match.group(2))))
        else:
            syllable, tone = item  # type: ignore[misc]
            pronunciation.append((_normalize_syllable(str(syllable)), int(tone)))
    return pronunciation


def load_default_pronunciation_lexicon(override_path: Optional[Path] = None) -> PronunciationLexicon:
    lexicon = {text: _parse_pronunciation(items) for text, items in _DEFAULT_LEXICON.items()}
    if override_path is not None:
        override_data = json.loads(Path(override_path).read_text(encoding="utf-8"))
        for text, items in override_data.items():
            lexicon[normalize_chinese_text(text)] = _parse_pronunciation(items)
    return lexicon


def text_to_pronunciation(
    text: str,
    lexicon: Mapping[str, Pronunciation],
) -> Pronunciation:
    normalized = normalize_chinese_text(text)
    pronunciation: Pronunciation = []
    position = 0
    max_key_len = max((len(k) for k in lexicon.keys()), default=1)

    while position < len(normalized):
        match_key = None
        max_span = min(max_key_len, len(normalized) - position)
        for span in range(max_span, 0, -1):
            candidate = normalized[position : position + span]
            if candidate in lexicon:
                match_key = candidate
                break
        if match_key is None:
            raise KeyError(f"Missing pronunciation for {normalized[position]!r} in {normalized!r}")
        pronunciation.extend(lexicon[match_key])
        position += len(match_key)
    return pronunciation


_DEFAULT_LEXICON: Dict[str, Sequence[object]] = {
    "好学": ["hao4", "xue2"],
    "还书": ["huan2", "shu1"],
    "还给": ["huan2", "gei3"],
    "还想": ["hai2", "xiang3"],
    "还要": ["hai2", "yao4"],
    "还在": ["hai2", "zai4"],
    "还能": ["hai2", "neng2"],
    "还很": ["hai2", "hen3"],
    "学生": ["xue2", "sheng5"],
    "医生": ["yi1", "sheng1"],
    "老师": ["lao3", "shi1"],
    "什么": ["shen2", "me5"],
    "晚上": ["wan3", "shang5"],
    "早上": ["zao3", "shang5"],
    "一点": ["yi4", "dian3"],
    "一": ["yi1"],
    "上": ["shang4"],
    "下": ["xia4"],
    "不": ["bu4"],
    "么": ["me5"],
    "书": ["shu1"],
    "了": ["le5"],
    "事": ["shi4"],
    "人": ["ren2"],
    "什": ["shen2"],
    "今": ["jin1"],
    "他": ["ta1"],
    "以": ["yi3"],
    "们": ["men5"],
    "优": ["you1"],
    "你": ["ni3"],
    "使": ["shi3"],
    "做": ["zuo4"],
    "八": ["ba1"],
    "再": ["zai4"],
    "写": ["xie3"],
    "到": ["dao4"],
    "前": ["qian2"],
    "医": ["yi1"],
    "十": ["shi2"],
    "去": ["qu4"],
    "可": ["ke3"],
    "右": ["you4"],
    "吃": ["chi1"],
    "后": ["hou4"],
    "吗": ["ma5"],
    "听": ["ting1"],
    "和": ["he2"],
    "咬": ["yao3"],
    "喝": ["he1"],
    "回": ["hui2"],
    "在": ["zai4"],
    "坐": ["zuo4"],
    "外": ["wai4"],
    "多": ["duo1"],
    "天": ["tian1"],
    "头": ["tou2"],
    "女": ["nv3"],
    "她": ["ta1"],
    "好": ["hao3"],
    "姨": ["yi2"],
    "学": ["xue2"],
    "家": ["jia1"],
    "少": ["shao3"],
    "左": ["zuo3"],
    "师": ["shi1"],
    "开": ["kai1"],
    "很": ["hen3"],
    "想": ["xiang3"],
    "意": ["yi4"],
    "我": ["wo3"],
    "打": ["da3"],
    "把": ["ba3"],
    "拿": ["na2"],
    "摇": ["yao2"],
    "放": ["fang4"],
    "早": ["zao3"],
    "明": ["ming2"],
    "是": ["shi4"],
    "晚": ["wan3"],
    "有": ["you3"],
    "来": ["lai2"],
    "校": ["xiao4"],
    "水": ["shui3"],
    "没": ["mei2"],
    "游": ["you2"],
    "点": ["dian3"],
    "牙": ["ya2"],
    "牛": ["niu2"],
    "生": ["sheng1"],
    "疼": ["teng2"],
    "痛": ["tong4"],
    "的": ["de5"],
    "看": ["kan4"],
    "睡": ["shui4"],
    "知": ["zhi1"],
    "答": ["da2"],
    "累": ["lei4"],
    "给": ["gei3"],
    "老": ["lao3"],
    "肥": ["fei2"],
    "能": ["neng2"],
    "腰": ["yao1"],
    "药": ["yao4"],
    "要": ["yao4"],
    "让": ["rang4"],
    "讲": ["jiang3"],
    "说": ["shuo1"],
    "读": ["du2"],
    "谁": ["shei2"],
    "走": ["zou3"],
    "边": ["bian1"],
    "还": ["hai2"],
    "这": ["zhe4"],
    "道": ["dao4"],
    "那": ["na4"],
    "里": ["li3"],
    "问": ["wen4"],
    "院": ["yuan4"],
    "饭": ["fan4"],
}
