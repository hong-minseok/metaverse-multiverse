import re
import sys
from pathlib import Path

import pandas as pd

TYPO_CORRECTIONS = {
    "습ㄴ다": "습니다", "습ㄷ다": "습니다", "였습ㄴ다": "였습니다",
    "었습ㄴ다": "었습니다", "합ㄴ다": "합니다", "입ㄴ다": "입니다",
    "있엇": "있었", "없엇": "없었", "했엇": "했었", "갔엇": "갔었",
    "했ㅆ습": "했습", "갔ㅆ습": "갔습", "있ㅆ습": "있습",
    "되요": "돼요", "됬": "됐",
}
FILLERS = ("음", "어", "그", "뭐", "아", "에", "으음", "음음", "어어")
REVERSED_RSES_ITEMS = (2, 5, 6, 8, 9)
MINIMUM_CHARACTERS = 10


def clean_speech(text):
    if pd.isna(text):
        return ""
    text = str(text)
    for typo, correction in TYPO_CORRECTIONS.items():
        text = text.replace(typo, correction)
    text = re.sub(rf"\b({'|'.join(FILLERS)})\b", "", text)
    text = re.sub(r"([.,])\1+", r"\1", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return " ".join(text.split())


def score_scales(raw):
    phq9 = raw[[f"PHQ-9_{item}" for item in range(1, 10)]]
    rses = raw[[f"Rosenberg_{item}" for item in range(1, 11)]].copy()
    reversed_columns = [f"Rosenberg_{item}" for item in REVERSED_RSES_ITEMS]
    rses[reversed_columns] = 5 - rses[reversed_columns]
    return phq9.sum(axis=1, skipna=False), rses.sum(axis=1, skipna=False)


raw_path, output_path = map(Path, sys.argv[1:3])
raw = pd.read_excel(raw_path)
phq9, rses = score_scales(raw)
data = pd.DataFrame({
    "sample_id": raw["ID"],
    "age": raw["AGE"],
    "male": raw["UserGender"],
    "PHQ9": phq9,
    "RSES": rses,
    "negative": raw["Negative_Memory"].map(clean_speech),
    "positive": raw["Positive_Memory"].map(clean_speech),
})
long_enough = (data[["negative", "positive"]].map(len) >= MINIMUM_CHARACTERS).all(axis=1)
data = data[long_enough]
output_path.parent.mkdir(parents=True, exist_ok=True)
data.sort_values("sample_id").to_csv(output_path, index=False)
