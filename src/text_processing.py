"""Text cleaning, tokenization and missing-data handling for the MCQ dataset."""

import re

import pandas as pd

OPTION_COLS = ["A", "B", "C", "D", "E"]
TEXT_COLS = ["prompt"] + OPTION_COLS

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def report_missing(df: pd.DataFrame) -> pd.Series:
    """Per-column missing-value counts for the text columns present in df."""
    cols = [c for c in TEXT_COLS if c in df.columns]
    return df[cols].isnull().sum()


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing a prompt or any option -- an MCQ row is unusable without all of them."""
    cols = [c for c in TEXT_COLS if c in df.columns]
    return df.dropna(subset=cols).reset_index(drop=True)


def clean_text(text: str) -> str:
    """Lowercase, strip non-alphanumeric characters, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Whitespace/regex tokenizer over already-cleaned text (falls back to cleaning first)."""
    return _TOKEN_RE.findall(text.lower())


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with missing rows dropped and cleaned/tokenized text columns added."""
    df = drop_missing_rows(df)
    df = df.copy()
    cols = [c for c in TEXT_COLS if c in df.columns]
    for col in cols:
        df[f"{col}_clean"] = df[col].map(clean_text)
        df[f"{col}_tokens"] = df[f"{col}_clean"].map(tokenize)
    return df
