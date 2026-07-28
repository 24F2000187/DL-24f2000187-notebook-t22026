"""Shared data loading and feature-engineering helpers for the MCQ answer-ranking task."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

OPTION_COLS = ["A", "B", "C", "D", "E"]


def load_csv(path):
    return pd.read_csv(path)


def _word_overlap(prompt, option):
    prompt_words = set(str(prompt).lower().split())
    option_words = set(str(option).lower().split())
    if not option_words:
        return 0.0
    return len(prompt_words & option_words) / len(option_words)


def to_long_format(df, has_labels):
    """Reshape one row-per-question (prompt + A..E) into one row per (id, option)."""
    records = []
    for row in df.itertuples(index=False):
        row = row._asdict()
        for letter in OPTION_COLS:
            records.append(
                {
                    "id": row["id"],
                    "option": letter,
                    "prompt": row["prompt"],
                    "text": row[letter],
                    "label": int(row.get("answer") == letter) if has_labels else None,
                }
            )
    long_df = pd.DataFrame.from_records(records)
    if not has_labels:
        long_df = long_df.drop(columns=["label"])
    return long_df


def add_features(long_df, vectorizer: TfidfVectorizer = None, fit_vectorizer=False):
    """Add length, word-overlap and TF-IDF cosine-similarity features to a long-format frame."""
    long_df = long_df.copy()
    long_df["prompt_len"] = long_df["prompt"].str.len()
    long_df["option_len"] = long_df["text"].str.len()
    long_df["len_ratio"] = long_df["option_len"] / long_df["prompt_len"].clip(lower=1)
    long_df["word_overlap"] = [
        _word_overlap(p, t) for p, t in zip(long_df["prompt"], long_df["text"])
    ]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")

    corpus = pd.concat([long_df["prompt"], long_df["text"]], ignore_index=True)
    if fit_vectorizer:
        vectorizer.fit(corpus.fillna(""))

    prompt_vecs = vectorizer.transform(long_df["prompt"].fillna(""))
    option_vecs = vectorizer.transform(long_df["text"].fillna(""))
    # row-wise cosine similarity between two sparse TF-IDF matrices
    numer = prompt_vecs.multiply(option_vecs).sum(axis=1).A1
    denom = (
        (prompt_vecs.multiply(prompt_vecs).sum(axis=1).A1 ** 0.5)
        * (option_vecs.multiply(option_vecs).sum(axis=1).A1 ** 0.5)
    )
    long_df["tfidf_cos_sim"] = numer / (denom + 1e-9)

    return long_df, vectorizer


FEATURE_COLS = ["prompt_len", "option_len", "len_ratio", "word_overlap", "tfidf_cos_sim"]


def top_k_predictions(long_df, scores, k=3):
    """Given per-(id, option) scores, return a DataFrame of id -> space-joined top-k option letters."""
    out = long_df[["id", "option"]].copy()
    out["score"] = scores
    ranked = (
        out.sort_values(["id", "score"], ascending=[True, False])
        .groupby("id")
        .head(k)
        .groupby("id")["option"]
        .apply(lambda opts: " ".join(opts))
        .reset_index()
        .rename(columns={"option": "Prediction"})
    )
    return ranked
