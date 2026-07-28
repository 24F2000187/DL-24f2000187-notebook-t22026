"""Train a baseline logistic-regression ranker for the MCQ answer-ranking task.

For every question, each of the 5 options (A-E) becomes one training row with a
binary label (1 = correct answer). The model scores how likely each option is
correct; at inference time the top-3 scored options are submitted.

Usage:
    python src/train.py --train data/train.csv --model-out models/baseline.joblib
"""

import argparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from utils import FEATURE_COLS, add_features, load_csv, to_long_format, top_k_predictions


def top3_accuracy(long_df, scores, true_answers):
    preds = top_k_predictions(long_df, scores, k=3)
    merged = preds.merge(true_answers, on="id")
    hit = merged.apply(lambda r: r["answer"] in r["Prediction"].split(), axis=1)
    return hit.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--model-out", default="models/baseline.joblib")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_csv(args.train)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["id"]))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    train_long = to_long_format(train_df, has_labels=True)
    train_long, vectorizer = add_features(train_long, fit_vectorizer=True)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_long[FEATURE_COLS], train_long["label"])

    val_long = to_long_format(val_df, has_labels=True)
    val_long, _ = add_features(val_long, vectorizer=vectorizer, fit_vectorizer=False)
    val_scores = model.predict_proba(val_long[FEATURE_COLS])[:, 1]

    acc = top3_accuracy(val_long, val_scores, val_df[["id", "answer"]])
    print(f"Validation top-3 accuracy: {acc:.4f}")

    joblib.dump({"model": model, "vectorizer": vectorizer}, args.model_out)
    print(f"Saved model + vectorizer to {args.model_out}")


if __name__ == "__main__":
    main()
