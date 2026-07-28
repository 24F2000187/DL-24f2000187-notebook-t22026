"""Run the trained baseline model on test.csv and write a submission file.

Usage:
    python src/inference.py --test data/test.csv --model models/baseline.joblib --out submission.csv
"""

import argparse

import joblib

from utils import FEATURE_COLS, add_features, load_csv, to_long_format, top_k_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--model", default="models/baseline.joblib")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model, vectorizer = bundle["model"], bundle["vectorizer"]

    test_df = load_csv(args.test)
    long_df = to_long_format(test_df, has_labels=False)
    long_df, _ = add_features(long_df, vectorizer=vectorizer, fit_vectorizer=False)

    scores = model.predict_proba(long_df[FEATURE_COLS])[:, 1]
    submission = top_k_predictions(long_df, scores, k=3)
    submission = submission.rename(columns={"id": "ID"})[["ID", "Prediction"]]
    submission.to_csv(args.out, index=False)
    print(f"Wrote {len(submission)} predictions to {args.out}")


if __name__ == "__main__":
    main()
