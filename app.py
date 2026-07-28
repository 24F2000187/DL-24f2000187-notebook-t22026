"""Gradio demo for the MCQ answer-ranking baseline.

Enter a question and its 5 options; the app ranks them by the baseline
TF-IDF + LogisticRegression model's predicted "correctness" probability
and shows the top-3 in this competition's submission format.

Run locally:
    python app.py

Deploys as-is to a Hugging Face Space (app.py + requirements.txt at the repo root).
"""

import os
import sys

import gradio as gr
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils import to_long_format, add_features, top_k_predictions, FEATURE_COLS  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "baseline_final.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "baseline.joblib")

_bundle = joblib.load(MODEL_PATH)
_model, _vectorizer = _bundle["model"], _bundle["vectorizer"]


def predict(prompt, option_a, option_b, option_c, option_d, option_e):
    options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d, "E": option_e}
    if not prompt.strip() or any(not v.strip() for v in options.values()):
        return "Please fill in the question and all 5 options.", ""

    row = {"id": 1, "prompt": prompt, **options}
    df = pd.DataFrame([row])

    long_df = to_long_format(df, has_labels=False)
    long_df, _ = add_features(long_df, vectorizer=_vectorizer, fit_vectorizer=False)
    scores = _model.predict_proba(long_df[FEATURE_COLS])[:, 1]

    ranked = top_k_predictions(long_df, scores, k=5).iloc[0]["Prediction"].split()
    score_by_option = dict(zip(long_df["option"], scores))

    top3 = " ".join(ranked[:3])
    breakdown = "\n".join(f"{opt}: {score_by_option[opt]:.3f}" for opt in ranked)
    return top3, breakdown


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Question (prompt)", lines=2),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=[
        gr.Textbox(label="Top-3 prediction (submission format)"),
        gr.Textbox(label="All 5 options ranked, with score", lines=5),
    ],
    title="Smart MCQ Solver -- Baseline Demo",
    description=(
        "TF-IDF + LogisticRegression baseline (see README.md / notebooks/final_notebook.ipynb "
        "for why this model, not a fancier one, was chosen for the final submission)."
    ),
)

if __name__ == "__main__":
    demo.launch()
