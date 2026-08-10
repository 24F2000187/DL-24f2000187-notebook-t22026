import os
import sys

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
    <style>
    .title {
        font-family: 'Inter', 'Outfit', sans-serif;
        color: #0f172a;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    div.stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        transition: background-color 0.2s ease, transform 0.1s ease !important;
    }
    div.stButton > button:hover {
        background-color: #2563eb !important;
        color: white !important;
        transform: translateY(-1px);
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }
    </style>
""", unsafe_allow_html=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils import to_long_format, add_features, top_k_predictions, FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "baseline_final.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "baseline.joblib")

@st.cache_resource
def load_model(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["vectorizer"]

_model, _vectorizer = load_model(MODEL_PATH)


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


st.markdown('<h1 class="title">🧠 Smart MCQ Solver</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">TF-IDF + LogisticRegression baseline demo to predict option correctness</p>', 
    unsafe_allow_html=True
)

st.markdown("### 📝 Enter MCQ Details")

prompt = st.text_area(
    "Question prompt", 
    placeholder="Enter the question or context here...", 
    height=120
)

st.write("Provide the five potential options:")
col1, col2 = st.columns(2)
with col1:
    option_a = st.text_input("Option A", placeholder="First option")
    option_b = st.text_input("Option B", placeholder="Second option")
    option_c = st.text_input("Option C", placeholder="Third option")
with col2:
    option_d = st.text_input("Option D", placeholder="Fourth option")
    option_e = st.text_input("Option E", placeholder="Fifth option")

st.markdown("---")

if st.button("Solve MCQ 🚀", use_container_width=True):
    if not prompt.strip() or not option_a.strip() or not option_b.strip() or not option_c.strip() or not option_d.strip() or not option_e.strip():
        st.warning("⚠️ Please fill out the question prompt and all five options.")
    else:
        with st.spinner("Running inference pipeline..."):
            top3, breakdown = predict(prompt, option_a, option_b, option_c, option_d, option_e)
        
        st.success("Inference complete!")
        
        res1, res2 = st.columns(2)
        with res1:
            st.markdown("#### 🏆 Top-3 Prediction")
            st.info(f"👉 **{top3}**")
            st.caption("Standard competition submission format (predicted choices ranked by confidence).")
        with res2:
            st.markdown("#### 📊 Confidence Scores")
            st.code(breakdown, language="text")
