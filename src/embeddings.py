"""Pretrained transformer embeddings (context-aware, mean-pooled) for the MCQ task."""

import numpy as np
import torch


def embed_texts(texts, tokenizer, model, batch_size=128, max_length=96):
    """Mean-pool the last hidden state (masking padding) into one vector per text."""
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vectors.append(pooled.numpy())
    return np.concatenate(vectors, axis=0)


def cosine_sim_rows(a, b):
    """Row-wise cosine similarity between two equally-shaped (N, D) arrays."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.sum(a_norm * b_norm, axis=1)


def rank_options_from_scores(scores_df):
    """scores_df: DataFrame indexed like the questions, one column per option letter."""
    return scores_df.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
