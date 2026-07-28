"""Combine per-option scores/rankings from several models into one ensemble ranking."""

import numpy as np
import pandas as pd

from mcq_formatting import OPTION_COLS


def softmax_rows(scores_df: pd.DataFrame, temperature=1.0) -> pd.DataFrame:
    """Row-wise softmax so heterogeneous raw scores (cosine sim, logits, ...)
    become comparable per-question probability distributions over the 5 options."""
    x = scores_df[OPTION_COLS].to_numpy() / temperature
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    probs = e / e.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, columns=OPTION_COLS, index=scores_df.index)


def average_probabilities(prob_dfs, weights=None) -> pd.DataFrame:
    """Elementwise (weighted) mean of several (n_questions, 5) probability DataFrames."""
    if weights is None:
        weights = [1.0] * len(prob_dfs)
    weights = np.array(weights) / sum(weights)
    stacked = np.stack([df[OPTION_COLS].to_numpy() for df in prob_dfs], axis=0)  # (n_models, n_q, 5)
    combined = np.tensordot(weights, stacked, axes=(0, 0))  # (n_q, 5)
    return pd.DataFrame(combined, columns=OPTION_COLS, index=prob_dfs[0].index)


def long_to_wide_scores(ids, options, scores) -> pd.DataFrame:
    """(id, option, score) triples -> one row per id, one column per option letter, in id order."""
    long_df = pd.DataFrame({"id": ids, "option": options, "score": scores})
    wide = long_df.pivot(index="id", columns="option", values="score")[OPTION_COLS]
    return wide


def rankings_from_scores(scores_df: pd.DataFrame):
    """One ranked list of option letters (best first) per row."""
    return scores_df[OPTION_COLS].apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1).tolist()


def borda_fusion(rankings_list):
    """Borda count: each model awards (num_options - rank) points per option; sum across models."""
    n = len(rankings_list[0])
    fused = []
    for i in range(n):
        points = {opt: 0 for opt in OPTION_COLS}
        for rankings in rankings_list:
            for rank, opt in enumerate(rankings[i]):
                points[opt] += len(OPTION_COLS) - rank
        fused.append(sorted(points, key=points.get, reverse=True))
    return fused


def reciprocal_rank_fusion(rankings_list, k=60):
    """RRF: score(option) = sum over models of 1 / (k + rank). Robust to score-scale differences."""
    n = len(rankings_list[0])
    fused = []
    for i in range(n):
        scores = {opt: 0.0 for opt in OPTION_COLS}
        for rankings in rankings_list:
            for rank, opt in enumerate(rankings[i], start=1):
                scores[opt] += 1.0 / (k + rank)
        fused.append(sorted(scores, key=scores.get, reverse=True))
    return fused
