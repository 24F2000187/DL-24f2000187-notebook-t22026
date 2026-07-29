"""mAP@k for the single-correct-answer MCQ ranking task (Kaggle-style MAP@3)."""


def average_precision_at_k(true_label, ranked_predictions, k=3):
    """1/rank of the correct label if it appears in the top k, else 0.

    Since each question has exactly one relevant label, average precision
    over the (at most k) predictions collapses to the reciprocal rank of the
    hit, truncated at k -- the standard Kaggle MAP@k for single-label tasks.
    """
    for rank, pred in enumerate(ranked_predictions[:k], start=1):
        if pred == true_label:
            return 1.0 / rank
    return 0.0


def mean_average_precision_at_k(true_labels, ranked_predictions_list, k=3):
    scores = [
        average_precision_at_k(t, preds, k=k)
        for t, preds in zip(true_labels, ranked_predictions_list)
    ]
    return sum(scores) / len(scores) if scores else 0.0
