"""
consistency_score.py — Explanation Consistency Score
FedTrust-Credit | Day 6

Implements explanation_consistency_score() and the per-client agreement[i] term
EXACTLY as given in Section 3.3's pseudocode. Do NOT approximate or modify the formula.

From Section 3.3:
    FUNCTION explanation_consistency_score(shap_vectors):
        pairwise_scores = []
        FOR EACH PAIR (i, j) IN CLIENTS, i != j:
            pairwise_scores.APPEND(COSINE_SIMILARITY(shap_vectors[i], shap_vectors[j]))
        RETURN MEAN(pairwise_scores)   # 1.0 = fully consistent, 0 = orthogonal
"""

import itertools
from typing import Sequence

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.
    Returns 0.0 if either vector has zero norm (handles degenerate cases gracefully).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def explanation_consistency_score(shap_vectors: Sequence[np.ndarray]) -> float:
    """
    Compute the global explanation-consistency score across all clients.

    Implements Section 3.3 pseudocode exactly:
        pairwise_scores = []
        FOR EACH PAIR (i, j) IN CLIENTS, i != j:
            pairwise_scores.APPEND(COSINE_SIMILARITY(shap_vectors[i], shap_vectors[j]))
        RETURN MEAN(pairwise_scores)

    Args:
        shap_vectors: List of 1-D numpy arrays (one per client).
                      Each is a mean-absolute-SHAP feature attribution vector.

    Returns:
        Mean pairwise cosine similarity. Range [0, 1].
        1.0 = all clients' explanations are identical.
        0.0 = all pairs are orthogonal.
    """
    pairwise_scores = []
    for i, j in itertools.combinations(range(len(shap_vectors)), 2):
        pairwise_scores.append(_cosine_similarity(shap_vectors[i], shap_vectors[j]))

    if not pairwise_scores:
        return 1.0  # Single client: trivially consistent

    return float(np.mean(pairwise_scores))


def per_client_agreement(shap_vectors: Sequence[np.ndarray]) -> list[float]:
    """
    Compute per-client agreement[i] terms used in the adjusted weight formula (Section 3.3):
        agreement[i] = MEAN(COSINE_SIMILARITY(shap_vectors[i], shap_vectors[j]) FOR EACH j != i)

    Args:
        shap_vectors: List of 1-D numpy arrays (one per client).

    Returns:
        List of floats, one per client, representing its average agreement with all others.
    """
    n = len(shap_vectors)
    agreements = []
    for i in range(n):
        scores_i = [
            _cosine_similarity(shap_vectors[i], shap_vectors[j])
            for j in range(n)
            if j != i
        ]
        agreements.append(float(np.mean(scores_i)) if scores_i else 1.0)
    return agreements
