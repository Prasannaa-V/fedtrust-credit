"""
shap_explanation.py — Per-client SHAP explanation vector generation
FedTrust-Credit | Day 5

Computes per-client SHAP vectors (mean absolute feature attributions) after local training.
Uses SHAP TreeExplainer for LightGBM models (exact, not approximate).

Background sample: client's local training data, capped at 200 samples (DECISIONS.md D-006).
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_vector(model, X_local: pd.DataFrame, max_background: int = 200) -> np.ndarray:
    """
    Compute the mean absolute SHAP feature attribution vector for a trained model.

    Called on each client immediately after local training (before sending weights to aggregator).
    The resulting 1-D vector (one value per feature) is what the aggregator uses to compute
    cross-client explanation consistency (Section 3.3).

    Args:
        model: Trained LightGBM (or sklearn-compatible) classifier.
        X_local: Client's local data (training set) as a DataFrame with named features.
        max_background: Maximum number of samples to compute SHAP values on.
                        Capped to prevent compute blowup per round (D-006).

    Returns:
        1-D numpy array of shape (n_features,) — mean absolute SHAP values per feature.
    """
    # Sample if too large
    if len(X_local) > max_background:
        sample = X_local.sample(n=max_background, random_state=42)
    else:
        sample = X_local

    # TreeExplainer: fast, exact for LightGBM tree models
    # Use default tree_path_dependent mode (no background data needed) for speed
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample.values)
    except Exception:
        # Fallback to KernelExplainer for non-tree models (slower but model-agnostic)
        background_small = sample.iloc[:50]
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background_small.values)
        shap_values = explainer.shap_values(sample.values, nsamples=50)

    # For binary classification, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class (default=1)

    # Mean absolute SHAP value per feature — the client's "explanation vector" (D-010)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return mean_abs_shap.astype(np.float32)
