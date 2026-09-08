"""
centralized_baseline.py — Centralized Training Baseline
FedTrust-Credit | Day 8

Trains on all data pooled (no federation) — the privacy-free upper-bound reference.
Reports accuracy, AUC-ROC, F1 on the same held-out test split used by federated runs.
Also computes explanation-consistency score by splitting centralized predictions
back along client boundaries (per Section 4.3).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from consistency_score import explanation_consistency_score
from shap_explanation import compute_shap_vector


def run_centralized_baseline(
    client_data: dict,  # {client_id: {"X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...}}
    results_dir: str | Path = "results/centralized_baseline",
) -> dict:
    """
    Pool all client data and train a single centralized model.
    Evaluate on the pooled test set and per-client subsets.
    Compute explanation-consistency score by partitioning centralized SHAP along client lines.

    Returns a dict with all metric results.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Pool all training data ────────────────────────────────────────────────
    X_train_all = pd.concat([d["X_train"] for d in client_data.values()], axis=0)
    y_train_all = pd.concat([d["y_train"] for d in client_data.values()], axis=0)
    X_test_all  = pd.concat([d["X_test"]  for d in client_data.values()], axis=0)
    y_test_all  = pd.concat([d["y_test"]  for d in client_data.values()], axis=0)

    print(f"[centralized] Pooled train: {len(X_train_all):,} rows | test: {len(X_test_all):,} rows")

    # ── Train centralized model ───────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train_all, y_train_all)

    # ── Pooled evaluation ─────────────────────────────────────────────────────
    y_pred_all = model.predict(X_test_all)
    y_prob_all = model.predict_proba(X_test_all)[:, 1]
    pooled_metrics = {
        "accuracy": float(accuracy_score(y_test_all, y_pred_all)),
        "auc":      float(roc_auc_score(y_test_all, y_prob_all)),
        "f1":       float(f1_score(y_test_all, y_pred_all, zero_division=0)),
    }
    print(f"[centralized] Pooled: acc={pooled_metrics['accuracy']:.4f} | auc={pooled_metrics['auc']:.4f} | f1={pooled_metrics['f1']:.4f}")

    # ── Per-client evaluation ─────────────────────────────────────────────────
    per_client_metrics = {}
    shap_vectors = []
    for client_id, d in client_data.items():
        y_pred = model.predict(d["X_test"])
        y_prob = model.predict_proba(d["X_test"])[:, 1]
        per_client_metrics[client_id] = {
            "accuracy": float(accuracy_score(d["y_test"], y_pred)),
            "auc":      float(roc_auc_score(d["y_test"], y_prob)),
            "f1":       float(f1_score(d["y_test"], y_pred, zero_division=0)),
        }
        # SHAP vector for client's training distribution
        shap_vec = compute_shap_vector(model, d["X_train"])
        shap_vectors.append(shap_vec)
        print(f"[centralized] {client_id}: acc={per_client_metrics[client_id]['accuracy']:.4f}")

    # ── Explanation-consistency score (centralized, split along client boundaries) ──
    consistency = explanation_consistency_score(shap_vectors)
    print(f"[centralized] Explanation-consistency score: {consistency:.4f}")

    results = {
        "variant": "centralized",
        "pooled": pooled_metrics,
        "per_client": per_client_metrics,
        "explanation_consistency_score": consistency,
    }

    out_path = results_dir / "centralized_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[centralized] Saved to {out_path}")

    return results
