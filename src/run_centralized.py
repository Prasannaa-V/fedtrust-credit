"""
run_centralized.py — Centralized training baseline (no federation)
FedTrust-Credit | Day 2 (compressed schedule)

Trains a single LightGBM on all three partitions pooled.
Evaluates on the same held-out test splits used by the federated runs.
Computes explanation-consistency score by partitioning SHAP along client boundaries (Sec 4.3).

Uses n_estimators=200 (D-012) as the privacy-free upper-bound reference.
"""

import json
import sys
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_partition import (
    load_and_clean, partition_noniid,
    get_client_splits, build_shared_feature_columns,
)
from consistency_score import explanation_consistency_score
from shap_explanation import compute_shap_vector

DATA_CSV    = ROOT / "data" / "lending_club" / "loan.csv"
RESULTS_DIR = ROOT / "results" / "centralized_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("[centralized] Loading and partitioning data...")
    df = load_and_clean(str(DATA_CSV))
    clients_raw = partition_noniid(df)

    # Shared feature space (same as federated runs for fair comparison)
    shared_cols = build_shared_feature_columns(clients_raw)
    print(f"[centralized] Shared features: {len(shared_cols)}")

    # Build per-client splits with aligned features
    splits = {}
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            clients_raw[cid], feature_cols=shared_cols
        )
        splits[cid] = {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}
        print(f"  {cid}: train={len(X_tr):,} | test={len(X_te):,}")

    # Pool all training data
    X_train_all = pd.concat([s["X_train"] for s in splits.values()], axis=0)
    y_train_all = pd.concat([s["y_train"] for s in splits.values()], axis=0)
    print(f"\n[centralized] Pooled train: {len(X_train_all):,} rows")

    # Train centralized model (n_estimators=200 per D-012)
    print("[centralized] Training centralized LightGBM (n_estimators=200)...")
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
    )
    model.fit(X_train_all, y_train_all)
    print("[centralized] Training complete.")

    # Pooled test evaluation
    X_test_all = pd.concat([s["X_test"] for s in splits.values()], axis=0)
    y_test_all = pd.concat([s["y_test"] for s in splits.values()], axis=0)
    y_pred_all = model.predict(X_test_all)
    y_prob_all = model.predict_proba(X_test_all)[:, 1]
    pooled = {
        "accuracy": float(accuracy_score(y_test_all, y_pred_all)),
        "auc":      float(roc_auc_score(y_test_all, y_prob_all)),
        "f1":       float(f1_score(y_test_all, y_pred_all, zero_division=0)),
    }
    print(f"[centralized] Pooled: acc={pooled['accuracy']:.4f} | auc={pooled['auc']:.4f} | f1={pooled['f1']:.4f}")

    # Per-client evaluation + SHAP vectors
    per_client = {}
    shap_vectors = []
    for cid, s in splits.items():
        y_pred = model.predict(s["X_test"])
        y_prob = model.predict_proba(s["X_test"])[:, 1]
        acc = float(accuracy_score(s["y_test"], y_pred))
        auc = float(roc_auc_score(s["y_test"], y_prob))
        f1  = float(f1_score(s["y_test"], y_pred, zero_division=0))
        per_client[cid] = {"accuracy": acc, "auc": auc, "f1": f1,
                           "num_test": len(s["X_test"])}
        # SHAP vector using client's training data (same privacy-boundary split as federated)
        shap_vec = compute_shap_vector(model, s["X_train"])
        shap_vectors.append(shap_vec)
        print(f"  [{cid}] acc={acc:.4f} | auc={auc:.4f} | f1={f1:.4f}")

    # Explanation-consistency score (Section 4.3: computed by splitting along client lines)
    consistency = explanation_consistency_score(shap_vectors)
    print(f"[centralized] Explanation-consistency score: {consistency:.4f}")

    # Communication overhead: N/A (no federation), but log model size for reference
    import pickle
    model_bytes = len(pickle.dumps(model))
    print(f"[centralized] Model size: {model_bytes:,} bytes")

    # Fairness spread
    accs = [v["accuracy"] for v in per_client.values()]
    f1s  = [v["f1"]       for v in per_client.values()]
    fairness = {
        "accuracy_spread": round(max(accs) - min(accs), 6),
        "f1_spread":       round(max(f1s)  - min(f1s),  6),
    }
    print(f"[centralized] Fairness spread: acc={fairness['accuracy_spread']:.4f} | f1={fairness['f1_spread']:.4f}")

    results = {
        "variant":                    "centralized",
        "pooled":                     pooled,
        "per_client":                 per_client,
        "explanation_consistency":    round(consistency, 6),
        "model_bytes":                model_bytes,
        "fairness_spread":            fairness,
        "n_estimators":               200,
        "shared_feature_count":       len(shared_cols),
    }

    out_path = RESULTS_DIR / "centralized_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[centralized] Results saved to {out_path}")

    print(f"\n{'='*65}")
    print(f"  Centralized Baseline COMPLETE")
    print(f"  acc={pooled['accuracy']:.4f} | auc={pooled['auc']:.4f} | f1={pooled['f1']:.4f}")
    print(f"  consistency={consistency:.4f} | acc_spread={fairness['accuracy_spread']:.4f}")
    print(f"{'='*65}")
