"""
src/export_models.py — Model Artifact Exporter for Cloud Serving
FedTrust-Credit | BITE412L Cloud Computing Project

Trains (or extracts) the production LightGBM credit risk models on real Lending Club
data and exports lightweight, serialized model artifacts to models/:
  - models/global_model.txt
  - models/client_1.txt
  - models/client_2.txt
  - models/client_3.txt
  - models/model_metadata.json

Enables sub-second boot time and <300MB RAM usage in cloud environments
(AWS EC2 t2.micro / t3.small, Docker containers, Render, Fly.io) without needing
to hold the 1.6GB raw CSV in memory on every service start.
"""

import json
import time
from pathlib import Path
import lightgbm as lgb
import pandas as pd

from data_partition import (
    load_and_clean,
    partition_noniid,
    get_client_splits,
    build_shared_feature_columns,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT_DIR / "data" / "lending_club" / "loan.csv"
MODELS_DIR = ROOT_DIR / "models"

LGB_PARAMS = {
    "objective": "binary",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

def export_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[export_models] Output directory: {MODELS_DIR}")
    print(f"[export_models] Loading Lending Club dataset: {DATA_CSV}...")
    start_t0 = time.time()

    df = load_and_clean(str(DATA_CSV))
    partitions = partition_noniid(df)

    shared_feature_cols = build_shared_feature_columns(partitions)
    print(f"[export_models] Shared feature columns: {len(shared_feature_cols)} features")

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "shared_feature_cols": shared_feature_cols,
        "clients": {},
        "global_model": {},
        "training_params": LGB_PARAMS,
    }

    all_X_train = []
    all_y_train = []
    all_X_test = []
    all_y_test = []

    for cid in ["client_1", "client_2", "client_3"]:
        print(f"[export_models] Training {cid}...")
        cdf = partitions[cid]
        X_tr, X_te, y_tr, y_te = get_client_splits(cdf, feature_cols=shared_feature_cols)
        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_X_test.append(X_te)
        all_y_test.append(y_te)

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_tr, y_tr)

        acc = float(model.score(X_te, y_te))
        model_path = MODELS_DIR / f"{cid}.txt"
        model.booster_.save_model(str(model_path))

        metadata["clients"][cid] = {
            "train_samples": len(X_tr),
            "test_samples": len(X_te),
            "test_accuracy": acc,
            "model_file": f"{cid}.txt",
        }
        print(f"[export_models]   {cid} saved to {model_path} (acc={acc:.4f})")

    # Pooled Global Model
    X_pooled = pd.concat(all_X_train, ignore_index=True)
    y_pooled = pd.concat(all_y_train, ignore_index=True)
    X_test_pooled = pd.concat(all_X_test, ignore_index=True)
    y_test_pooled = pd.concat(all_y_test, ignore_index=True)

    print(f"[export_models] Training global pooled model on {len(X_pooled):,} samples...")
    global_model = lgb.LGBMClassifier(**LGB_PARAMS)
    global_model.fit(X_pooled, y_pooled)

    global_acc = float(global_model.score(X_test_pooled, y_test_pooled))
    global_path = MODELS_DIR / "global_model.txt"
    global_model.booster_.save_model(str(global_path))

    metadata["global_model"] = {
        "train_samples": len(X_pooled),
        "test_samples": len(X_test_pooled),
        "test_accuracy": global_acc,
        "model_file": "global_model.txt",
    }
    print(f"[export_models] Global model saved to {global_path} (acc={global_acc:.4f})")

    # Save metadata JSON
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_t0
    print(f"[export_models] Export complete in {total_time:.2f}s! All artifacts stored in {MODELS_DIR}")

if __name__ == "__main__":
    export_models()
