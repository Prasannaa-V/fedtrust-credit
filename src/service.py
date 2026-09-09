"""
service.py — FastAPI Backend Service for FedTrust-Credit
Exposes REST API endpoints for:
  - Benchmark metrics inspection (Centralized vs FedAvg vs FedTrust-Credit)
  - Non-IID partition statistics and communication overhead accounting
  - Round-by-round federated training trajectory logs
  - Static serving of publication-quality benchmark plots
  - Interactive federated round simulation (consistency-aware aggregation mathematics)
  - Real-time credit risk assessment with TreeSHAP explainability and multi-client consensus

ALL MODELS ARE TRAINED ON REAL LENDING CLUB DATA — NO SYNTHETIC DATA.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root & src are in path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
STATIC_DIR = SRC_DIR / "static"
MODELS_DIR = ROOT_DIR / "models"
DATA_CSV = ROOT_DIR / "data" / "lending_club" / "loan.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from consistency_score import _cosine_similarity, explanation_consistency_score, per_client_agreement
from aggregation_strategy import explanation_consistency_aware_aggregate
from data_partition import load_and_clean, partition_noniid, prepare_features, get_client_splits, build_shared_feature_columns

app = FastAPI(
    title="FedTrust-Credit Service",
    description="Privacy-Preserving Credit Risk Assessment with Explanation-Consistency-Aware Aggregation",
    version="1.0.0",
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Credit Risk Engine — REAL DATA ─────────────────────────────────────────────

class CreditRiskEngine:
    """
    Trained LightGBM risk models on REAL Lending Club data.
    Global model: trained on pooled data from all 3 client partitions.
    Client models: trained on each client's own non-IID partition.
    TreeSHAP explainers for real-time risk scoring and attribution.
    """
    def __init__(self):
        self.global_model: Optional[lgb.LGBMClassifier] = None
        self.client_models: Dict[str, lgb.LGBMClassifier] = {}
        self.global_explainer: Optional[shap.TreeExplainer] = None
        self.client_explainers: Dict[str, shap.TreeExplainer] = {}
        self.shared_feature_cols: List[str] = []
        self.is_ready = False
        self._background_sample: Optional[pd.DataFrame] = None
        self._init_models()

    def _init_models(self):
        """Load real Lending Club data, partition, and train models (or load pre-trained artifacts)."""
        start_time = time.time()

        # ── Fast Cloud Boot: Load pre-trained models if available (<0.5s boot, <300MB RAM)
        meta_path = MODELS_DIR / "model_metadata.json"
        global_path = MODELS_DIR / "global_model.joblib"
        if meta_path.exists() and global_path.exists():
            print(f"[CreditRiskEngine] Loading pre-trained model artifacts from {MODELS_DIR} (Cloud fast-boot)...")
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                self.shared_feature_cols = metadata["shared_feature_cols"]

                # Load client models
                for cid in ["client_1", "client_2", "client_3"]:
                    client_model_path = MODELS_DIR / f"{cid}.joblib"
                    if client_model_path.exists():
                        self.client_models[cid] = joblib.load(client_model_path)
                        self.client_explainers[cid] = shap.TreeExplainer(self.client_models[cid])

                # Load global model
                self.global_model = joblib.load(global_path)
                self.global_explainer = shap.TreeExplainer(self.global_model)

                elapsed = time.time() - start_time
                self.is_ready = True
                print(f"[CreditRiskEngine] All pre-trained models & TreeSHAP explainers loaded in {elapsed:.2f}s!")
                print(f"[CreditRiskEngine] Ready to serve predictions (Memory footprint: <300MB).")
                return
            except Exception as e:
                print(f"[CreditRiskEngine] Warning: Pre-trained model loading failed ({e}). Falling back to CSV...")

        # ── Fallback: Train from raw CSV if models not pre-serialized
        if not DATA_CSV.exists():
            print(f"[CreditRiskEngine] WARNING: Dataset not found at {DATA_CSV}")
            print("[CreditRiskEngine] Service will start without prediction capability.")
            return

        print(f"[CreditRiskEngine] Loading REAL Lending Club dataset from {DATA_CSV}...")

        # Load and clean the full dataset
        df = load_and_clean(str(DATA_CSV))
        partitions = partition_noniid(df)

        # Build shared feature space (D-015)
        print("[CreditRiskEngine] Building shared feature columns across 3 client partitions...")
        self.shared_feature_cols = build_shared_feature_columns(partitions)
        print(f"[CreditRiskEngine] Shared feature space: {len(self.shared_feature_cols)} features")

        # LightGBM training params (same as local_training.py)
        lgb_params = {
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

        # Train per-client models
        all_X_train = []
        all_y_train = []
        for cid in ["client_1", "client_2", "client_3"]:
            cdf = partitions[cid]
            X_tr, X_te, y_tr, y_te = get_client_splits(cdf, feature_cols=self.shared_feature_cols)
            all_X_train.append(X_tr)
            all_y_train.append(y_tr)

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr, y_tr)
            self.client_models[cid] = model

            self.client_explainers[cid] = shap.TreeExplainer(model)

            acc = model.score(X_te, y_te)
            print(f"[CreditRiskEngine]   {cid}: trained on {len(X_tr):,} samples | test acc={acc:.4f}")

        # Global model: trained on pooled data from all 3 partitions
        X_pooled = pd.concat(all_X_train, ignore_index=True)
        y_pooled = pd.concat(all_y_train, ignore_index=True)
        print(f"[CreditRiskEngine] Training global model on {len(X_pooled):,} pooled samples...")

        self.global_model = lgb.LGBMClassifier(**lgb_params)
        self.global_model.fit(X_pooled, y_pooled)

        # SHAP explainer for global model
        self.global_explainer = shap.TreeExplainer(self.global_model)

        elapsed = time.time() - start_time
        self.is_ready = True
        print(f"[CreditRiskEngine] All models trained on REAL data in {elapsed:.1f}s")
        print(f"[CreditRiskEngine] Ready to serve predictions.")

    def _build_feature_row(self, applicant: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert a loan applicant dict into a 1-row DataFrame aligned to the
        shared feature space. Maps user-friendly inputs to the real one-hot
        encoded feature columns used during training.
        """
        # Start with all features zeroed
        row = {col: 0.0 for col in self.shared_feature_cols}

        # Numeric features
        row["loan_amnt"] = float(applicant.get("loan_amnt", 15000.0))
        row["int_rate"] = float(applicant.get("int_rate", 12.0))
        row["annual_inc"] = float(applicant.get("annual_inc", 70000.0))
        row["dti"] = float(applicant.get("dti", 18.0))

        # Term: one-hot (column names are like term_36.0 or term_60.0)
        term_val = str(float(applicant.get("term", 36)))
        term_col = f"term_{term_val}"
        if term_col in row:
            row[term_col] = 1

        # Grade: one-hot
        grade = str(applicant.get("grade", "B")).upper()
        grade_col = f"grade_{grade}"
        if grade_col in row:
            row[grade_col] = 1

        # Home ownership: one-hot
        ho = str(applicant.get("home_ownership", "RENT")).upper()
        ho_col = f"home_ownership_{ho}"
        if ho_col in row:
            row[ho_col] = 1

        # Purpose: one-hot
        purpose = str(applicant.get("purpose", "debt_consolidation")).upper()
        # Try matching common formats
        purpose_col = f"purpose_{purpose}"
        if purpose_col in row:
            row[purpose_col] = 1
        else:
            # Try with original casing variations
            for col in self.shared_feature_cols:
                if col.startswith("purpose_") and purpose.lower() in col.lower():
                    row[col] = 1
                    break

        # Address state: one-hot
        state = str(applicant.get("addr_state", "NY")).upper()
        state_col = f"addr_state_{state}"
        if state_col in row:
            row[state_col] = 1

        df_row = pd.DataFrame([row])[self.shared_feature_cols]
        return df_row

    def predict_and_explain(self, applicant: Dict[str, Any]) -> Dict[str, Any]:
        """Score a loan applicant and produce TreeSHAP explanations with multi-client consensus."""
        if not self.is_ready:
            return {
                "error": "Models not loaded — Lending Club dataset not found.",
                "default_probability_pct": 0.0,
                "risk_tier": "Unknown",
                "decision": "Service not ready",
                "explanation_consistency": 0.0,
                "feature_attributions": [],
                "client_perspectives": {},
            }

        df_row = self._build_feature_row(applicant)

        # Global inference
        global_prob = float(self.global_model.predict_proba(df_row)[0, 1])
        shap_vals = self.global_explainer.shap_values(df_row.values)

        # Handle SHAP output shapes (TreeExplainer may return list for binary)
        if isinstance(shap_vals, list) and len(shap_vals) >= 2:
            s_vec = np.array(shap_vals[1]).flatten()
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            s_vec = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            s_vec = shap_vals[0]
        else:
            s_vec = np.array(shap_vals).flatten()

        # Build feature attribution list — select top impactful features
        features_explained = []
        for idx, (feat, sv) in enumerate(zip(self.shared_feature_cols, s_vec)):
            feat_val = float(df_row.iloc[0, idx])
            features_explained.append({
                "feature": feat,
                "value": feat_val,
                "shap_value": float(sv),
                "impact": "Increases Risk" if sv > 0 else "Decreases Risk",
                "abs_importance": abs(float(sv)),
            })
        features_explained.sort(key=lambda x: x["abs_importance"], reverse=True)

        # Client-level predictions & local SHAP
        client_evals = {}
        client_shap_vectors = []
        for c_id in ["client_1", "client_2", "client_3"]:
            c_model = self.client_models[c_id]
            c_explainer = self.client_explainers[c_id]
            c_prob = float(c_model.predict_proba(df_row)[0, 1])
            c_sv = c_explainer.shap_values(df_row.values)
            if isinstance(c_sv, list) and len(c_sv) >= 2:
                c_vec = np.array(c_sv[1]).flatten()
            elif isinstance(c_sv, np.ndarray) and c_sv.ndim == 3:
                c_vec = c_sv[0, :, 1]
            elif isinstance(c_sv, np.ndarray) and c_sv.ndim == 2:
                c_vec = c_sv[0]
            else:
                c_vec = np.array(c_sv).flatten()

            client_shap_vectors.append(c_vec)
            top_idx = int(np.argmax(np.abs(c_vec)))
            client_evals[c_id] = {
                "default_probability": round(c_prob * 100, 2),
                "top_factor": self.shared_feature_cols[top_idx],
                "top_factor_impact": "Increases Risk" if c_vec[top_idx] > 0 else "Decreases Risk",
            }

        # Multi-institution explanation consistency for this applicant
        client_consistency = float(explanation_consistency_score(client_shap_vectors))

        if global_prob < 0.15:
            risk_tier = "Low (Prime)"
            decision = "Approved (Low Default Risk)"
        elif global_prob < 0.30:
            risk_tier = "Moderate (Near-Prime)"
            decision = "Conditional Approval / Manual Review"
        else:
            risk_tier = "High (Subprime)"
            decision = "Decline Recommended (High Default Risk)"

        return {
            "default_probability_pct": round(global_prob * 100, 2),
            "risk_tier": risk_tier,
            "decision": decision,
            "explanation_consistency": round(client_consistency, 4),
            "feature_attributions": features_explained[:10],
            "client_perspectives": client_evals,
        }

print("[service] Initializing CreditRiskEngine with REAL Lending Club data...")
engine = CreditRiskEngine()


# ── REST API Routes ────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "FedTrust-Credit Web Dashboard & REST API",
        "version": "1.0.0",
        "model_initialized": engine.is_ready,
        "data_source": "Real Lending Club (1.34M records)" if engine.is_ready else "Not loaded",
    }

@app.get("/api/status")
def get_status():
    return {
        "framework": "FedTrust-Credit",
        "objective": "Explanation-Consistency-Aware Federated Learning for Credit Risk",
        "dataset": "Lending Club (accepted 2007-2018)",
        "total_records_evaluated": 1344976,
        "federated_clients": 3,
        "communication_rounds": 20,
        "consistency_gain": 1.0,
        "bandwidth_overhead": "+0.0316%",
        "model_ready": engine.is_ready,
        "feature_space_size": len(engine.shared_feature_cols),
        "status": "online",
    }

@app.get("/api/metrics/summary")
def get_metrics_summary():
    json_path = RESULTS_DIR / "full_metrics_summary.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Metrics summary file not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.get("/api/metrics/partitions")
def get_partitions():
    json_path = RESULTS_DIR / "partition_stats.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Partition stats file not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.get("/api/metrics/rounds")
def get_rounds_history():
    ours_file = RESULTS_DIR / "consistency_aware" / "all_rounds.jsonl"
    rounds_data = []
    if ours_file.exists():
        with open(ours_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rounds_data.append(json.loads(line.strip()))
    return {
        "total_rounds": len(rounds_data),
        "history": rounds_data,
    }

@app.get("/api/plots/{plot_name}")
def get_plot(plot_name: str):
    clean_name = plot_name.replace(".png", "")
    plot_file = PLOTS_DIR / f"{clean_name}.png"
    if not plot_file.exists():
        raise HTTPException(status_code=404, detail=f"Plot '{clean_name}.png' not found.")
    return FileResponse(plot_file, media_type="image/png")


# ── Simulation & Prediction Request Models ────────────────────────────────────

class SimulateRoundRequest(BaseModel):
    consistency_gain: float = Field(default=1.0, ge=0.0, le=10.0, description="Gamma gain parameter")
    client_sizes: List[int] = Field(
        default=[502172, 230148, 64848],
        description="Dataset rows for Client 1, Client 2, Client 3",
    )
    correlation_noise: float = Field(default=0.02, ge=0.0, le=0.5, description="Attribution noise factor")

class LoanApplicantRequest(BaseModel):
    loan_amnt: float = Field(default=15000.0, ge=1000.0, le=50000.0)
    term: int = Field(default=36, description="36 or 60 months")
    int_rate: float = Field(default=11.5, ge=4.0, le=35.0)
    grade: str = Field(default="B", description="A, B, C, D, E, F, or G")
    annual_inc: float = Field(default=75000.0, ge=10000.0, le=1000000.0)
    dti: float = Field(default=18.5, ge=0.0, le=60.0)
    home_ownership: str = Field(default="RENT", description="RENT, OWN, MORTGAGE")
    purpose: str = Field(default="debt_consolidation", description="debt_consolidation, credit_card, etc.")
    addr_state: str = Field(default="NY")


@app.post("/api/federated/simulate-round")
def simulate_federated_round(req: SimulateRoundRequest):
    """
    Simulates a federated aggregation round using the exact formulas from Section 3.3.
    Demonstrates the difference between FedAvg baseline (gain=0) and FedTrust-Credit (gain=gamma).
    """
    sizes = req.client_sizes
    total_size = sum(sizes)
    base_weights = [s / total_size for s in sizes]

    base_shap = np.array([0.34, 0.28, 0.22, 0.16, 0.12, 0.25, 0.08, 0.06], dtype=np.float32)
    
    rng = np.random.RandomState(int(req.consistency_gain * 100) + 42)
    noise1 = rng.normal(0, req.correlation_noise, size=len(base_shap))
    noise2 = rng.normal(0, req.correlation_noise * 1.2, size=len(base_shap))
    noise3 = rng.normal(0, req.correlation_noise * 1.5, size=len(base_shap))

    v1 = np.maximum(0, base_shap + noise1)
    v2 = np.maximum(0, base_shap * 0.95 + noise2)
    v3 = np.maximum(0, base_shap * 0.92 + noise3)
    shap_vectors = [v1, v2, v3]

    sim_matrix = [
        [_cosine_similarity(v1, v1), _cosine_similarity(v1, v2), _cosine_similarity(v1, v3)],
        [_cosine_similarity(v2, v1), _cosine_similarity(v2, v2), _cosine_similarity(v2, v3)],
        [_cosine_similarity(v3, v1), _cosine_similarity(v3, v2), _cosine_similarity(v3, v3)],
    ]

    global_cons = explanation_consistency_score(shap_vectors)
    agreements = per_client_agreement(shap_vectors)

    adjusted_raw = [
        base_weights[i] * (1.0 + req.consistency_gain * agreements[i])
        for i in range(len(sizes))
    ]
    weight_sum = sum(adjusted_raw)
    adjusted_weights = [w / weight_sum for w in adjusted_raw]

    weight_bytes = 3188672
    shap_bytes = 1008 if req.consistency_gain > 0.0 else 0
    total_bytes = weight_bytes + shap_bytes
    overhead_pct = (shap_bytes / weight_bytes) * 100.0

    return {
        "consistency_gain": req.consistency_gain,
        "global_consistency": round(global_cons, 5),
        "pairwise_similarity_matrix": [[round(x, 4) for x in row] for row in sim_matrix],
        "clients": [
            {
                "client_id": f"client_{i+1}",
                "name": ["Bank 1 (Prime/Low-Risk)", "Bank 2 (East Coast)", "Bank 3 (Subprime)"][i],
                "size": sizes[i],
                "base_weight_pct": round(base_weights[i] * 100, 3),
                "agreement_score": round(agreements[i], 4),
                "adjusted_weight_pct": round(adjusted_weights[i] * 100, 3),
                "weight_delta_pct": round((adjusted_weights[i] - base_weights[i]) * 100, 4),
            }
            for i in range(len(sizes))
        ],
        "communication": {
            "weight_bytes": weight_bytes,
            "shap_bytes": shap_bytes,
            "total_bytes": total_bytes,
            "overhead_percentage": round(overhead_pct, 4),
        },
    }

@app.post("/api/predict/risk")
def predict_credit_risk(applicant: LoanApplicantRequest):
    """
    Scores loan applicant default risk using REAL Lending Club-trained models
    and produces TreeSHAP explanations alongside multi-institution consensus evaluation.
    """
    res = engine.predict_and_explain(applicant.model_dump())
    return res


# ── Mount Static Frontend Assets ──────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "FedTrust-Credit API is running. Static UI not built yet."})
    return FileResponse(index_file)
