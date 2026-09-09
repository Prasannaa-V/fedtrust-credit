"""
test_service_api.py — Unit and Integration tests for FedTrust-Credit FastAPI Service
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from service import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_initialized"] is True

def test_status_endpoint():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "framework" in data
    assert data["federated_clients"] == 3

def test_metrics_summary_endpoint():
    res = client.get("/api/metrics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "pooled_metrics" in data
    assert "fedavg" in data["pooled_metrics"]
    assert "ours" in data["pooled_metrics"]

def test_metrics_partitions_endpoint():
    res = client.get("/api/metrics/partitions")
    assert res.status_code == 200
    data = res.json()
    assert "client_1" in data
    assert "client_2" in data
    assert "client_3" in data

def test_plots_endpoint():
    res = client.get("/api/plots/consistency_comparison")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"

def test_simulate_round_endpoint():
    payload = {
        "consistency_gain": 1.0,
        "correlation_noise": 0.02,
        "client_sizes": [627716, 287686, 81060],
    }
    res = client.post("/api/federated/simulate-round", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "global_consistency" in data
    assert len(data["clients"]) == 3
    assert data["communication"]["overhead_percentage"] > 0

def test_predict_risk_endpoint():
    payload = {
        "loan_amnt": 12000.0,
        "term": 36,
        "int_rate": 7.9,
        "grade": "A",
        "annual_inc": 110000.0,
        "dti": 11.5,
        "home_ownership": "MORTGAGE",
        "purpose": "debt_consolidation",
        "addr_state": "NY",
    }
    res = client.post("/api/predict/risk", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "default_probability_pct" in data
    assert "risk_tier" in data
    assert "feature_attributions" in data
    assert len(data["feature_attributions"]) > 0
    assert "client_perspectives" in data

def test_index_page_serves():
    res = client.get("/")
    assert res.status_code == 200
    assert "FedTrust-Credit" in res.text
