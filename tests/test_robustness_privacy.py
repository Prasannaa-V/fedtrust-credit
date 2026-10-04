"""
test_robustness_privacy.py — DP + Byzantine regression tests (Tasks 1-2)
Fast unit tests: no full federated runs required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from local_training import FederatedClient
from consistency_score import per_client_agreement


def _tiny_client():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(80, 4), columns=list("abcd"))
    y = pd.Series((rng.rand(80) > 0.5).astype(int))
    return FederatedClient("t", X.iloc[:60], X.iloc[60:], y.iloc[:60], y.iloc[60:])


def test_dp_epsilon_none_is_clean():
    c = _tiny_client()
    p, _, _ = c.fit([np.array([0.5])], config={"round": 1})
    assert len(p[0]) == 60


def test_dp_noise_applied_and_finite():
    c = _tiny_client()
    p, _, _ = c.fit([np.array([0.5])], config={
        "round": 1, "dp_epsilon": 1.0, "dp_delta": 1e-5, "dp_clip": 1.0,
        "dp_seed": 7})
    assert np.all(np.isfinite(p[0]))
    # smaller epsilon => larger sigma; smoke-check both run
    c2 = _tiny_client()
    p2, _, _ = c2.fit([np.array([0.5])], config={
        "round": 1, "dp_epsilon": 0.5, "dp_delta": 1e-5, "dp_clip": 1.0,
        "dp_seed": 7})
    assert np.all(np.isfinite(p2[0]))


def test_byzantine_divergent_shap_gets_lower_agreement():
    base = np.array([0.34, 0.28, 0.22, 0.16], dtype=np.float32)
    v1 = base
    v2 = base * 0.95
    v3 = np.flip(base)  # malicious divergence
    agreements = per_client_agreement([v1, v2, v3])
    assert agreements[2] < agreements[0]
    assert agreements[2] < agreements[1]


def test_simulate_round_attack_mode():
    from fastapi.testclient import TestClient
    from service import app
    client = TestClient(app)
    honest = client.post("/api/federated/simulate-round", json={
        "consistency_gain": 1.0, "correlation_noise": 0.02,
        "client_sizes": [627716, 287686, 81060],
        "attack_mode": "none"}).json()
    attacked = client.post("/api/federated/simulate-round", json={
        "consistency_gain": 1.0, "correlation_noise": 0.02,
        "client_sizes": [627716, 287686, 81060],
        "attack_mode": "label_flip"}).json()
    assert attacked["global_consistency"] < honest["global_consistency"]
