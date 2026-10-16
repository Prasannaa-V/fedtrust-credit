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


def test_dp_reports_sensitivity_diagnostics():
    import math
    c = _tiny_client()
    _, _, m = c.fit([np.array([0.5])], config={
        "round": 1, "dp_epsilon": 1.0, "dp_delta": 1e-5, "dp_clip": 1.0,
        "dp_seed": 7})
    assert np.isfinite(m["update_l2_norm"]) and m["update_l2_norm"] >= 0
    assert np.isfinite(m["dp_pre_noise_norm"])
    expected_sigma = 1.0 * math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
    assert abs(m["dp_sigma"] - expected_sigma) < 1e-6
    assert np.isfinite(m["dp_noise_l2_norm"])


def test_dp_shap_noise_opt_in_only():
    c = _tiny_client()
    _, _, m_clean = c.fit([np.array([0.5])], config={"round": 1})
    assert "dp_shap_sigma" not in m_clean  # default path untouched
    c2 = _tiny_client()
    _, _, m_noisy = c2.fit([np.array([0.5])], config={
        "round": 1, "dp_shap_epsilon": 1.0, "dp_delta": 1e-5,
        "dp_shap_clip": 1.0, "dp_seed": 7})
    assert np.isfinite(m_noisy["dp_shap_sigma"])
    assert np.isfinite(np.array(m_noisy["shap_vector"])).all()


def test_latency_telemetry_noops_offline():
    from aws_integrations import AWSCloudManager
    mgr = AWSCloudManager()
    # Must not raise with or without a CloudWatch client configured.
    mgr.log_latency_to_cloudwatch("simulate-round", 0.012)


def test_dashboard_latency_widget_is_histogram():
    import json
    sys.path.insert(0, str(ROOT / "src"))
    from deploy_cloudwatch_dashboard import build_body
    body = json.loads(build_body("us-east-1"))
    latency = [w for w in body["widgets"]
               if w.get("properties", {}).get("title", "").startswith("API Latency")]
    assert len(latency) == 1
    assert latency[0]["properties"].get("view") == "bar"
    assert latency[0]["properties"].get("stat") == "p95"


def test_rdp_gaussian_closed_form():
    import math
    sys.path.insert(0, str(ROOT / "src"))
    from dp_accountant import rdp_gaussian, analytic_sigma
    # Mironov Prop. 7: RDP(2) with sigma=4.845, C=1 is 2/(2*sigma^2).
    assert abs(rdp_gaussian(2.0, 4.8450367) - 2.0 / (2 * 4.8450367 ** 2)) < 1e-9
    # Analytic sigma must match the FederatedClient calibration.
    assert abs(analytic_sigma(1.0, 1.0, 1e-5)
               - math.sqrt(2.0 * math.log(1.25 / 1e-5))) < 1e-9


def test_composed_budget_monotonic():
    sys.path.insert(0, str(ROOT / "src"))
    from dp_accountant import total_epsilon
    small = total_epsilon(10.0, 1e-5, 9)["composed_epsilon"]
    big = total_epsilon(0.5, 1e-5, 9)["composed_epsilon"]
    more = total_epsilon(1.0, 1e-5, 18)["composed_epsilon"]
    base = total_epsilon(1.0, 1e-5, 9)["composed_epsilon"]
    assert big < base < small  # less noise per release -> larger total
    assert more > base  # more releases -> larger total


def test_update_norm_filter_flags_gross_poison():
    sys.path.insert(0, str(ROOT / "src"))
    from byzantine_defense import flag_updates
    rng = np.random.RandomState(0)
    honest = [rng.uniform(0, 1, size=500).astype(np.float32) for _ in range(2)]
    poisoned = (honest[0] + rng.normal(0, 2.0, size=500)).astype(np.float32)
    flags = flag_updates([honest[0], honest[1], poisoned])
    assert flags == [False, False, True]


def test_update_norm_filter_clean_passes():
    sys.path.insert(0, str(ROOT / "src"))
    from byzantine_defense import flag_updates
    rng = np.random.RandomState(1)
    base = rng.uniform(0, 1, size=500).astype(np.float32)
    vecs = [np.maximum(0, base + rng.normal(0, 0.02, size=500))
            for _ in range(3)]
    assert flag_updates(vecs) == [False, False, False]
