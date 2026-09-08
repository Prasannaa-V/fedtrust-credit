"""
tests/test_consistency_score.py — Sanity tests for explanation_consistency_score()
FedTrust-Credit | Day 6

Three sanity tests MUST pass before explanation_consistency_score is wired into aggregation.

Test 1: Identical SHAP vectors -> score ~= 1.0 (within 1e-6)
Test 2: Orthogonal SHAP vectors (one-hot basis) -> score ~= 0.0
Test 3: Partial agreement (2 similar, 1 different) -> score in (0.3, 0.7)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from consistency_score import explanation_consistency_score, per_client_agreement


# ── Test 1: Identical vectors ──────────────────────────────────────────────────

def test_identical_vectors_score_one():
    """
    Three identical SHAP vectors must produce a consistency score of exactly 1.0.
    Tolerance: within 1e-6 (floating point only).
    """
    v = np.array([0.5, 0.2, 0.1, 0.05, 0.15], dtype=np.float32)
    shap_vectors = [v.copy(), v.copy(), v.copy()]
    score = explanation_consistency_score(shap_vectors)
    assert abs(score - 1.0) < 1e-6, (
        f"Identical SHAP vectors should score 1.0, got {score:.8f}"
    )


# ── Test 2: Orthogonal vectors ─────────────────────────────────────────────────

def test_orthogonal_vectors_score_zero():
    """
    Three mutually orthogonal one-hot SHAP vectors must produce a score of ~0.0.
    (Cosine similarity of orthogonal vectors = 0.)
    """
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    shap_vectors = [v1, v2, v3]
    score = explanation_consistency_score(shap_vectors)
    assert abs(score - 0.0) < 1e-6, (
        f"Orthogonal SHAP vectors should score 0.0, got {score:.8f}"
    )


# ── Test 3: Partial agreement ──────────────────────────────────────────────────

def test_partial_agreement_score_in_range():
    """
    Two similar vectors and one deliberately different vector should produce
    a score strictly between 0.3 and 0.7.
    """
    v1 = np.array([0.5, 0.3, 0.2], dtype=np.float32)        # client 1
    v2 = np.array([0.48, 0.31, 0.21], dtype=np.float32)     # client 2 (very similar to 1)
    v3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)        # client 3 (orthogonal to 1&2)
    shap_vectors = [v1, v2, v3]
    score = explanation_consistency_score(shap_vectors)
    assert 0.3 < score < 0.7, (
        f"Partial-agreement SHAP vectors should score between 0.3 and 0.7, got {score:.4f}"
    )


# ── Additional: per_client_agreement sanity check ─────────────────────────────

def test_per_client_agreement_identical():
    """Per-client agreement should be 1.0 for all clients when vectors are identical."""
    v = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    shap_vectors = [v.copy(), v.copy(), v.copy()]
    agreements = per_client_agreement(shap_vectors)
    for i, a in enumerate(agreements):
        assert abs(a - 1.0) < 1e-6, f"Client {i} agreement should be 1.0, got {a:.8f}"


def test_consistency_gain_zero_recovers_fedavg():
    """
    When CONSISTENCY_GAIN=0, adjusted_weight[i] = base_weight[i] regardless of SHAP.
    Verifies Section 3.3's claim: 'setting it to zero recovers standard FedAvg exactly'.
    Test: with CONSISTENCY_GAIN=0, the aggregated scalar must equal the plain
    data-size-weighted mean of client prediction scalars.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from aggregation_strategy import explanation_consistency_aware_aggregate

    # Dummy prediction scalars (one per client — mean of their probability vectors)
    # Represented as 1-element arrays matching our aggregation contract
    w1 = np.array([0.3], dtype=np.float32)   # client 1 mean pred
    w2 = np.array([0.5], dtype=np.float32)   # client 2 mean pred
    w3 = np.array([0.7], dtype=np.float32)   # client 3 mean pred
    sizes = [100, 200, 300]  # total = 600

    # Expected FedAvg: weighted mean scalar
    total = sum(sizes)
    expected_scalar = sum(
        (s / total) * float(w[0])
        for s, w in zip(sizes, [w1, w2, w3])
    )

    aggregated, _ = explanation_consistency_aware_aggregate(
        client_weights=[w1, w2, w3],
        client_sizes=sizes,
        shap_vectors=[np.array([0.5, 0.3, 0.2]), np.array([0.1, 0.8, 0.1]), np.array([0.3, 0.4, 0.3])],
        consistency_gain=0.0,
    )

    np.testing.assert_allclose(
        float(aggregated[0]), expected_scalar, rtol=1e-5,
        err_msg="CONSISTENCY_GAIN=0 must recover FedAvg exactly"
    )


if __name__ == "__main__":
    # Run tests directly (no pytest required)
    tests = [
        test_identical_vectors_score_one,
        test_orthogonal_vectors_score_zero,
        test_partial_agreement_score_in_range,
        test_per_client_agreement_identical,
        test_consistency_gain_zero_recovers_fedavg,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    if failed:
        sys.exit(1)
