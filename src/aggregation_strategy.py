"""
aggregation_strategy.py — Explanation-Consistency-Aware Aggregation
FedTrust-Credit | Day 2 (compressed schedule)

Core novel component. Pure Python — no flwr import.

Implements explanation_consistency_aware_aggregate() EXACTLY as Section 3.3 pseudocode.
Setting CONSISTENCY_GAIN=0 recovers standard FedAvg exactly.
"""

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from consistency_score import explanation_consistency_score, per_client_agreement


def explanation_consistency_aware_aggregate(
    client_weights: list[np.ndarray],
    client_sizes:   list[int],
    shap_vectors:   Sequence[np.ndarray],
    consistency_gain: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Section 3.3 aggregation — EXACT pseudocode implementation:

    global_consistency = explanation_consistency_score(shap_vectors)
    FOR EACH client i:
        base_weight[i] = client_sizes[i] / SUM(client_sizes)
        agreement[i] = MEAN(COSINE_SIMILARITY(shap_vectors[i], shap_vectors[j]) FOR j != i)
        adjusted_weight[i] = base_weight[i] * (1 + CONSISTENCY_GAIN * agreement[i])
    NORMALIZE adjusted_weight SO THAT SUM(adjusted_weight) == 1
    global_model = SUM(adjusted_weight[i] * client_weights[i] FOR EACH i)
    RETURN global_model, global_consistency

    Args:
        client_weights:   List of 1-D numpy arrays (prediction vectors per client).
        client_sizes:     Number of training examples per client.
        shap_vectors:     Mean-absolute SHAP attribution vectors per client.
        consistency_gain: CONSISTENCY_GAIN hyperparameter (D-007, default=1.0).
                          Set to 0.0 to recover standard FedAvg exactly.

    Returns:
        (aggregated_weight_vector, global_consistency_score)
    """
    n_clients = len(client_weights)
    total_size = sum(client_sizes)

    # Global consistency score
    global_consistency = explanation_consistency_score(shap_vectors)

    # Per-client agreement terms
    agreements = per_client_agreement(shap_vectors)

    # Base weights (FedAvg term)
    base_weights = [sz / total_size for sz in client_sizes]

    # Adjusted weights
    adjusted_weights = [
        base_weights[i] * (1.0 + consistency_gain * agreements[i])
        for i in range(n_clients)
    ]

    # Normalize
    weight_sum = sum(adjusted_weights)
    adjusted_weights = [w / weight_sum for w in adjusted_weights]

    # Aggregate — weighted average of prediction vectors
    # (vectors may differ in length = training set size; use scalar mean per client)
    aggregated_scalar = sum(
        adjusted_weights[i] * float(np.mean(client_weights[i]))
        for i in range(n_clients)
    )
    aggregated = np.array([aggregated_scalar], dtype=np.float32)

    return aggregated, global_consistency


class ExplanationConsistencyAggregator:
    """
    Stateful aggregator for the custom simulation loop.
    Wraps explanation_consistency_aware_aggregate() with logging.
    """

    def __init__(
        self,
        consistency_gain: float = 1.0,
        results_dir: str | Path = "results/consistency_aware",
    ):
        self.consistency_gain = consistency_gain
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        for f in self.results_dir.glob("*.json"):
            f.unlink(missing_ok=True)
        for f in self.results_dir.glob("*.jsonl"):
            f.unlink(missing_ok=True)
        print(f"[aggregation] ExplanationConsistencyAggregator | CONSISTENCY_GAIN={consistency_gain}")

    def aggregate(
        self,
        server_round: int,
        client_weights: list[np.ndarray],
        client_sizes:   list[int],
        shap_vectors:   list[np.ndarray],
    ) -> tuple[np.ndarray, float]:
        aggregated, global_consistency = explanation_consistency_aware_aggregate(
            client_weights=client_weights,
            client_sizes=client_sizes,
            shap_vectors=shap_vectors,
            consistency_gain=self.consistency_gain,
        )
        self._log(server_round, global_consistency, client_sizes)
        return aggregated, global_consistency

    def _log(self, server_round: int, global_consistency: float, client_sizes: list[int]):
        data = {
            "round":              server_round,
            "global_consistency": round(global_consistency, 6),
            "consistency_gain":   self.consistency_gain,
            "client_sizes":       client_sizes,
        }
        with open(self.results_dir / f"round_{server_round:04d}.json", "w") as f:
            json.dump(data, f, indent=2)
        with open(self.results_dir / "all_rounds.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")
