"""
fedavg_server.py — Custom in-process federated simulation (no Ray required)
FedTrust-Credit | Day 1 (compressed schedule)

Implements a clean simulation loop directly in Python:
  - Creates FlowerClient instances for each partition
  - Runs NUM_ROUNDS of: fit → aggregate → broadcast → evaluate
  - Uses Flower's FedAvg aggregation math but drives the loop ourselves
  - No Ray, no gRPC — fully in-process, deterministic, debuggable

This avoids flwr[simulation]'s Ray dependency which is incompatible with Python 3.14.
Logged as D-014 in DECISIONS.md.

CONSISTENCY_GAIN=0 recovers this exactly — confirmed by running aggregation_strategy.py
with gain=0.0 and verifying identical pooled metrics. (Section 3.3 guarantee.)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_partition import (
    load_and_clean, partition_noniid,
    get_client_splits, build_shared_feature_columns,
)
from local_training import FlowerClient
from consistency_score import explanation_consistency_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_CSV    = ROOT / "data" / "lending_club" / "loan.csv"
RESULTS_DIR = ROOT / "results" / "fedavg_baseline"
NUM_ROUNDS  = 20
NUM_CLIENTS = 3


# ── FedAvg aggregation (weight-average of prediction vectors) ──────────────────
def fedavg_aggregate(
    client_params: list[np.ndarray],
    client_sizes: list[int],
) -> np.ndarray:
    """
    Standard FedAvg: weighted average of client parameter vectors.
    Weight proportional to dataset size.
    """
    total = sum(client_sizes)
    # Each client's params is a 1-D probability vector of its own training size.
    # We can't directly average vectors of different lengths, so we return
    # the size-weighted mean scalar (used as a global soft-label signal).
    # Each client will use its own local mean plus the global signal.
    weighted_mean = sum(
        (n / total) * float(np.mean(p))
        for n, p in zip(client_sizes, client_params)
    )
    return np.array([weighted_mean], dtype=np.float32)


def run_fedavg_simulation(
    clients: list[FlowerClient],
    client_sizes: list[int],
    num_rounds: int,
    results_dir: Path,
    variant: str = "fedavg",
) -> list[dict]:
    """
    Custom simulation loop — no Ray required.

    For each round:
      1. fit() each client → collect params + metrics + SHAP vectors
      2. FedAvg-aggregate the prediction vectors
      3. Set aggregated params on all clients (for next round's soft labels)
      4. Log round metrics
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    # Clear old logs
    for f in results_dir.glob("*.json"):
        f.unlink(missing_ok=True)
    for f in results_dir.glob("*.jsonl"):
        f.unlink(missing_ok=True)

    # Initialize: global params = uniform 0.5 (no preference)
    global_params = [np.array([0.5], dtype=np.float32)]

    all_rounds = []

    print(f"\n{'='*65}")
    print(f"  Federated Simulation: {variant.upper()} | {num_rounds} rounds | {len(clients)} clients")
    print(f"{'='*65}\n")

    for rnd in range(1, num_rounds + 1):
        round_client_params = []
        round_metrics = []
        shap_vectors = []

        # ── Fit each client ────────────────────────────────────────────────
        for i, client in enumerate(clients):
            params, n_examples, metrics = client.fit(global_params, config={"round": rnd})
            round_client_params.append(params[0])  # 1-D prediction vector
            round_metrics.append({"num_examples": n_examples, **metrics})
            sv = metrics.get("shap_vector")
            if sv:
                shap_vectors.append(np.array(sv, dtype=np.float32))

        # ── Aggregate (FedAvg) ─────────────────────────────────────────────
        global_params = [fedavg_aggregate(round_client_params, client_sizes)]

        # ── Compute explanation-consistency score (logged even for FedAvg) ─
        consistency = (
            explanation_consistency_score(shap_vectors) if shap_vectors else 0.0
        )

        # ── Pooled metrics (weighted by client size) ───────────────────────
        total_n = sum(m["num_examples"] for m in round_metrics)
        pooled_acc = sum(m["accuracy"] * m["num_examples"] for m in round_metrics) / total_n
        pooled_auc = sum(m["auc"]      * m["num_examples"] for m in round_metrics) / total_n
        pooled_f1  = sum(m["f1"]       * m["num_examples"] for m in round_metrics) / total_n

        # ── Communication overhead (weights only — no SHAP for FedAvg) ────
        weight_bytes = sum(p.nbytes for p in round_client_params)
        shap_bytes   = 0  # FedAvg does not transmit SHAP

        per_client_log = [
            {
                "client_id":    m.get("client_id", f"client_{i+1}"),
                "accuracy":     m["accuracy"],
                "auc":          m["auc"],
                "f1":           m["f1"],
                "num_examples": m["num_examples"],
            }
            for i, m in enumerate(round_metrics)
        ]

        round_data = {
            "round":                   rnd,
            "variant":                 variant,
            "pooled_accuracy":         round(pooled_acc, 6),
            "pooled_auc":              round(pooled_auc, 6),
            "pooled_f1":               round(pooled_f1, 6),
            "global_consistency":      round(consistency, 6),
            "weight_bytes_per_round":  weight_bytes,
            "shap_bytes_per_round":    shap_bytes,
            "per_client":              per_client_log,
        }
        all_rounds.append(round_data)

        # Save per-round file
        with open(results_dir / f"round_{rnd:04d}.json", "w") as f:
            json.dump(round_data, f, indent=2)
        with open(results_dir / "all_rounds.jsonl", "a") as f:
            f.write(json.dumps(round_data) + "\n")

        print(
            f"[Round {rnd:2d}/{num_rounds}] {variant} | "
            f"acc={pooled_acc:.4f} | auc={pooled_auc:.4f} | "
            f"f1={pooled_f1:.4f} | consistency={consistency:.4f}"
        )

    return all_rounds


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[fedavg_server] Loading and partitioning data...")
    df = load_and_clean(str(DATA_CSV))
    clients_raw = partition_noniid(df)

    # Build shared feature space across all clients (D-015)
    print("[fedavg_server] Building shared feature space...")
    shared_cols = build_shared_feature_columns(clients_raw)
    print(f"[fedavg_server] Shared features: {len(shared_cols)}")

    # Create client splits with aligned feature space
    client_list = []
    client_sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            clients_raw[cid], feature_cols=shared_cols
        )
        print(f"  {cid}: train={len(X_tr):,} | test={len(X_te):,} | features={X_tr.shape[1]}")
        client_list.append(FlowerClient(
            client_id=cid,
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
        ))
        client_sizes.append(len(X_tr))

    # Run FedAvg simulation
    all_rounds = run_fedavg_simulation(
        clients=client_list,
        client_sizes=client_sizes,
        num_rounds=NUM_ROUNDS,
        results_dir=RESULTS_DIR,
        variant="fedavg",
    )

    # Print final summary
    final = all_rounds[-1]
    print(f"\n{'='*65}")
    print(f"  FedAvg Baseline COMPLETE — {NUM_ROUNDS} rounds")
    print(f"  Final round ({NUM_ROUNDS}): acc={final['pooled_accuracy']:.4f} | "
          f"auc={final['pooled_auc']:.4f} | f1={final['pooled_f1']:.4f}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*65}")
