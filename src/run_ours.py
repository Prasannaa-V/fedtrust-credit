"""
run_ours.py — Explanation-Consistency-Aware FedAvg simulation
FedTrust-Credit | Day 2 (compressed schedule)

Identical simulation loop as fedavg_server.py but uses ExplanationConsistencyAggregator
instead of plain FedAvg. CONSISTENCY_GAIN=1.0 (D-007).

Setting CONSISTENCY_GAIN=0.0 here recovers fedavg_server.py results exactly (Section 3.3).
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
from local_training import FederatedClient
from aggregation_strategy import ExplanationConsistencyAggregator
from consistency_score import explanation_consistency_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_CSV         = ROOT / "data" / "lending_club" / "loan.csv"
RESULTS_DIR      = ROOT / "results" / "consistency_aware"
NUM_ROUNDS       = 20
NUM_CLIENTS      = 3
CONSISTENCY_GAIN = 1.0    # D-007


def run_consistency_aware_simulation(
    clients: list[FederatedClient],
    client_sizes: list[int],
    num_rounds: int,
    results_dir: Path,
    consistency_gain: float,
) -> list[dict]:
    """Custom simulation loop with explanation-consistency-aware aggregation."""
    aggregator = ExplanationConsistencyAggregator(
        consistency_gain=consistency_gain,
        results_dir=results_dir,
    )

    global_params = [np.array([0.5], dtype=np.float32)]
    all_rounds = []

    print(f"\n{'='*65}")
    print(f"  Federated Simulation: OURS (consistency-aware)")
    print(f"  CONSISTENCY_GAIN={consistency_gain} | rounds={num_rounds} | clients={len(clients)}")
    print(f"  Results -> {results_dir}")
    print(f"{'='*65}\n")

    for rnd in range(1, num_rounds + 1):
        round_client_params = []
        round_metrics = []
        shap_vectors = []

        # ── Fit each client ────────────────────────────────────────────────
        for client in clients:
            params, n_examples, metrics = client.fit(global_params, config={"round": rnd})
            round_client_params.append(params[0])
            round_metrics.append({"num_examples": n_examples, **metrics})
            sv = metrics.get("shap_vector")
            if sv:
                shap_vectors.append(np.array(sv, dtype=np.float32))

        # ── Consistency-aware aggregate ────────────────────────────────────
        agg_vec, global_consistency = aggregator.aggregate(
            server_round=rnd,
            client_weights=round_client_params,
            client_sizes=client_sizes,
            shap_vectors=shap_vectors,
        )
        global_params = [agg_vec]

        # ── Pooled metrics ─────────────────────────────────────────────────
        total_n = sum(m["num_examples"] for m in round_metrics)
        pooled_acc = sum(m["accuracy"] * m["num_examples"] for m in round_metrics) / total_n
        pooled_auc = sum(m["auc"]      * m["num_examples"] for m in round_metrics) / total_n
        pooled_f1  = sum(m["f1"]       * m["num_examples"] for m in round_metrics) / total_n

        # ── Communication overhead ─────────────────────────────────────────
        weight_bytes = sum(p.nbytes for p in round_client_params)
        shap_bytes   = sum(len(m.get("shap_vector", [])) * 4 for m in round_metrics)

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
            "variant":                 "ours",
            "pooled_accuracy":         round(pooled_acc, 6),
            "pooled_auc":              round(pooled_auc, 6),
            "pooled_f1":               round(pooled_f1, 6),
            "global_consistency":      round(global_consistency, 6),
            "consistency_gain":        consistency_gain,
            "weight_bytes_per_round":  weight_bytes,
            "shap_bytes_per_round":    shap_bytes,
            "per_client":              per_client_log,
        }
        all_rounds.append(round_data)

        with open(results_dir / f"round_{rnd:04d}.json", "w") as f:
            json.dump(round_data, f, indent=2)
        with open(results_dir / "all_rounds.jsonl", "a") as f:
            f.write(json.dumps(round_data) + "\n")

        print(
            f"[Round {rnd:2d}/{num_rounds}] ours | "
            f"acc={pooled_acc:.4f} | auc={pooled_auc:.4f} | "
            f"f1={pooled_f1:.4f} | consistency={global_consistency:.4f}"
        )

    return all_rounds


if __name__ == "__main__":
    print("[run_ours] Loading and partitioning data...")
    df = load_and_clean(str(DATA_CSV))
    clients_raw = partition_noniid(df)

    print("[run_ours] Building shared feature space...")
    shared_cols = build_shared_feature_columns(clients_raw)
    print(f"[run_ours] Shared features: {len(shared_cols)}")

    client_list = []
    client_sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            clients_raw[cid], feature_cols=shared_cols
        )
        print(f"  {cid}: train={len(X_tr):,} | test={len(X_te):,}")
        client_list.append(FederatedClient(
            client_id=cid,
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
        ))
        client_sizes.append(len(X_tr))

    all_rounds = run_consistency_aware_simulation(
        clients=client_list,
        client_sizes=client_sizes,
        num_rounds=NUM_ROUNDS,
        results_dir=RESULTS_DIR,
        consistency_gain=CONSISTENCY_GAIN,
    )

    final = all_rounds[-1]
    print(f"\n{'='*65}")
    print(f"  Consistency-Aware COMPLETE — {NUM_ROUNDS} rounds")
    print(f"  Final: acc={final['pooled_accuracy']:.4f} | "
          f"auc={final['pooled_auc']:.4f} | "
          f"f1={final['pooled_f1']:.4f} | "
          f"consistency={final['global_consistency']:.4f}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*65}")
