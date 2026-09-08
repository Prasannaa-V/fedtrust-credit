"""
run_experiment.py — Unified experiment runner for all three variants
FedTrust-Credit | Day 3 (compressed schedule)

Runs three variants from the same codebase under identical conditions:
  1. fedavg      — CONSISTENCY_GAIN=0.0 (standard FedAvg baseline)
  2. ours        — CONSISTENCY_GAIN=1.0 (explanation-consistency-aware aggregation)
  3. centralized — pooled data, single-model benchmark

Follows D-014 (custom deterministic in-process simulation loop).
All metrics written to /results/ from real execution only.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_partition import (
    load_and_clean,
    partition_noniid,
    get_client_splits,
    build_shared_feature_columns,
)
from local_training import FederatedClient
from fedavg_server import run_fedavg_simulation
from run_ours import run_consistency_aware_simulation
from centralized_baseline import run_centralized_baseline
from evaluation import main as run_evaluation_main

DATA_CSV = ROOT / "data" / "lending_club" / "loan.csv"
NUM_ROUNDS = 20
CONSISTENCY_GAIN = 1.0  # D-007

Variant = Literal["fedavg", "ours", "centralized", "all", "eval"]


def prepare_environment():
    """Load and partition data, and compute shared feature columns."""
    print(f"[run_experiment] Loading Lending Club dataset: {DATA_CSV}...")
    df = load_and_clean(str(DATA_CSV))
    partitions = partition_noniid(df)

    print("[run_experiment] Building shared feature columns across partitions...")
    shared_cols = build_shared_feature_columns(partitions)
    print(f"[run_experiment] Shared feature space: {len(shared_cols)} features.")

    return partitions, shared_cols


def build_clients(partitions: dict, shared_cols: list[str]) -> tuple[list[FederatedClient], list[int]]:
    """Instantiate fresh FederatedClient objects for each partition."""
    clients = []
    client_sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        cdf = partitions[cid]
        X_tr, X_te, y_tr, y_te = get_client_splits(cdf, feature_cols=shared_cols)
        client = FederatedClient(
            client_id=cid,
            X_train=X_tr,
            X_test=X_te,
            y_train=y_tr,
            y_test=y_te,
        )
        clients.append(client)
        client_sizes.append(len(y_tr))
    return clients, client_sizes


def main():
    parser = argparse.ArgumentParser(description="FedTrust-Credit Unified Runner")
    parser.add_argument(
        "--variant",
        choices=["fedavg", "ours", "centralized", "all", "eval"],
        default="eval",
        help="Which variant to run: fedavg | ours | centralized | all | eval (default: eval)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help="Number of federated rounds (default: 20)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=CONSISTENCY_GAIN,
        help="CONSISTENCY_GAIN hyperparameter for 'ours' (default: 1.0)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Custom results directory for output (optional)",
    )
    args = parser.parse_args()

    if args.variant == "eval":
        print("\n[run_experiment] Running evaluation & plotting over existing results...")
        run_evaluation_main()
        return

    partitions, shared_cols = prepare_environment()

    if args.variant in ("fedavg", "all"):
        print("\n>>> STARTING FEDAVG BASELINE EXPERIMENT <<<")
        clients, client_sizes = build_clients(partitions, shared_cols)
        results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results" / "fedavg_baseline"
        run_fedavg_simulation(
            clients=clients,
            client_sizes=client_sizes,
            num_rounds=args.rounds,
            results_dir=results_dir,
            variant="fedavg",
        )

    if args.variant in ("ours", "all"):
        print(f"\n>>> STARTING FEDTRUST-CREDIT (OURS, GAIN={args.gain}) EXPERIMENT <<<")
        clients, client_sizes = build_clients(partitions, shared_cols)
        results_dir = Path(args.results_dir) if args.results_dir else ROOT / "results" / "consistency_aware"
        run_consistency_aware_simulation(
            clients=clients,
            client_sizes=client_sizes,
            num_rounds=args.rounds,
            results_dir=results_dir,
            consistency_gain=args.gain,
        )

    if args.variant in ("centralized", "all"):
        print("\n>>> STARTING CENTRALIZED BENCHMARK EXPERIMENT <<<")
        splits = {}
        for cid in ["client_1", "client_2", "client_3"]:
            cdf = partitions[cid]
            X_tr, X_te, y_tr, y_te = get_client_splits(cdf, feature_cols=shared_cols)

            splits[cid] = {"X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te}

        results_dir = ROOT / "results" / "centralized_baseline"
        run_centralized_baseline(client_data=splits, results_dir=results_dir)

    print("\n>>> GENERATING ALL EVALUATION PLOTS AND SUMMARY <<<")
    run_evaluation_main()
    print("\n[run_experiment] Experiment pipeline finished successfully.")


if __name__ == "__main__":
    main()
