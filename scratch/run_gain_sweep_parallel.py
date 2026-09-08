"""
scratch/run_gain_sweep_parallel.py — Parallel runner for CONSISTENCY_GAIN sweep (0.5, 2.0, 5.0, 10.0)
FedTrust-Credit | D-018 Gain Sweep Investigation

Uses multiprocessing to run all 4 gain values concurrently across 16 CPU cores.
Parent process loads and partitions data once, leveraging Linux fork copy-on-write memory sharing.
"""

import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_partition import (
    load_and_clean,
    partition_noniid,
    get_client_splits,
    build_shared_feature_columns,
)
from local_training import FederatedClient
from run_ours import run_consistency_aware_simulation

SWEEP_DIR = ROOT / "results" / "gain_sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
DATA_CSV = ROOT / "data" / "lending_club" / "loan.csv"
NUM_ROUNDS = 20

# Global shared variables in parent
SHARED_PARTITIONS = None
SHARED_COLS = None


def worker_task(gain: float):
    """Worker process executing 20 rounds of federated training for a single gain."""
    pid = os.getpid()
    gain_str = f"gain_{gain}"
    out_dir = SWEEP_DIR / gain_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set lightgbm threads per worker
    os.environ["OMP_NUM_THREADS"] = "4"

    print(f"[Worker PID {pid}] Starting GAIN={gain} -> {out_dir}")
    t0 = time.time()

    client_list = []
    client_sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            SHARED_PARTITIONS[cid], feature_cols=SHARED_COLS
        )
        client_list.append(FederatedClient(
            client_id=cid,
            X_train=X_tr,
            X_test=X_te,
            y_train=y_tr,
            y_test=y_te,
        ))
        client_sizes.append(len(X_tr))

    all_rounds = run_consistency_aware_simulation(
        clients=client_list,
        client_sizes=client_sizes,
        num_rounds=NUM_ROUNDS,
        results_dir=out_dir,
        consistency_gain=gain,
    )

    t_elapsed = time.time() - t0
    final = all_rounds[-1]
    ss_cons = float(np.mean([r["global_consistency"] for r in all_rounds[3:]]))

    res = {
        "gain": gain,
        "elapsed_sec": round(t_elapsed, 1),
        "final_accuracy": final["pooled_accuracy"],
        "final_auc": final["pooled_auc"],
        "final_f1": final["pooled_f1"],
        "final_consistency": final["global_consistency"],
        "steady_state_consistency_mean": ss_cons,
    }
    with open(out_dir / "sweep_metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[Worker PID {pid}] COMPLETED GAIN={gain} in {t_elapsed/60:.1f} min | final_cons={final['global_consistency']:.4f} | ss_cons={ss_cons:.4f}\n")
    return res


def main():
    global SHARED_PARTITIONS, SHARED_COLS

    print("[parallel_sweep] Loading Lending Club dataset once in parent process...")
    t_start = time.time()
    df = load_and_clean(str(DATA_CSV))
    SHARED_PARTITIONS = partition_noniid(df)
    SHARED_COLS = build_shared_feature_columns(SHARED_PARTITIONS)
    print(f"[parallel_sweep] Data ready in {time.time() - t_start:.1f}s. Forking 4 workers...")

    gains = [0.5, 2.0, 5.0, 10.0]

    ctx = get_context("fork")
    processes = []
    for g in gains:
        p = ctx.Process(target=worker_task, args=(g,))
        p.start()
        processes.append((g, p))

    print(f"[parallel_sweep] All 4 worker processes spawned: {[p.pid for _, p in processes]}")

    for g, p in processes:
        p.join()
        print(f"[parallel_sweep] Process for GAIN={g} joined with exitcode={p.exitcode}")

    print(f"\n[parallel_sweep] ALL WORKERS FINISHED in {(time.time() - t_start)/60:.1f} min total!")


if __name__ == "__main__":
    main()
