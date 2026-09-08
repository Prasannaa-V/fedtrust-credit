"""
scratch/run_sweep_worker2.py — Worker 2 for CONSISTENCY_GAIN sweep (5.0, 10.0)
FedTrust-Credit | D-018 Gain Sweep Investigation
"""

import json
import sys
import time
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

print("[worker2] Loading data...")
df = load_and_clean(str(DATA_CSV))
partitions = partition_noniid(df)
shared_cols = build_shared_feature_columns(partitions)

gains = [5.0, 10.0]

for gain in gains:
    out_dir = SWEEP_DIR / f"gain_{gain}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}\n  [WORKER 2] STARTING GAIN={gain}\n{'='*65}")
    t0 = time.time()

    client_list = []
    client_sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            partitions[cid], feature_cols=shared_cols
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

    elapsed = time.time() - t0
    final = all_rounds[-1]
    ss_cons = float(np.mean([r["global_consistency"] for r in all_rounds[3:]]))

    res = {
        "gain": gain,
        "elapsed_sec": round(elapsed, 1),
        "final_accuracy": final["pooled_accuracy"],
        "final_auc": final["pooled_auc"],
        "final_f1": final["pooled_f1"],
        "final_consistency": final["global_consistency"],
        "steady_state_consistency_mean": ss_cons,
    }
    with open(out_dir / "sweep_metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[WORKER 2] GAIN={gain} DONE in {elapsed/60:.1f}m | final_cons={final['global_consistency']:.4f} | ss_cons={ss_cons:.4f}\n")

print("\n[WORKER 2] ALL TASKS COMPLETE.")
