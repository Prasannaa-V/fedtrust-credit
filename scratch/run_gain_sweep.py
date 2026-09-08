"""
scratch/run_gain_sweep.py — Run CONSISTENCY_GAIN sweep (0.5, 2.0, 5.0, 10.0)
FedTrust-Credit | D-018 Gain Sweep Investigation
"""

import json
import sys
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

print("[gain_sweep] Loading dataset...")
df = load_and_clean(str(DATA_CSV))
partitions = partition_noniid(df)
shared_cols = build_shared_feature_columns(partitions)

gains = [0.5, 2.0, 5.0, 10.0]

for gain in gains:
    gain_str = f"gain_{gain}"
    out_dir = SWEEP_DIR / gain_str
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  RUNNING SWEEP: CONSISTENCY_GAIN = {gain} (20 rounds)")
    print(f"  Destination: {out_dir}")
    print(f"{'='*70}")

    # Build fresh clients for each run
    client_list = []
    client_sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(partitions[cid], feature_cols=shared_cols)
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

    final = all_rounds[-1]
    ss_cons = float(np.mean([r["global_consistency"] for r in all_rounds[3:]]))
    print(f"\n[gain_sweep] Finished GAIN={gain}:")
    print(f"  Final Round (20): acc={final['pooled_accuracy']:.4f} | auc={final['pooled_auc']:.4f} | f1={final['pooled_f1']:.4f} | cons={final['global_consistency']:.4f}")
    print(f"  Steady-State (rounds 4-20 mean): cons={ss_cons:.4f}")

print("\n[gain_sweep] ALL 4 RUNS COMPLETE.")
