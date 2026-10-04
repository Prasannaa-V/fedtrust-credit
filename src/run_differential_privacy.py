"""
run_differential_privacy.py — Privacy vs Utility sweep (Task 1)
FedTrust-Credit | (epsilon, delta)-DP via clipped Gaussian mechanism (D-022)

Sweeps epsilon in [0.1, 10.0] with delta=1e-5, runs a short federated
simulation per epsilon using FederatedClient DP config, and plots the
privacy-utility tradeoff.

Outputs:
  results/dp/dp_sweep.json
  results/plots/differential_privacy_tradeoff.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_partition import (
    load_and_clean, partition_noniid,
    get_client_splits, build_shared_feature_columns,
)
from local_training import FederatedClient
from aggregation_strategy import explanation_consistency_aware_aggregate

DATA_CSV = ROOT / "data" / "lending_club" / "loan.csv"
OUT_DIR = ROOT / "results" / "dp"
PLOTS_DIR = ROOT / "results" / "plots"


def run_dp_simulation(partitions, shared_cols, epsilon, delta, clip,
                      num_rounds, gain=1.0, seed=42, max_train=None):
    clients = []
    sizes = []
    rng = np.random.RandomState(seed)
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            partitions[cid], feature_cols=shared_cols)
        if max_train is not None and len(X_tr) > max_train:
            idx = rng.choice(len(X_tr), size=int(max_train), replace=False)
            idx.sort()
            X_tr = X_tr.iloc[idx].reset_index(drop=True)
            y_tr = y_tr.iloc[idx].reset_index(drop=True)
        clients.append(FederatedClient(cid, X_tr, X_te, y_tr, y_te))
        sizes.append(len(X_tr))
    global_params = [np.array([0.5], dtype=np.float32)]
    acc_hist = []
    for rnd in range(1, num_rounds + 1):
        params, metrics_list, shap_vecs = [], [], []
        for i, c in enumerate(clients):
            cfg = {"round": rnd, "dp_seed": seed + rnd * 10 + i}
            if epsilon is not None:
                cfg.update({"dp_epsilon": epsilon, "dp_delta": delta,
                            "dp_clip": clip})
            p, n, m = c.fit(global_params, config=cfg)
            params.append(p[0])
            metrics_list.append({"num_examples": n, **m})
            if m.get("shap_vector"):
                shap_vecs.append(np.array(m["shap_vector"], dtype=np.float32))
        agg, cons = explanation_consistency_aware_aggregate(
            params, sizes, shap_vecs, consistency_gain=gain)
        global_params = [agg]
        total_n = sum(m["num_examples"] for m in metrics_list)
        pooled_acc = sum(m["accuracy"] * m["num_examples"]
                         for m in metrics_list) / total_n
        pooled_auc = sum(m["auc"] * m["num_examples"]
                         for m in metrics_list) / total_n
        acc_hist.append(float(pooled_acc))
        print(f"  [eps={epsilon} rnd {rnd}/{num_rounds}] "
              f"acc={pooled_acc:.4f} auc={pooled_auc:.4f} cons={cons:.4f}")
    return acc_hist


def save_plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = [r["epsilon"] if r["epsilon"] else 100.0 for r in rows]
    acc = [r["final_accuracy"] for r in rows]
    labels = [str(r["epsilon"]) if r["epsilon"] else "no-DP" for r in rows]
    order = np.argsort(eps)
    eps = [eps[i] for i in order]
    acc = [acc[i] for i in order]
    labels = [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(eps, acc, marker="o")
    for x, y, lb in zip(eps, acc, labels):
        ax.annotate(lb, (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget epsilon (log scale, larger = less noise)")
    ax.set_ylabel("Pooled accuracy")
    ax.set_title("Differential privacy vs utility (delta=1e-5, Gaussian mechanism)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[dp] plot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--epsilons", type=str, default="0.5,1.0,5.0,10.0")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--n-estimators", type=int, default=None)
    args = ap.parse_args()

    if args.n_estimators is not None:
        FederatedClient.LGB_PARAMS = {**FederatedClient.LGB_PARAMS,
                                      "n_estimators": int(args.n_estimators)}

    eps_list = [float(x) for x in args.epsilons.split(",") if x.strip()]
    print("[dp] loading + partitioning...")
    df = load_and_clean(str(DATA_CSV))
    parts = partition_noniid(df)
    shared = build_shared_feature_columns(parts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    # clean (no-DP) reference, short run
    print("\n=== epsilon=None (no DP clean reference) ===")
    hist = run_dp_simulation(parts, shared, None, args.delta, args.clip,
                             args.rounds, max_train=args.max_train)
    rows.append({"epsilon": None, "final_accuracy": hist[-1],
                 "accuracy_traj": hist})
    for eps in eps_list:
        print(f"\n=== epsilon={eps} ===")
        hist = run_dp_simulation(parts, shared, eps, args.delta, args.clip,
                                 args.rounds, max_train=args.max_train)
        rows.append({"epsilon": eps, "final_accuracy": hist[-1],
                     "accuracy_traj": hist})

    with open(OUT_DIR / "dp_sweep.json", "w") as f:
        json.dump({"delta": args.delta, "clip": args.clip,
                   "rounds": args.rounds, "rows": rows}, f, indent=2)
    print(f"[dp] results -> {OUT_DIR / 'dp_sweep.json'}")
    save_plot(rows, PLOTS_DIR / "differential_privacy_tradeoff.png")


if __name__ == "__main__":
    main()
