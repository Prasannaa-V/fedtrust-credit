"""
byzantine_attack.py — Adversarial Robustness & Byzantine Client Defense
FedTrust-Credit | Task 2 (100% roadmap)

Simulates a malicious client (Bank 3) under two attacks:
  1. label_flip    : flip default labels (1 -> 0) before local training
  2. weight_poison : add Gaussian noise to client_3 parameter vector post-training

Runs the same poisoned federation under:
  - fedavg (CONSISTENCY_GAIN=0.0, no defense)
  - ours   (CONSISTENCY_GAIN=1.0, consistency-aware down-weighting)

Logs per-round pooled accuracy, global consistency, attacker agreement a_3
and attacker adjusted weight w_3* so the defense is directly observable.

Outputs (all from real execution):
  results/byzantine/attack_results.json
  results/plots/byzantine_robustness.png
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
from consistency_score import explanation_consistency_score, per_client_agreement
from fedavg_server import fedavg_aggregate
from aggregation_strategy import explanation_consistency_aware_aggregate

DATA_CSV = ROOT / "data" / "lending_club" / "loan.csv"
OUT_DIR = ROOT / "results" / "byzantine"
PLOTS_DIR = ROOT / "results" / "plots"


def build_poisoned_clients(partitions, shared_cols, attack="label_flip",
                           flip_frac=1.0, poison_scale=2.0, seed=42,
                           max_train=None, n_estimators=None):
    """Build 3 FederatedClients; poison client_3 training labels if requested."""
    if n_estimators is not None:
        FederatedClient.LGB_PARAMS = {**FederatedClient.LGB_PARAMS,
                                      "n_estimators": int(n_estimators)}
    rng = np.random.RandomState(seed)
    clients = []
    sizes = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, X_te, y_tr, y_te = get_client_splits(
            partitions[cid], feature_cols=shared_cols)
        if max_train is not None and len(X_tr) > max_train:
            idx = rng.choice(len(X_tr), size=int(max_train), replace=False)
            idx.sort()
            X_tr = X_tr.iloc[idx].reset_index(drop=True)
            y_tr = y_tr.iloc[idx].reset_index(drop=True)
            n_te = min(len(X_te), max(2000, int(max_train * 0.25)))
            X_te = X_te.iloc[:n_te].reset_index(drop=True)
            y_te = y_te.iloc[:n_te].reset_index(drop=True)
        if cid == "client_3" and attack == "label_flip":
            y_tr = y_tr.copy()
            pos_idx = np.where(y_tr.values == 1)[0]
            n_flip = int(len(pos_idx) * flip_frac)
            flip_idx = rng.choice(pos_idx, size=n_flip, replace=False) if n_flip else []
            y_tr.iloc[flip_idx] = 0
            print(f"[byzantine] label_flip: flipped {n_flip}/{len(pos_idx)} "
                  f"defaults in client_3 (frac={flip_frac})")
        c = FederatedClient(client_id=cid, X_train=X_tr, X_test=X_te,
                            y_train=y_tr, y_test=y_te)
        # stash poison config for weight attack applied post-fit
        c._byz_attack = attack if cid == "client_3" else "none"
        c._byz_scale = poison_scale
        c._byz_rng = np.random.RandomState(seed + 7)
        clients.append(c)
        sizes.append(len(X_tr))
    return clients, sizes


def run_variant(clients, sizes, num_rounds, gain, attack, poison_scale):
    """Run one poisoned federation; return per-round records."""
    global_params = [np.array([0.5], dtype=np.float32)]
    records = []
    for rnd in range(1, num_rounds + 1):
        params, metrics_list, shap_vecs = [], [], []
        for client in clients:
            p, n, m = client.fit(global_params, config={"round": rnd})
            vec = p[0].astype(np.float32)
            # weight poisoning: corrupt client_3 update AFTER training
            if getattr(client, "_byz_attack", "none") == "weight_poison" \
                    and client.client_id == "client_3" and attack == "weight_poison":
                vec = vec + client._byz_rng.normal(
                    0, poison_scale, size=vec.shape).astype(np.float32)
                p = [vec]
            params.append(vec)
            metrics_list.append({"num_examples": n, **m})
            if m.get("shap_vector"):
                shap_vecs.append(np.array(m["shap_vector"], dtype=np.float32))
        if gain == 0.0:
            agg = fedavg_aggregate(params, sizes)
            cons = float(explanation_consistency_score(shap_vecs)) if shap_vecs else 0.0
            agreements = per_client_agreement(shap_vecs) if shap_vecs else [0, 0, 0]
            total = sum(sizes)
            adj_w = [s / total for s in sizes]
        else:
            agg, cons = explanation_consistency_aware_aggregate(
                client_weights=params, client_sizes=sizes,
                shap_vectors=shap_vecs, consistency_gain=gain)
            agreements = per_client_agreement(shap_vecs)
            total = sum(sizes)
            base = [s / total for s in sizes]
            raw = [base[i] * (1.0 + gain * agreements[i]) for i in range(3)]
            s = sum(raw)
            adj_w = [w / s for w in raw]
        global_params = [agg]
        total_n = sum(m["num_examples"] for m in metrics_list)
        pooled_acc = sum(m["accuracy"] * m["num_examples"]
                         for m in metrics_list) / total_n
        records.append({
            "round": rnd,
            "gain": gain,
            "pooled_accuracy": round(float(pooled_acc), 6),
            "global_consistency": round(float(cons), 6),
            "attacker_agreement": round(float(agreements[2]), 6),
            "attacker_weight": round(float(adj_w[2]), 6),
            "weights": [round(float(w), 6) for w in adj_w],
        })
        print(f"[rnd {rnd:2d}] gain={gain} acc={pooled_acc:.4f} "
              f"cons={cons:.4f} a3={agreements[2]:.4f} w3={adj_w[2]:.4f}")
    return records


def save_plot(fedavg_recs, ours_recs, attack, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds = [r["round"] for r in fedavg_recs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(rounds, [r["pooled_accuracy"] for r in fedavg_recs],
                 marker="o", label="FedAvg (gain=0, no defense)")
    axes[0].plot(rounds, [r["pooled_accuracy"] for r in ours_recs],
                 marker="s", label="Ours (gain=1.0, consistency-aware)")
    axes[0].set_xlabel("Federated round")
    axes[0].set_ylabel("Pooled accuracy")
    axes[0].set_title(f"Pooled accuracy under {attack} (Bank 3 malicious)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(rounds, [r["attacker_weight"] for r in fedavg_recs],
                 marker="o", label="FedAvg attacker weight")
    axes[1].plot(rounds, [r["attacker_weight"] for r in ours_recs],
                 marker="s", label="Ours attacker weight")
    axes[1].set_xlabel("Federated round")
    axes[1].set_ylabel("Bank-3 aggregation weight")
    axes[1].set_title("Attacker down-weighting via explanation consensus")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[byzantine] plot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=["label_flip", "weight_poison"],
                    default="label_flip")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--flip-frac", type=float, default=1.0)
    ap.add_argument("--poison-scale", type=float, default=2.0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Subsample cap per client train set for fast eval")
    ap.add_argument("--n-estimators", type=int, default=None,
                    help="Override LightGBM n_estimators for fast eval")
    args = ap.parse_args()

    print("[byzantine] loading + partitioning...")
    df = load_and_clean(str(DATA_CSV))
    parts = partition_noniid(df)
    shared = build_shared_feature_columns(parts)
    print(f"[byzantine] shared features: {len(shared)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"attack": args.attack, "rounds": args.rounds,
               "flip_frac": args.flip_frac, "poison_scale": args.poison_scale,
               "variants": {}}

    for gain, name in [(0.0, "fedavg"), (args.gain, "ours")]:
        clients, sizes = build_poisoned_clients(
            parts, shared, attack=args.attack,
            flip_frac=args.flip_frac, poison_scale=args.poison_scale,
            max_train=args.max_train, n_estimators=args.n_estimators)
        print(f"\n=== variant={name} gain={gain} attack={args.attack} ===")
        recs = run_variant(clients, sizes, args.rounds, gain,
                           args.attack, args.poison_scale)
        summary["variants"][name] = recs

    with open(OUT_DIR / "attack_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[byzantine] results -> {OUT_DIR / 'attack_results.json'}")
    save_plot(summary["variants"]["fedavg"], summary["variants"]["ours"],
              args.attack, PLOTS_DIR / "byzantine_robustness.png")


if __name__ == "__main__":
    main()
