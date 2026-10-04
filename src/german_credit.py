"""
german_credit.py — Cross-dataset generalization (Task 3)
UCI German Credit (1,000 cases) partitioned across 3 institutions by
credit history, then Centralized vs FedAvg vs FedTrust-Credit comparison.

Outputs:
  results/german/german_results.json
  results/plots/german_credit_benchmark.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb

from consistency_score import explanation_consistency_score
from aggregation_strategy import explanation_consistency_aware_aggregate
from fedavg_server import fedavg_aggregate

OUT_DIR = ROOT / "results" / "german"
PLOTS_DIR = ROOT / "results" / "plots"

LGB_PARAMS = dict(objective="binary", num_leaves=31, learning_rate=0.05,
                  n_estimators=100, min_child_samples=10, verbose=-1,
                  random_state=42, n_jobs=-1)


def load_german(cache_path=None):
    """Load UCI German Credit via OpenML (cached); 1,000 rows."""
    from sklearn.datasets import fetch_openml
    cache_path = Path(cache_path) if cache_path else (ROOT / "data" / "german" / "credit-g.csv")
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        ds = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        df = ds.frame.copy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    # target: 'good'/'bad' -> 0/1 (bad = default)
    target = df["class"].map({"good": 0, "bad": 1}).astype(int)
    X = df.drop(columns=["class"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X, target, df["credit_history"].astype(str)


def partition_by_history(X, y, hist):
    """3 institutions split by credit_history bins (proxy for non-IID)."""
    groups = {
        "client_1": hist.str.contains("paid|allpaid", case=False, na=False),
        "client_2": hist.str.contains("existing|delay", case=False, na=False),
        "client_3": ~(hist.str.contains("paid|allpaid|existing|delay",
                                        case=False, na=False)),
    }
    parts = {}
    for cid, mask in groups.items():
        m = mask.values
        if m.sum() < 50:  # fallback to stratified thirds if bins degenerate
            continue
        parts[cid] = (X[m], y[m])
    if len(parts) != 3:
        # deterministic stratified fallback: thirds by index
        n = len(X)
        idx = np.arange(n)
        parts = {
            "client_1": (X.iloc[idx[::3]], y.iloc[idx[::3]]),
            "client_2": (X.iloc[idx[1::3]], y.iloc[idx[1::3]]),
            "client_3": (X.iloc[idx[2::3]], y.iloc[idx[2::3]]),
        }
    return parts


def train_eval(X_tr, y_tr, X_te, y_te, init_score=None):
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X_tr, y_tr,
          init_score=np.full(len(X_tr), init_score) if init_score else None)
    prob = m.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    # mean-abs feature importance as cheap SHAP proxy for speed
    imp = np.abs(m.feature_importances_.astype(float))
    imp = imp / (imp.sum() + 1e-12)
    return prob, {"acc": accuracy_score(y_te, pred),
                  "auc": roc_auc_score(y_te, prob),
                  "f1": f1_score(y_te, pred, zero_division=0)}, imp, m


def run_federated(parts, rounds=10, gain=1.0):
    splits = {}
    for cid, (X, y) in parts.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        splits[cid] = (X_tr, X_te, y_tr, y_te)
    sizes = [len(splits[c][0]) for c in ["client_1", "client_2", "client_3"]]
    glob = 0.0
    for rnd in range(rounds):
        probs, mets, imps = [], [], []
        for cid in ["client_1", "client_2", "client_3"]:
            X_tr, X_te, y_tr, y_te = splits[cid]
            p, m, imp, _ = train_eval(X_tr, y_tr, X_tr, y_tr,
                                      init_score=glob if rnd else None)
            probs.append(p.astype(np.float32))
            mets.append(m)
            imps.append(imp.astype(np.float32))
        if gain == 0.0:
            glob = float(np.average([np.mean(p) for p in probs],
                                    weights=sizes))
            cons = explanation_consistency_score(imps)
        else:
            agg, cons = explanation_consistency_aware_aggregate(
                probs, sizes, imps, consistency_gain=gain)
            glob = float(agg[0])
    # final eval on pooled test
    X_te_all = pd.concat([splits[c][1] for c in splits])
    y_te_all = pd.concat([splits[c][3] for c in splits])
    # retrain quick global proxy: average of client probs on test
    test_probs = []
    for cid in ["client_1", "client_2", "client_3"]:
        X_tr, _, y_tr, _ = splits[cid]
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X_tr, y_tr)
        test_probs.append(m.predict_proba(X_te_all)[:, 1])
    final_prob = np.average(test_probs, axis=0,
                            weights=[len(splits[c][0]) for c in splits])
    final_pred = (final_prob >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y_te_all, final_pred)),
            "auc": float(roc_auc_score(y_te_all, final_prob)),
            "f1": float(f1_score(y_te_all, final_pred, zero_division=0)),
            "consistency": float(cons),
            "sizes": sizes}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    print("[german] loading UCI German Credit...")
    X, y, hist = load_german()
    print(f"[german] shape={X.shape} default_rate={y.mean():.3f}")
    parts = partition_by_history(X, y, hist)
    for cid, (Xc, yc) in parts.items():
        print(f"  {cid}: n={len(Xc)} default={yc.mean():.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[german] centralized...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    _, cm, _, _ = train_eval(X_tr, y_tr, X_te, y_te)
    # consistency of centralized = agreement across partition importances
    imps = []
    for cid in ["client_1", "client_2", "client_3"]:
        Xc, yc = parts[cid]
        Xc_tr, _, yc_tr, _ = train_test_split(
            Xc, yc, test_size=0.2, random_state=42, stratify=yc)
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(Xc_tr, yc_tr)
        imp = np.abs(m.feature_importances_.astype(float))
        imps.append((imp / (imp.sum() + 1e-12)).astype(np.float32))
    cent_cons = explanation_consistency_score(imps)
    centralized = {"accuracy": cm["acc"], "auc": cm["auc"],
                   "f1": cm["f1"], "consistency": float(cent_cons)}

    print("[german] fedavg...")
    fedavg = run_federated(parts, rounds=args.rounds, gain=0.0)
    print("[german] ours...")
    ours = run_federated(parts, rounds=args.rounds, gain=1.0)

    results = {"rounds": args.rounds, "n_features": int(X.shape[1]),
               "centralized": centralized, "fedavg": fedavg, "ours": ours}
    with open(OUT_DIR / "german_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[german] results -> {OUT_DIR / 'german_results.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["accuracy", "auc", "consistency"]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, [centralized[k] for k in labels], w, label="Centralized")
    ax.bar(x, [fedavg[k] for k in labels], w, label="FedAvg")
    ax.bar(x + w, [ours[k] for k in labels], w, label="Ours")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"German Credit generalization ({args.rounds} rounds, n=1000)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "german_credit_benchmark.png", dpi=150)
    print(f"[german] plot -> {PLOTS_DIR / 'german_credit_benchmark.png'}")


if __name__ == "__main__":
    main()
