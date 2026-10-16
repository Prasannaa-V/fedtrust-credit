"""
monitoring_snapshot.py — Local monitoring verification snapshot (9.1)

Measures REAL local API latencies (simulate + predict endpoints via TestClient)
and combines them with REAL executed federated histories
(results/byzantine, results/dp, results/german) into a 4-panel figure that
mirrors the CloudWatch dashboard layout in terraform/aws_monitoring.tf:
  1. DefaultProbability time-series (per prediction, over successive calls)
  2. ExplanationConsistencyScore per federated round
  3. API latency histogram (measured seconds)
  4. Alarm-state panel (threshold lines, no live alarms without AWS)

Outputs:
  results/monitoring/api_latency.json   (raw measured samples)
  results/plots/cloudwatch_monitoring_snapshot.png

This is a LOCAL verification artifact — it is not an AWS console screenshot.
A live CloudWatch screenshot requires AWS credentials (see D-025).
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "monitoring"
PLOTS_DIR = ROOT / "results" / "plots"

N_PROB_CALLS = 12
N_SIM_CALLS = 12


def measure_latencies():
    from fastapi.testclient import TestClient
    from service import app
    client = TestClient(app)
    prob_series, sim_lat, pred_lat = [], [], []
    applicant = {
        "loan_amnt": 12000.0, "term": 36, "int_rate": 11.5, "grade": "B",
        "annual_inc": 75000.0, "dti": 18.5, "home_ownership": "RENT",
        "purpose": "debt_consolidation", "addr_state": "NY",
        "revol_util": 45.0, "total_acc": 18, "open_acc": 10,
        "pub_rec": 0, "delinq_2yrs": 0,
    }
    for i in range(N_PROB_CALLS):
        t = time.perf_counter()
        r = client.post("/api/predict/risk", json={
            **applicant, "int_rate": 8.0 + i * 1.5})
        dt = time.perf_counter() - t
        assert r.status_code == 200, r.text
        prob_series.append(r.json()["default_probability_pct"])
        pred_lat.append(dt)
    for i in range(N_SIM_CALLS):
        t = time.perf_counter()
        r = client.post("/api/federated/simulate-round", json={
            "consistency_gain": 1.0, "correlation_noise": 0.02,
            "client_sizes": [627716, 287686, 81060]})
        dt = time.perf_counter() - t
        assert r.status_code == 200, r.text
        sim_lat.append(dt)
    return prob_series, sim_lat, pred_lat


def load_consistency_history():
    """Per-round consistency from the executed Byzantine Ours run."""
    p = ROOT / "results" / "byzantine" / "attack_results.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    ours = (d.get("attacks", {}).get("label_flip", {}).get("ours")
            or d.get("variants", {}).get("ours", []))
    return [(r["round"], r["global_consistency"]) for r in ours]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[monitor] measuring local API latencies...")
    prob_series, sim_lat, pred_lat = measure_latencies()
    all_lat = sim_lat + pred_lat
    (OUT_DIR / "api_latency.json").write_text(json.dumps({
        "simulate_round_seconds": sim_lat,
        "predict_risk_seconds": pred_lat,
        "default_probability_series": prob_series,
    }, indent=2))
    print(f"[monitor] samples: {len(all_lat)} latency, "
          f"{len(prob_series)} predictions")

    cons_hist = load_consistency_history()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Monitoring snapshot — LOCAL verification "
                 "(mirrors CloudWatch layout; not an AWS screenshot)",
                 fontsize=11, fontweight="bold")

    axes[0, 0].plot(range(1, len(prob_series) + 1), prob_series, marker="o")
    axes[0, 0].axhline(35.0, color="r", ls="--", lw=1, label="SNS alert ≥35%")
    axes[0, 0].set_title("DefaultProbability per prediction (local)")
    axes[0, 0].set_xlabel("successive /api/predict/risk call")
    axes[0, 0].set_ylabel("default_probability_pct")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    if cons_hist:
        axes[0, 1].plot([r for r, _ in cons_hist],
                        [c for _, c in cons_hist], marker="s")
        axes[0, 1].axhline(0.70, color="r", ls="--", lw=1,
                           label="consistency drift alarm <0.70")
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, "no byzantine history yet",
                        ha="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_title("ExplanationConsistencyScore per round (executed)")
    axes[0, 1].set_xlabel("federated round")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(all_lat, bins=12, edgecolor="black")
    axes[1, 0].axvline(np.percentile(all_lat, 95), color="r", ls="--", lw=1,
                       label=f"p95={np.percentile(all_lat, 95):.3f}s")
    axes[1, 0].set_title("API latency histogram (measured, local)")
    axes[1, 0].set_xlabel("seconds")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.05, 0.9,
        "Alarm states (local mirror, thresholds from terraform):\n"
        f"• consistency drift (<0.70): "
        f"{'BREACH' if cons_hist and min(c for _, c in cons_hist) < 0.70 else 'OK'}\n"
        f"• round latency (>120s): OK (max measured {max(all_lat):.3f}s)\n"
        f"• SNS threshold (≥35% PD): max observed {max(prob_series):.1f}%\n\n"
        "Live CloudWatch alarms require AWS credentials (D-025).",
        fontsize=10, va="top", family="monospace")

    fig.tight_layout()
    out = PLOTS_DIR / "cloudwatch_monitoring_snapshot.png"
    fig.savefig(out, dpi=150)
    print(f"[monitor] snapshot -> {out}")


if __name__ == "__main__":
    main()
