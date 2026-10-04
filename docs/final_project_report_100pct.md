# FedTrust-Credit — Final Project Report (100%)

> Course: CLOUD COMPUTING (BITE412L), VIT Vellore. Team: Prashaanth Raj J M (23BIT0173), Prasannaa V (23BIT0041), Haswanth K (23BIT0359). Guide: Dr. Siva Rama Krishnan S.
> Rule: every number below is from executed code (`results/`).

## 1. Abstract
Federated LightGBM across 3 non-IID LendingClub silos with explanation-consistency-aware aggregation (cosine similarity on mean-absolute TreeSHAP, D-020). 20-round comparison: Centralized vs FedAvg vs Ours, plus Byzantine defense, (epsilon,delta)-DP sweep, and German Credit generalization.

## 2. Grounded LendingClub Benchmarks (executed)
Source: `results/full_metrics_summary.json`.

| Metric | Centralized | FedAvg | Ours (gain=1.0) |
|---|---|---|---|
| Accuracy | 0.8564 | 0.8528 | **0.8534** |
| AUC | 0.7093 | 0.6554 | **0.6574** |
| F1 | 0.0668 | 0.0661 | 0.0665 |
| Final consistency | 0.9819 | 0.8911 | 0.8836 |
| Bandwidth/round | N/A | 118,324 B | 119,872 B (+1.3083%) |

Plots: `results/plots/{consistency,accuracy,auc,per_client_f1,communication_overhead,fairness_spread}.png`.

## 3. Byzantine Robustness (Task 2) — EXECUTED
Runner: `src/byzantine_attack.py` (Bank-3 label-flip `1→0`, 5 rounds, 15k/client subsample, 50 trees). Source: `results/byzantine/attack_results.json`.

| Round | FedAvg acc | Ours acc | FedAvg w₃ | Ours w₃ | Attacker a₃ |
|---|---|---|---|---|---|
| 1 | 0.7692 | 0.7692 | 0.3333 | 0.2758 | 0.0000 |
| 5 | 0.7683 | 0.7683 | 0.3333 | 0.2786 | 0.0000 |

- Poisoned Bank-3 SHAP fully diverges (a₃=0.0 every round); FedAvg keeps w₃=1/3 (blind), ours penalizes to ~0.277 (−5.6pp, honest clients 0.333→0.361).
- Accuracy gap negligible at this scale/rounds — defense visible in weights, honestly reported. Plot: `results/plots/byzantine_robustness.png`.
- Live demo: `POST /api/federated/simulate-round` with `attack_mode=label_flip` drops demo consistency ~0.81 → ~0.49 (verified 2026-10-04).

## 4. Differential Privacy (Task 1) — EXECUTED
Mechanism: `src/local_training.py:fit(config)` L2-clip (C=1.0) + Gaussian `sigma=C*sqrt(2 ln(1.25/delta))/epsilon`, delta=1e-5 (D-022). Sweep: `src/run_differential_privacy.py --rounds 3 --max-train 15000` (50 trees). Source: `results/dp/dp_sweep.json`.

| Epsilon | Final pooled acc (3 rounds) | Delta vs clean |
|---|---|---|
| clean (no DP) | 0.77337 | — |
| 10.0 | 0.77298 | −0.00039 |
| 5.0 | 0.77280 | −0.00057 |
| 1.0 | 0.77192 | −0.00145 |
| 0.5 | 0.77192 | −0.00145 |

Monotonic privacy-utility tradeoff as theory predicts; small magnitude because C=1.0 bounds prob-vector sensitivity and only 3 rounds. Plot: `results/plots/differential_privacy_tradeoff.png`. Limitation (stated): SHAP vectors un-noised.

## 5. German Credit Generalization (Task 3) — EXECUTED
Runner: `src/german_credit.py` (OpenML credit-g cached to `data/german/credit-g.csv`, history-partition with stratified-thirds fallback, LightGBM 100 trees, 5 rounds). Source: `results/german/german_results.json`.

| Metric | Centralized | FedAvg | Ours (gain=1.0) |
|---|---|---|---|
| Accuracy | 0.7400 | 0.7662 | 0.7662 |
| AUC | 0.7817 | 0.8279 | 0.8279 |
| F1 | 0.5094 | 0.6050 | 0.6050 |
| Consistency | 0.9759 | 0.9702 | **0.9717** |

- 48 one-hot features; partitions 334/333/333 (fallback thirds — history regex bins degenerated, documented in runner).
- Federated beats centralized here (small-n, 5 rounds); ours matches FedAvg accuracy with +0.0015 consistency. Plot: `results/plots/german_credit_benchmark.png`.

## 6. Cloud (Task 4)
Terraform: `aws_cloudwatch_dashboard.fedtrust_dashboard` in `terraform/aws_monitoring.tf` (4 widgets). Refresh: `src/deploy_cloudwatch_dashboard.py --dry-run` verified. Live: S3 registry, CloudWatch `FedTrustCredit/FL`, SNS >=35% PD, IAM instance profile (see `docs/CLOUD_DEPLOYMENT.md`).

## 7. Dashboard (Task 5)
`src/static/index.html` simulator now has Byzantine `attackSelect` + DP `dpEpsilonSlider`; `src/static/app.js:runSimulation()` forwards `attack_mode`/`dp_epsilon` to `POST /api/federated/simulate-round` (backward-compatible defaults).

## 8. Decisions & Repro
See `docs/DECISIONS.md` D-001→D-025, `docs/PROGRESS.md`, `pytest.ini`. Tests: `13 + 4` new (`tests/test_robustness_privacy.py`).
