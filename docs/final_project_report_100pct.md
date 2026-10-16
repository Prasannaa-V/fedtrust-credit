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
Runner: `src/byzantine_attack.py --attacks none,label_flip,weight_poison --rounds 10 --max-train 15000` (50 trees). Source: `results/byzantine/attack_results.json`.

| Attack | FedAvg final acc | Ours final acc | FedAvg w₃ | Ours w₃ | Attacker a₃ |
|---|---|---|---|---|---|
| none (clean) | 0.7693 | 0.7691 | 0.3333 | 0.3376 | 0.6459 |
| label_flip (Bank-3 `1→0`) | 0.7683 | 0.7681 | 0.3333 | 0.2769 | 0.0000 |
| weight_poison (σ=2.0 post-train) | 0.7688 | 0.7687 | 0.3333 | 0.3374 | 0.6444 |

- Label-flip fully diverges Bank-3 SHAP (honest a₃≈0.646 → 0.0000); FedAvg keeps w₃=1/3 (blind), ours penalizes to ~0.277 (−5.6pp, honest clients 0.333→0.361).
- Attacked FedAvg degrades −0.0010 vs clean (0.7693→0.7683); ours holds at the same level (0.7681) — the accuracy effect is within noise at this scale, so the demonstrated defense is the weight penalty, honestly reported. Plots: `byzantine_robustness.png`, `byzantine_clean_vs_attacked.png`.
- Weight-poison (post-training noise) leaves SHAP untouched (a₃≈0.64) and accuracy unchanged — consistency scoring cannot see parameter-space attacks that bypass explanations. **Closed by the update-norm pre-filter** (`src/byzantine_defense.py`, `--defense update-norm`): RMS-deviation tripwire (median + 3·MAD, 10% relative floor) flags Bank-3 every round `[F,F,T]` under weight-poison and excludes it (w₃=0.0), while never flagging honest or label-flip rounds; evidence in `results/byzantine/attack_results_defense.json`. Coarse by design — catches gross corruption, not stealth.
- Live demo: `POST /api/federated/simulate-round` with `attack_mode=label_flip` drops demo consistency ~0.81 → ~0.49 (verified 2026-10-04).

## 4. Differential Privacy (Task 1) — EXECUTED
Mechanism: `src/local_training.py:fit(config)` L2-clip (C=1.0) + Gaussian `sigma=C*sqrt(2 ln(1.25/delta))/epsilon`, delta=1e-5 (D-022). Sweep: `src/run_differential_privacy.py --rounds 3 --max-train 15000` (50 trees). Source: `results/dp/dp_sweep.json`.

| Epsilon | Final pooled acc (3 rounds) | Delta vs clean | σ (C=1.0, δ=1e-5) | Mean update L2 |
|---|---|---|---|---|
| clean (no DP) | 0.77337 | — | 0.000 | 19.965 |
| 10.0 | 0.77298 | −0.00039 | 0.484 | 27.317 |
| 5.0 | 0.77280 | −0.00057 | 0.969 | 27.695 |
| 1.0 | 0.77192 | −0.00145 | 4.845 | 13.002 |
| 0.5 | 0.77192 | −0.00145 | 9.690 | 14.723 |

Measured update sensitivity: per-round mean client-update L2 ≈ 13–28 (max ≈ 49.7 on round 1 from the 0.5 cold-start signal); the clip C=1.0 bounds each shared vector before Gaussian perturbation, which is why utility loss stays ≤0.0015. Inset panel in the tradeoff plot shows the clean-run sensitivity trajectory. Per-round norms live in `results/dp/dp_sweep.json` → `rows[].sensitivity[]`, reported by `FederatedClient.fit()` metrics (`update_l2_norm`, `dp_sigma`, `dp_pre_noise_norm`).
Composition (RDP accountant, `src/dp_accountant.py`, Mironov Prop. 7, one release per client per round → n=9, δ′=1e-5):

| Per-release ε | σ | Composed total ε (n=9) |
|---|---|---|
| 10.0 | 0.484 | 49.86 |
| 5.0 | 0.969 | 20.14 |
| 1.0 | 4.845 | 3.16 |
| 0.5 | 9.690 | 1.53 |

Formal guarantee (D-022/D-026, honest scope): each DP-enabled client release satisfies per-message `(ε,δ)`-DP via the Gaussian mechanism; the table above is the RDP-composed total budget, not a single-release claim. Limitation (stated): SHAP vectors un-noised unless `dp_shap_epsilon` is set (see below).

## 5. German Credit Generalization (Task 3) — EXECUTED (20 rounds)
Runner: `src/german_credit.py` (OpenML credit-g cached to `data/german/credit-g.csv`, disjoint exact-category history partition, LightGBM 100 trees, 20 rounds). Source: `results/german/german_results.json`.

| Metric | Centralized | FedAvg | Ours (gain=1.0) |
|---|---|---|---|
| Accuracy | 0.7700 | 0.7711 | 0.7711 |
| AUC | 0.7957 | 0.7519 | 0.7519 |
| F1 | 0.5490 | 0.5208 | 0.5208 |
| Consistency | 0.9360 | 0.9224 | **0.9226** |

- 48 one-hot features; disjoint partitions — client_1 `existing paid` (530, 31.9% default), client_2 `critical/other existing` (293, 17.1%), client_3 delayed/all-paid/no-credits (177, 45.8%). Genuinely non-IID.
- Variant-aware fairness (each variant's own final-round models on home test splits): per-client acc 0.7075/0.8644/0.7222, spread 15.69pp, F1 spread 29.37pp — numerically identical for FedAvg vs ours because high explanation agreement leaves the distilled models coincident (consistent with the D-018 null-effect finding, honestly reported rather than tuned away).
- Ours matches FedAvg accuracy with +0.0002 consistency. Plot: `results/plots/german_credit_benchmark.png` (bars + per-round test-accuracy trajectory).

## 6. Cloud (Task 4) — widgets + telemetry executed, live screenshot pending creds
Terraform: `aws_cloudwatch_dashboard.fedtrust_dashboard` in `terraform/aws_monitoring.tf` (DefaultProbability time-series, ConsistencyScore per-round, **APILatencySeconds p95 histogram**, alarm panel). Service now emits `APILatencySeconds` (Endpoint dimension) from timed handlers via `AWSCloudManager.log_latency_to_cloudwatch()`; deploy script mirrors the layout (`--dry-run` verified, histogram `view: bar` covered by test).
Local verification (executed): `src/monitoring_snapshot.py` → `results/plots/cloudwatch_monitoring_snapshot.png` with real measured latencies (`results/monitoring/api_latency.json`), executed consistency history, and threshold panel — explicitly labeled local, not an AWS capture. Credentialed close-out: `docs/AWS_SCREENSHOT_RUNBOOK.md` (10-min steps). Live: S3 registry, CloudWatch `FedTrustCredit/FL`, SNS >=35% PD, IAM instance profile (see `docs/CLOUD_DEPLOYMENT.md`). Blocker (verified): this environment has no boto3/Terraform/AWS CLI/credentials, so `put_dashboard` + console screenshot must run where AWS is configured (D-029).

## 7. Dashboard (Task 5)
`src/static/index.html` simulator now has Byzantine `attackSelect` + DP `dpEpsilonSlider`; `src/static/app.js:runSimulation()` forwards `attack_mode`/`dp_epsilon` to `POST /api/federated/simulate-round` (backward-compatible defaults).

## 8. Decisions & Repro
See `docs/DECISIONS.md` D-001→D-025, `docs/PROGRESS.md`, `pytest.ini`. Tests: `13 + 4` new (`tests/test_robustness_privacy.py`).
