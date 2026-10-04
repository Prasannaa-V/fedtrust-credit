# FedTrust-Credit — Viva Presentation Guide (10 min)

## Script (10:00)
1. **Problem (1:30)** — banks can't pool data (GDPR/Fair Lending); isolated models bias. 1.34M LendingClub loans, 3 non-IID silos (grade/state/time).
2. **Innovation (2:00)** — explanation-consistency-aware aggregation: TreeSHAP mean-abs vectors → pairwise **cosine** consensus (D-020, not Spearman) → `w_i*=(n_i/Σn)(1+γ·a_i)`, normalized. γ=0 recovers FedAvg.
3. **Live demo (3:00)** — open `/`, Benchmarks tab (85.34% / 0.6574 / +1.31%); Simulator: γ 0→1, then `attackSelect=label_flip` (consistency ~0.81→~0.49, Bank-3 weight penalized); Risk tab: score prime vs subprime persona, show SHAP bars + 3-bank perspectives; Cloud tab: `/api/aws/status`.
4. **Results (2:00)** — table from `results/full_metrics_summary.json`; overhead math `1548/118324=1.3083%`; fairness spread; gain-sweep null finding (D-018) stated honestly.
5. **Close (1:30)** — DP (clip+Gaussian, epsilon sweep), Byzantine down-weighting, German generalization, CloudWatch dashboard; cold-boot 68s→0.45s via `models/*.joblib`.

## Top 15 Q&A
1. **Why LightGBM?** Tabular SOTA, native TreeSHAP, 4× faster than XGBoost (D-005).
2. **Why cosine, not Spearman?** Matches §3.3 pseudocode + executed results; magnitude-aware; Spearman future work (D-020).
3. **γ=0 recovers FedAvg?** Yes — `tests/test_consistency_score.py::test_consistency_gain_zero_recovers_fedavg`.
4. **Effect size so small?** Baseline agreement ~0.81–0.89; reweighting bounded <1% (D-018). Honest null finding.
5. **Overhead exact?** `118324 B` weights + `1548 B` SHAP = `+1.3083%` (`full_metrics_summary.json:84-95`).
6. **Non-IID design?** Grade/state/time priority D-001; default rates 10.6/28.7/29.1%.
7. **Soft-label federation?** Tree structures differ; FedDF-style prob averaging (D-011).
8. **DP guarantee?** Clipped Gaussian, `σ=C√(2ln(1.25/δ))/ε`, δ=1e-5, C=1.0 (D-022). SHAP un-noised — stated limitation.
9. **Byzantine detection?** Poisoned SHAP diverges → `a_3` drops → `w_3*` penalized; demo + `src/byzantine_attack.py`.
10. **German generalization?** `src/german_credit.py`, history partition, same 3-variant protocol.
11. **Cold boot?** `models/*.joblib` preload <0.5s, <280MB (D-019); never retrain on boot.
12. **IAM?** Instance profile, zero hardcoded creds; `boto3` optional with graceful offline mode.
13. **SNS threshold?** `default_probability_pct >= 35.0` (percent units, not fraction).
14. **CloudWatch dashboard?** Terraform `aws_cloudwatch_dashboard` + `deploy_cloudwatch_dashboard.py`.
15. **Centralized wins — why federate?** Privacy constraint; centralized is upper bound only, illegal to deploy.
