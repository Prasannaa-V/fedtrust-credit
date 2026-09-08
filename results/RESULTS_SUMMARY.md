# FedTrust-Credit — Results Summary

> **Project**: FedTrust-Credit: A Federated Learning Framework with Explanation-Consistency-Aware Aggregation for Privacy-Preserving Credit Risk Assessment  
> **Course**: CLOUD COMPUTING – BITE412L, Fall Semester 2026-27  
> **Team**: Prashaanth Raj J M (23BIT0173) · Prasannaa V (23BIT0041) · Haswanth K (23BIT0359)  
> **Guide**: Dr. Siva Rama Krishnan S  
> **Generated**: 2026-09-08 (Execution from 20-round simulations on 1.34M Lending Club records)

---

## 1. Executive Summary

This report documents the empirical results comparing three credit risk assessment paradigms executed on identical non-IID partitioned Lending Club data (1,344,976 cleaned loan records across 3 simulated institutional clients):

1. **FedAvg Baseline**: Standard federated learning with size-weighted averaging of model parameter/soft-label vectors (`CONSISTENCY_GAIN = 0.0`).
2. **FedTrust-Credit (Ours)**: Novel federated aggregation that re-weights client updates by their pairwise SHAP explanation agreement (`CONSISTENCY_GAIN = 1.0`, Section 3.3).
3. **Centralized Benchmark**: Single model trained on all pooled client training data (upper-bound reference, no federation).

All reported numbers are strictly derived from executed code runs (`results/fedavg_baseline/`, `results/consistency_aware/`, `results/centralized_baseline/centralized_results.json`, and `results/full_metrics_summary.json`).

---

## 2. Quantitative Comparison Table

| Metric | FedAvg Baseline | FedTrust-Credit (Ours) | Centralized Benchmark | Delta (Ours vs. FedAvg) |
|---|:---:|:---:|:---:|:---:|
| **Pooled Accuracy** | **0.8257** | **0.8257** | 0.8284 | -0.00006 |
| **Pooled AUC-ROC** | **0.6565** | **0.6565** | 0.7225 | +0.00007 |
| **Pooled F1 Score** | **0.1187** | **0.1186** | 0.0943 | -0.00016 |
| **Explanation Consistency (Final Round)** | **0.8123** | **0.8141** | 0.7620 | **+0.0018** |
| **Explanation Consistency (Steady-State Mean)** | **0.8114** | **0.8118** | 0.7620 | **+0.0004** |
| **Parameter Payload (Bytes / Round)** | 3,188,672 B | 3,188,672 B | N/A (local) | 0 B |
| **SHAP Payload (Bytes / Round)** | 0 B | **1,008 B** | N/A | +1,008 B |
| **Total Bandwidth per Round** | 3.041 MB | 3.042 MB | N/A | **+0.0316%** |
| **Institutional Accuracy Spread** | 18.81% | 18.87% | 17.83% | +0.06% |
| **Institutional F1 Spread** | 32.24% | 32.10% | 15.36% | -0.14% |

---

## 3. Per-Client Performance Breakdown

Institutional partitioning creates genuine non-IID demographic and temporal heterogeneity across institutions:
- **Client 1 (Prime Portfolio: Grades A/B)**: 627,716 records, default rate 10.6%
- **Client 2 (Regional Bank: East Coast States)**: 287,686 records, default rate 28.7%
- **Client 3 (Fintech / Recent Vintage: 2016–2018)**: 81,060 records, default rate 29.1%

### Test Set Performance by Client

| Client | Metric | FedAvg Baseline | FedTrust-Credit (Ours) | Centralized Benchmark |
|---|---|:---:|:---:|:---:|
| **Client 1** (Grade A/B) | Accuracy<br>AUC-ROC<br>F1 Score | 0.8936<br>0.6600<br>0.0004 | 0.8936<br>0.6600<br>0.0004 | 0.8936<br>0.6605<br>0.0004 |
| **Client 2** (East Coast) | Accuracy<br>AUC-ROC<br>F1 Score | 0.7114<br>0.6485<br>0.3193 | 0.7114<br>0.6485<br>0.3191 | 0.7179<br>0.6484<br>0.1434 |
| **Client 3** (Recent Vintage) | Accuracy<br>AUC-ROC<br>F1 Score | 0.7056<br>0.6574<br>0.3229 | 0.7050<br>0.6582<br>0.3215 | 0.7154<br>0.6560<br>0.1540 |

---

## 4. Key Findings and Scientific Insights

### 4.1 Explanation Consistency Under Non-IID Credit Data
- **High Inherent Agreement**: On the Lending Club benchmark, both federated systems exhibit a baseline explanation consistency of ~0.81 (measured as mean pairwise cosine similarity between clients' mean absolute SHAP vectors). Primary credit risk drivers (`int_rate`, `dti`, `annual_inc`, `sub_grade`) remain globally dominant predictors of default across loan grades and regions.
- **Superior Consistency over Centralized Benchmark**: Centralized training produced an explanation consistency score of **0.7620** when partitioned across the 3 client test splits, whereas federated aggregation achieved **0.8141**. Federated multi-client consensus encourages individual clients to arrive at more mutually consistent explanatory weights.

### 4.2 Minimal Communication Overhead (+0.0316%)
- Model parameters (soft-label probability distribution vectors) require 3,188,672 bytes per round.
- Transmitting the 1D mean absolute SHAP attribution vector (84 shared features × 4 bytes per float32 = 336 bytes per client, totaling 1,008 bytes for all 3 clients) adds just **0.984 KB per round**.
- The resulting overhead is **+0.0316%**, proving that incorporating explanation consistency into federated aggregation is practical even over bandwidth-constrained banking WAN links.

### 4.3 Convergence and Stability
- Both federated variants converge within 4 rounds and remain exceptionally stable through Round 20.
- Re-weighting with `CONSISTENCY_GAIN = 1.0` preserves the predictive power of FedAvg (pooled accuracy 82.57%, AUC 0.6565) without any degradation in convergence speed or classification utility.

### 4.4 Institutional Fairness Spread
- The accuracy spread across clients is ~18.8% (Client 1: 89.4% vs. Client 3: 70.5%), which directly reflects the base default rate disparity (10.6% vs 29.1%).
- FedTrust-Credit slightly narrows the institutional F1 spread (32.10% vs 32.24% in FedAvg).

---

## 5. Generated Artifacts and Visualizations

All publication-quality plots have been saved to `results/plots/`:
1. `results/plots/consistency_comparison.png` — Explanation-consistency trajectory across 20 rounds compared to centralized benchmark.
2. `results/plots/accuracy_comparison.png` — Pooled accuracy convergence across 20 federated rounds.
3. `results/plots/auc_comparison.png` — Pooled AUC-ROC curves over 20 rounds.
4. `results/plots/per_client_f1.png` — Per-client F1 score comparisons by institutional partition.
5. `results/plots/communication_overhead.png` — Payload breakdown showing the minimal +0.0316% SHAP bandwidth footprint.
6. `results/plots/fairness_spread.png` — Institutional accuracy and F1 disparities across the three paradigms.

---

## 6. Cloud Deployment Status

Per the project scope adjustment (logged in `docs/DECISIONS.md` under **D-016**):
- Live multi-node cloud deployment to AWS EC2, Azure VM, AWS IAM, S3, and CloudWatch was **deliberately deferred** in this sprint.
- The complete multi-client federated simulation is fully implemented and validated locally.
- The cloud architecture remains fully designed as specified in Section 3.5 and is designated as **"designed, not yet deployed"**.
