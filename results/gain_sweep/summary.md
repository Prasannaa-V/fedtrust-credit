# CONSISTENCY_GAIN Hyperparameter Sweep Summary

> **Investigation**: Evaluation of Explanation-Consistency-Aware Federated Aggregation across varying `CONSISTENCY_GAIN` values on Lending Club credit risk data (20 rounds, 3 non-IID clients).  
> **Date**: 2026-09-08  
> **Source**: Real executions logged in `results/fedavg_baseline/`, `results/consistency_aware/`, and `results/gain_sweep/`.

---

## 1. Quantitative Sweep Results

| CONSISTENCY_GAIN | Final Accuracy (Round 20) | Final AUC-ROC (Round 20) | Final F1 Score (Round 20) | Final Consistency (Round 20) | Steady-State Accuracy (Rounds 4-20 Mean) | Steady-State AUC (Rounds 4-20 Mean) | Steady-State F1 (Rounds 4-20 Mean) | Steady-State Consistency (Rounds 4-20 Mean) | Consistency Delta vs. FedAvg (SS) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0 (FedAvg)** | 0.8257 | 0.6565 | 0.1187 | 0.8123 | 0.8257 | 0.6565 | 0.1188 | **0.8114** | — |
| **0.5** | 0.8257 | 0.6565 | 0.1186 | 0.8146 | 0.8257 | 0.6565 | 0.1186 | **0.8120** | +0.0006 (+0.07%) |
| **1.0 (Ours Default)** | 0.8257 | 0.6565 | 0.1186 | 0.8141 | 0.8258 | 0.6566 | 0.1187 | **0.8118** | +0.0004 (+0.05%) |
| **2.0** | 0.8259 | 0.6565 | 0.1188 | 0.8146 | 0.8258 | 0.6565 | 0.1188 | **0.8115** | +0.0001 (+0.01%) |
| **5.0** | 0.8259 | 0.6566 | 0.1188 | 0.8117 | 0.8258 | 0.6565 | 0.1187 | **0.8127** | +0.0013 (+0.16%) |
| **10.0** | 0.8258 | 0.6563 | 0.1187 | 0.8141 | 0.8258 | 0.6564 | 0.1187 | **0.8128** | +0.0014 (+0.17%) |

---

## 2. Aggregation Math Sanity Check (Round 4 Post-Convergence)

To verify whether the weight-adjustment formula was functioning correctly or suppressed by normalization, the exact weights and agreements were inspected at Round 4:

- **Client Sizes**: Client 1 = 502,172 (62.99%), Client 2 = 230,148 (28.87%), Client 3 = 64,848 (8.13%)
- **Base Weights (FedAvg)**: $w_1 = 0.629945$, $w_2 = 0.288707$, $w_3 = 0.081348$
- **Pairwise SHAP Agreements**: $a_1 = 0.800512$, $a_2 = 0.807239$, $a_3 = 0.800482$

### Weight Re-weighting Across Gain Values at Round 4:
- **GAIN = 0.0**: $w_1 = 0.629945$, $w_2 = 0.288707$, $w_3 = 0.081348$ (Delta = 0.000%)
- **GAIN = 0.5**: $w_1 = 0.629509$, $w_2 = 0.289200$, $w_3 = 0.081291$ (Delta: $w_2$ shifted $+0.171\%$)
- **GAIN = 1.0**: $w_1 = 0.629267$, $w_2 = 0.289474$, $w_3 = 0.081259$ (Delta: $w_2$ shifted $+0.266\%$)
- **GAIN = 2.0**: $w_1 = 0.629007$, $w_2 = 0.289768$, $w_3 = 0.081225$ (Delta: $w_2$ shifted $+0.368\%$)
- **GAIN = 5.0**: $w_1 = 0.628726$, $w_2 = 0.290086$, $w_3 = 0.081188$ (Delta: $w_2$ shifted $+0.478\%$)
- **GAIN = 10.0**: $w_1 = 0.628591$, $w_2 = 0.290239$, $w_3 = 0.081170$ (Delta: $w_2$ shifted $+0.531\%$)

---

## 3. Mathematical Analysis & Null-Effect Root Cause

The weight-adjustment mathematics from Section 3.3 is operating **completely correctly as specified in pseudocode**, with no normalization bugs. The negligible effect size is explained by mathematical and empirical factors:

1. **High Inherent Explanatory Agreement ($a_i \approx 0.80$)**:
   The cross-client agreement terms differ by only $\Delta a = 0.8072 - 0.8005 = 0.0067$. Primary credit risk drivers (`int_rate`, `dti`, `annual_inc`, `sub_grade`) remain globally dominant across all partitions.
2. **Mathematical Asymptotic Bound of the Formula**:
   The formula adjusts weights as:
   $$\tilde{w}_i = \frac{w_i (1 + G a_i)}{\sum_k w_k (1 + G a_k)}$$
   When $a_i \approx \bar{a}$, the factor $(1 + G a_i) \approx (1 + G \bar{a})$ is nearly identical across all clients. The normalization denominator $\sum_k w_k (1 + G a_k)$ cancels out this common scaling factor.
   The asymptotic limit as $G \to \infty$ of the relative weight ratio between any two clients is:
   $$\lim_{G \to \infty} \frac{1 + G a_i}{1 + G a_j} = \frac{a_i}{a_j} = \frac{0.8072}{0.8005} \approx 1.0084$$
   **Therefore, even at infinite gain, the relative weight shift between Client 2 and Client 1 is mathematically bounded by at most $+0.84\%$.**
3. **Conclusion**:
   No value of `CONSISTENCY_GAIN` produces a statistically meaningful or practically significant improvement in consistency over FedAvg on this dataset. The consistency score remains tightly bounded within $[0.8114, 0.8128]$ across all 6 experimental runs.
