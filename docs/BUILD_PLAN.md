# FedTrust-Credit — Build Plan (Compressed: 3-Day Schedule)

> **Project**: FedTrust-Credit: A Federated Learning Framework with Explanation-Consistency-Aware Aggregation for Privacy-Preserving Credit Risk Assessment
> **Team**: Prashaanth Raj J M (23BIT0173) · Prasannaa V (23BIT0041) · Haswanth K (23BIT0359)
> **Course**: CLOUD COMPUTING – BITE412L, Fall Semester 2026-27
> **Guide**: Dr. Siva Rama Krishnan S
>
> **Schedule compressed 2026-09-08**: Original 10-day plan → 3 days (6-8 hrs/day).
> Cloud deployment (original Day 10) explicitly deferred — document as "designed, not yet deployed."

---

## ✅ COMPLETED — Data Loading & Non-IID Partitioning

- [x] Download Lending Club dataset (1.6 GB, 2,260,701 raw rows)
- [x] Clean to 1,344,976 rows (80.0% repaid / 20.0% default)
- [x] Non-IID partition by grade / addr_state / issue_d:
  - [x] **Client 1 (Grade A/B)**: 627,716 rows | 10.6% default
  - [x] **Client 2 (East Coast states)**: 287,686 rows | 28.7% default
  - [x] **Client 3 (2016-12 to 2018-12)**: 81,060 rows | 29.1% default
- [x] Sanity tests (consistency_score): 4/5 PASSED (5th deferred — needs flwr)
- [x] `results/partition_stats.json` saved

---

## Day 1 (2026-09-08) — Flower Federation + FedAvg Baseline

> **Gate**: FedAvg simulation must run end-to-end and show real per-client training output before Day 2 begins.

### Environment
- [x] Install `flwr==1.9.0` and `shap==0.49.1` in venv
- [x] Verify install: protobuf 5.29.5 + pure-Python fallback (D-014)

### Flower Client (`src/local_training.py`)
- [x] `FederatedClient` (pure Python, no flwr import) with LightGBM (D-005, D-011)
  - [x] `get_parameters()` — predicted probability vector as parameters
  - [x] `set_parameters()` — receive global soft-label signal
  - [x] `fit()` — local LightGBM training with soft-label blending, return weights + metrics
  - [x] `evaluate()` — accuracy, AUC-ROC, F1 on local test split
- [x] `build_shared_feature_columns()` + `feature_cols` alignment added to `data_partition.py` (D-015)

### FedAvg Server (`src/fedavg_server.py`)
- [x] Custom in-process simulation loop in `src/fedavg_server.py` (no Ray, D-014)
- [x] 20 rounds × 3 clients, all clients every round
- [x] Per-round JSON logs saved to `results/fedavg_baseline/`
- [x] **GATE PASSED** ✅ — 20 rounds complete, real per-client output confirmed
  - Final: acc=0.8257 | auc=0.6565 | f1=0.1187 | consistency=0.8123
  - FedAvg baseline consistency mean (rounds 4-20): **0.812**

---

## Day 2 (2026-09-09) — SHAP + Consistency-Aware Aggregation + Centralized Baseline

> **Priority**: Correctness over speed. This is the novel contribution — no approximations.

### SHAP Generation (`src/shap_explanation.py`) — completed
- [x] Verify `compute_shap_vector()` works with LightGBM TreeExplainer
- [x] Wire into `FlowerClient.fit()` / `FederatedClient.fit()` — SHAP vector returned in metrics dict

### Sanity Tests (`tests/test_consistency_score.py`) — 5/5 PASSED
- [x] Run all 5 tests with flwr installed (5th test: CONSISTENCY_GAIN=0 → FedAvg)
- [x] **All 5 PASSED before wiring SHAP into aggregation**

### Consistency-Aware Aggregation (`src/aggregation_strategy.py`) — completed
- [x] Verify `ExplanationConsistencyStrategy` and `ExplanationConsistencyAggregator` integrate with real round data
- [x] Implement `src/run_experiment.py` to run all 3 variants from one script:
  - [x] variant `fedavg` → CONSISTENCY_GAIN=0 (recovers fedavg_server.py results exactly)
  - [x] variant `ours` → CONSISTENCY_GAIN=1.0
  - [x] variant `centralized` → pooled data, no federation
- [x] Log per-round: global_consistency, weights (bytes), shap_bytes for overhead calc

### Centralized Baseline (`src/centralized_baseline.py`) — completed
- [x] Run on pooled train data, evaluate on same held-out splits
- [x] Compute explanation-consistency by partitioning SHAP along client lines (Sec 4.3)
- [x] Save to `results/centralized_baseline/centralized_results.json`

### Gate Check
- [x] **GATE PASSED** ✅ — All 3 variants produce output, consistency-aware results logged

---

## Day 3 (2026-09-10) — Full Evaluation + Results + Graphs

### Run All Variants (`src/run_experiment.py`)
- [x] 20 federated rounds, identical data split for all 3 variants
- [x] All metrics from Section 4.3 recorded (real execution only):
  - [x] Per-client: accuracy, AUC-ROC, F1
  - [x] Pooled test set: accuracy, AUC-ROC, F1
  - [x] Explanation-consistency score per round
  - [x] Communication overhead: bytes/round (weights + SHAP vs weights only)
  - [x] Institutional fairness spread: error rate spread across 3 clients

### Plots (`results/plots/`)
- [x] Explanation-consistency score over rounds (3 variants) — `consistency_comparison.png`
- [x] Pooled accuracy over rounds (3 variants) — `accuracy_comparison.png`
- [x] Pooled AUC-ROC over rounds (3 variants) — `auc_comparison.png`
- [x] Per-client F1 comparison bar chart — `per_client_f1.png`
- [x] Communication overhead comparison — `communication_overhead.png`
- [x] Fairness spread comparison — `fairness_spread.png`

### Documentation
- [x] Update `PROGRESS.md` with all real Day 3 numbers
- [x] Write `results/RESULTS_SUMMARY.md` — what was actually found (no fabricated text)

### Cloud Architecture Note (Deferred)
- [x] Add note to `DECISIONS.md`: cloud deployment (AWS EC2, Azure VM, IAM, S3, CloudWatch) is **designed, not yet deployed** (D-016) — deliberate scope cut to protect the algorithmic core within deadline

---

## Non-Negotiable Rules (unchanged)

1. **Follow the report exactly**: Dataset schema (Sec 3.4), tool stack (Sec 3.5), aggregation pseudocode (Sec 3.3) are fixed.
2. **No fabricated numbers**: Every metric in `/results/` comes from code execution only.
3. **Log every unspecified decision** in `DECISIONS.md` with a one-line reason.
4. **Sanity tests first**: All 5 tests in `tests/test_consistency_score.py` must pass before SHAP enters aggregation.
5. **PROGRESS.md** updated at the end of every day with what actually ran and produced output.
6. **Cloud deployment is deferred**: Do not attempt AWS/Azure in these 3 days. Document as designed-not-deployed.
