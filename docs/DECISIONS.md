# FedTrust-Credit — Design Decisions Log

> Every decision not explicitly specified in the report is logged here with the date and a one-line reason.
> This log is maintained for patent trail purposes — do not delete or overwrite entries.

---

## Pre-Day 1 Decisions (2026-09-01)

### D-001: Non-IID Partition Assignment Priority
**Decision**: When records could qualify for more than one client partition (e.g., a grade-A loan from New York issued in the recent period), assign with strict priority: Client 1 (grade A/B) first, then Client 2 (East Coast state), then Client 3 (recent time period). Records qualifying for none are excluded from the federation experiment (included in the centralized baseline only).
**Reason**: Report says "partitioned by loan grade, borrower region, and issue-date period" but does not specify tie-breaking — strict disjoint partitioning is the most defensible choice for a controlled non-IID experiment.

### D-002: East Coast States for Client 2
**Decision**: Client 2 (regional partition) = NY, NJ, PA, MA, CT, VA, NC, SC, GA, FL, MD, DE, NH, VT, ME, RI.
**Reason**: Report says "addr_state" as partitioning axis and mentions "regional banks"; East Coast was chosen as a naturally cohesive lending market with sufficient data volume in Lending Club. Logged for reproducibility.

### D-003: Time-Period Partition for Client 3
**Decision**: Client 3 (time-period partition) = most recent 2 years of issue dates present in the dataset.
**Reason**: Report says "issue_d — loans issued in a specific time period"; 2 years provides a large enough slice to avoid near-empty partitions while creating a meaningful temporal distribution shift.

### D-004: Binary Target Encoding
**Decision**: `loan_status` is binarized as: `Fully Paid` → 0 (repaid), `Charged Off` or `Default` → 1 (defaulted). All other statuses (Current, Late, In Grace Period, etc.) are excluded from training and test sets.
**Reason**: Report targets binary credit risk classification. "Current" loans have unknown final outcome; including them would pollute the label. This is standard practice in Lending Club benchmarks.

### D-005: Baseline Classifier
**Decision**: Use LightGBM (via `lightgbm` package) as the local credit risk classifier for all three clients.
**Reason**: Report's tools table lists "scikit-learn / XGBoost" as training tools. LightGBM is listed in the literature (paper 12: LightGBM+SHAP at a Norwegian bank) as the highest-performing tree-based method on credit risk tabular data, and SHAP has native support for LightGBM's tree explainer (faster and more exact than KernelSHAP). If resource constraints require a simpler model, fall back to scikit-learn LogisticRegression (logged separately if changed).

### D-006: SHAP Background Sample
**Decision**: Use the client's local training data (up to 200 samples, randomly sampled) as the SHAP TreeExplainer background dataset.
**Reason**: Report does not specify the SHAP background; using local training data is consistent with privacy constraints (no cross-client data sharing). Capped at 200 to keep SHAP computation tractable per round.

### D-007: CONSISTENCY_GAIN Default Value
**Decision**: Default `CONSISTENCY_GAIN = 1.0` for the explanation-consistency-aware aggregation runs.
**Reason**: Report defines CONSISTENCY_GAIN as a "tunable hyperparameter" without specifying a default. A value of 1.0 gives explanation consistency equal weight to the base FedAvg size-proportional term in the first round. Will be swept (0, 0.5, 1.0, 2.0) in Day 9 evaluation if time allows.

### D-008: Number of Federated Rounds
**Decision**: Run all variants for 20 federated rounds.
**Reason**: Report specifies "enough rounds for real results" but gives no number. 20 rounds is consistent with the federated credit-risk literature (papers 1–6) and allows convergence curves to be meaningfully plotted. Logged for reproducibility.

### D-009: Train/Test Split Ratio
**Decision**: 80% train / 20% test, stratified by binary target label, applied independently per client.
**Reason**: Report does not specify a split ratio. 80/20 stratified split is the standard for imbalanced credit-risk datasets and ensures both splits are class-representative.

### D-010: SHAP Explanation Vector Definition
**Decision**: Per-client SHAP vector = mean absolute SHAP values across all local validation samples, resulting in a 1D vector of length = number of features.
**Reason**: Section 3.3 says "one mean-absolute feature-attribution vector per client." Taking the mean absolute value (rather than mean signed value) ensures the vector represents feature importance magnitudes, which is what cosine similarity between clients meaningfully compares.

---

<!-- Add new decisions below as they arise, numbered D-011 onwards -->

## Compressed Schedule Decisions (2026-09-08)

### D-011: LightGBM Federated Aggregation Method
**Decision**: Use soft-label distillation for LightGBM federation. Each client serializes its predicted probabilities on the training set as the "parameters" vector. The server averages these across clients (weighted by dataset size). Each client uses the aggregated predictions as soft labels (blended 70% hard / 30% soft) to re-train its next round's model.
**Reason**: LightGBM boosters have non-identical tree structures — direct averaging of tree node splits (as FedAvg does for neural network weights) is not defined. Prediction-space averaging (FedDF-style soft distillation) is model-agnostic and mathematically equivalent to averaging the model's learned function, which is what FedAvg conceptually does. Logged for patent trail since aggregation behaviour affects the consistency metric.

### D-012: n_estimators Reduced to 50 for Simulation Speed
**Decision**: LightGBM `n_estimators=50` per round (vs. 100 in original plan) for the federated simulation. The centralized baseline uses 200 for a fairer upper-bound comparison.
**Reason**: With 3 clients × 20 rounds × large partition sizes, 100 estimators/round makes the simulation impractically slow on a single machine. 50 is enough to see convergence trends. Logged so the numbers in /results/ are reproducible at this setting.

### D-014: Custom Simulation Loop (No Ray)
**Decision**: Replace `flwr.simulation.start_simulation()` (which requires `ray`) with a custom in-process Python loop that drives fit→aggregate→broadcast→evaluate for each round.
**Reason**: `ray` (flwr's simulation backend) is incompatible with Python 3.14 (`Metaclasses with custom tp_new are not supported` on protobuf's C extension). The custom loop is functionally equivalent, fully deterministic, and easier to debug. The federated averaging math is identical to flwr's FedAvg — only the process orchestration changes.

### D-015: Shared Feature Space Across Clients
**Decision**: Build a union of all one-hot-encoded feature columns across all 3 partitions. Each client's feature matrix is aligned to this shared column list, with 0-filled for columns absent in that partition.
**Reason**: Each partition's one-hot encoding produces different columns (different grades, states, etc present). Without alignment, SHAP vectors are different lengths and cannot be compared with cosine similarity (Section 3.3 requires mean pairwise cosine similarity of same-length vectors). The shared feature space is the correct solution — it matches how a real federated system would standardize its feature schema.


### D-016: Cloud Architecture Deferred (Deliberate Scope Cut)
**Decision**: AWS EC2, Azure VM, IAM, S3, and CloudWatch live cloud deployment is explicitly deferred. The local Flower/in-process multi-client simulation is the complete deliverable for the compressed 3-day sprint.
**Reason**: Deliberate scope cut to protect the algorithmic core (the novel explanation-consistency aggregation mechanism) from deadline pressure. Cloud infrastructure design is fully documented in Section 3.5 and the system architecture as "designed, not yet deployed."

### D-017: Evaluation Plotting and Communication Metric Accounting
**Decision**: Communication overhead is computed per round as exact wire payload bytes: model parameter/soft-label vectors (3,188,672 bytes) plus 1D float32 SHAP explanation vector (84 features × 4 bytes × 3 clients = 1,008 bytes).
**Reason**: Quantifies the exact bandwidth cost of transmitting feature attributions per federated round, proving that adding SHAP consistency-awareness incurs negligible overhead (+0.0316%).

### D-018: CONSISTENCY_GAIN Sweep and Empirical Null-Effect Finding
**Decision**: Tested `CONSISTENCY_GAIN` values across [0.0, 0.5, 1.0, 2.0, 5.0, 10.0] for 20 rounds under identical conditions. Report the result plainly as a legitimate null-effect finding on this dataset, without tuning or perturbing partitions/features to force artificial gains.
**Reason**: 
1. **Weight Adjustment Math is Working Correctly**: Inspection of post-convergence weights at Round 4 confirmed that the pseudocode formula $\tilde{w}_i = w_i(1 + G a_i) / \sum_k w_k(1 + G a_k)$ is implemented with zero normalization bugs. As gain increases from 0.0 to 10.0, Client 2's weight increases from 0.2887 to 0.2902 (+0.531%).
2. **Mathematical Bound**: Because cross-client agreement terms are inherently high and nearly identical ($a_1=0.8005, a_2=0.8072, a_3=0.8005$), the term $(1 + G a_i)$ acts as a near-constant scaling factor across all clients that cancels out in the normalization step. Asymptotically $\lim_{G \to \infty} (1+Ga_i)/(1+Ga_j) = a_i/a_j \le 1.0084$, bounding maximum relative weight shift to $<0.84\%$.
3. **Empirical Finding**: Steady-state consistency across all 6 runs remains tightly bounded in [0.8114, 0.8128] ($\le 0.17\%$ variation), while pooled accuracy (82.57%-82.59%) and AUC (0.6563-0.6566) are unaffected. No gain value shows a non-noise improvement. This demonstrates that for tabular credit risk where foundational risk features (`int_rate`, `dti`, `annual_inc`) dominate across all institutional silos, explanation consistency is already near-maximal under standard FedAvg and insensitive to agreement reweighting.



