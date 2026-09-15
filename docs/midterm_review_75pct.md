# FedTrust-Credit — 75% Implementation Review

> **Course**: CLOUD COMPUTING – BITE412L, Fall Semester 2026-27
> **School**: SCORE, VIT Vellore
> **Team**:
> - Prashaanth Raj J M — `23BIT0173`
> - Prasannaa V — `23BIT0041`
> - Haswanth K — `23BIT0359`
>
> **Guide**: Dr. Siva Rama Krishnan S, Associate Professor Grade 1, VIT Vellore
> **GitHub**: https://github.com/Prasannaa-V/fedtrust-credit

---

## 1. Project Overview

Financial institutions cannot share raw customer data due to privacy regulations (GDPR, Fair Lending, CCPA), yet an isolated bank's model suffers from distributional bias and poor generalization.

**FedTrust-Credit** addresses this with a **Federated Learning** framework in which three simulated banks — each operating in a private cloud VPC — collaboratively train a credit risk model without ever sharing raw loan records. Our **novel contribution** is the **Explanation-Consistency-Aware Aggregation** mechanism: rather than aggregating model weights uniformly (FedAvg), the central aggregator additionally collects each client's TreeSHAP explanation vectors and re-weights contributions based on pairwise Spearman explanation agreement. This ensures the global model behaves interpretably and consistently across institutions — a prerequisite for regulatory compliance.

---

## 2. Implementation Status

| Module | Status | Notes |
|---|---|---|
| Non-IID data partitioning (3 banks) | ✅ Complete | By loan grade, region, time period |
| Local LightGBM training per client | ✅ Complete | Real Lending Club data (1.34M rows) |
| FedAvg baseline | ✅ Complete | 20-round benchmark |
| Explanation-Consistency-Aware Aggregation | ✅ Complete | Spearman-weighted FedAvg |
| TreeSHAP explainability per client | ✅ Complete | Per-round SHAP vectors |
| 20-round federated benchmark | ✅ Complete | Full metrics logged |
| Centralized baseline (upper bound) | ✅ Complete | Pooled training reference |
| Model serialization (fast cloud boot) | ✅ Complete | 68s → 0.45s boot time |
| FastAPI REST service | ✅ Complete | `/api/predict/risk`, `/api/metrics/*` |
| Docker containerization | ✅ Complete | <400MB image, healthcheck |
| AWS EC2 deployment | ✅ Complete | Live on `us-east-1` |
| **AWS S3** — model registry | ✅ Complete | `fedtrust-models` bucket |
| **AWS CloudWatch** — telemetry | ✅ Complete | `FedTrustCredit/FL` namespace |
| **AWS SNS** — high-risk email alerts | ✅ Complete | Email fires on risk >= 35% |
| **AWS IAM** — instance profile | ✅ Complete | Zero hardcoded credentials |
| Multi-cloud Terraform IaC | ✅ Complete | AWS + Azure, modular |
| GitHub repository | ✅ Complete | Private, all code committed |
| Automated test suite | ✅ Complete | 13/13 tests pass in 5.6s |

---

## 3. System Architecture

Three bank clients, each in an isolated cloud network, train locally on private partitions of the Lending Club dataset. They share only model weights and SHAP explanation vectors with the central aggregator. The aggregator scores explanation alignment across clients and adjusts contribution weights accordingly before broadcasting the updated global model.

```
  Bank 1 (AWS VPC vpc-bank-1)   Bank 2 (AWS VPC vpc-bank-2)   Bank 3 (Azure VNet)
  Grades A-B / Low-risk          Grades C-D / Mid-risk          Grades E-G / Subprime
         |                              |                              |
         |—— weights + SHAP vector ——————|—— weights + SHAP vector ——————|
         v                              v                              v
  +——————————————————————————————————————————————————————————————————————+
  |        Central Aggregator (AWS EC2, us-east-1)                      |
  |  1. Collect weights W1, W2, W3 from all clients                     |
  |  2. Collect SHAP vectors phi1, phi2, phi3                           |
  |  3. Score pairwise Spearman consistency: CS(phi_i, phi_j)           |
  |  4. Compute per-client aggregation weights alpha_i from CS scores   |
  |  5. W_global = sum(alpha_i x W_i)  [Consistency-Aware FedAvg]      |
  |  6. Broadcast W_global to all clients                               |
  +——————————————————————————————————————————————————————————————————————+
         |                   |                   |
    AWS S3           CloudWatch (metrics)    SNS (high-risk alerts)
```

---

## 4. Novel Contribution — Aggregation Algorithm

Standard FedAvg aggregates model weights proportional to local dataset size:

```
W_global = sum( (n_i / N) x W_i )
```

Our **Explanation-Consistency-Aware** variant adjusts each client's contribution using their SHAP vector's Spearman alignment with other clients:

```
CS(i, j)  = SpearmanCorr(|phi_i|, |phi_j|)        # pairwise explanation consistency
score(i)  = mean(CS(i, j)) for all j != i           # per-client consensus score
gain(i)   = 1 + lambda x (score(i) - score_mean)    # consistency gain factor
alpha(i)  = (n_i x gain(i)) / sum(n_j x gain(j))    # normalized aggregation weight
W_global  = sum( alpha(i) x W_i )
```

Clients whose SHAP explanations agree more with the global consensus receive proportionally higher weight — directly incentivizing explainability alignment without any raw data sharing.

---

## 5. Dataset and Partitioning

| Property | Value |
|---|---|
| Source | LendingClub accepted loans 2007–2018 (public) |
| Raw rows | ~2.26M records |
| After cleaning | 1,344,976 rows |
| Target | `loan_status` → binary (Fully Paid = 0, Default = 1) |
| Features | 26 engineered features (rate, DTI, grade, employment, utilization, etc.) |

**Non-IID Partitioning by Loan Grade:**

| Client | Grade Range | Profile | Default Rate |
|---|---|---|---|
| Bank 1 (Client 1) | A–B | Prime / Low-risk | ~8% |
| Bank 2 (Client 2) | C–D | Standard / Mid-risk | ~17% |
| Bank 3 (Client 3) | E–G | Subprime / High-risk | ~32% |

This creates genuine non-IID conditions: each client's local model sees a systematically different population, causing explanation drift between banks — which our method corrects.

---

## 6. Benchmark Results (20 Federated Rounds)

### 6.1 Pooled Accuracy

| Method | Accuracy | AUC-ROC | F1 |
|---|---|---|---|
| Centralized (upper bound) | **85.64%** | **0.7093** | **0.0668** |
| FedTrust-Credit (Ours) | **85.34%** | **0.6574** | 0.0665 |
| FedAvg (baseline) | 85.28% | 0.6554 | 0.0661 |

Our method outperforms FedAvg on accuracy (+0.06%) and AUC-ROC (+0.002) while preserving full data privacy.

### 6.2 Per-Client Accuracy

| Client | FedAvg | Ours | Centralized |
|---|---|---|---|
| Client 1 (Bank A–B) | 90.65% | **90.68%** | 90.75% |
| Client 2 (Bank C–D) | 78.97% | **79.03%** | 79.37% |
| Client 3 (Bank E–G) | 76.30% | **76.49%** | 77.40% |

All three clients improve under our method vs. FedAvg.

### 6.3 Communication Overhead

| Method | Bytes/Round | Overhead vs FedAvg |
|---|---|---|
| FedAvg | 118,324 B | baseline |
| Ours | 119,872 B | **+1.31%** |

Adding SHAP vectors (1,548 B/round) costs only **1.31% extra bandwidth** — negligible in real networks.

### 6.4 Fairness (Accuracy Spread Across Clients)

| Method | Accuracy Spread | F1 Spread |
|---|---|---|
| FedAvg | 0.1436 | 0.1165 |
| Ours | **0.1418** | 0.1260 |
| Centralized | 0.1334 | 0.0894 |

Our method reduces accuracy disparity across banks vs. FedAvg — a fairer outcome for smaller/minority-population banks.

---

## 7. Cloud Infrastructure Deployed

### 7.1 AWS Services (All Live and Verified)

| Service | Role | Status |
|---|---|---|
| **EC2** (us-east-1) | Central aggregator, Docker host for FastAPI service | ✅ Running |
| **S3** `fedtrust-models` | Model registry — stores serialized LightGBM + SHAP artifacts | ✅ Uploaded |
| **CloudWatch** `FedTrustCredit/FL` | Real-time telemetry: DefaultProbability metric per prediction | ✅ Logging |
| **SNS** `fedtrust-risk-alerts` | Email alert when borrower default probability >= 35% | ✅ Firing |
| **IAM** EC2 Instance Profile | Zero hardcoded credentials; role-based least-privilege access | ✅ Active |

### 7.2 Cloud Boot Optimization (Model Serialization)

Before serialization, the service loaded the 1.6 GB raw CSV on every startup:

| Metric | Before | After |
|---|---|---|
| Cold boot time | 68.3 s | **0.45 s** |
| RAM footprint | ~2.5 GB | **<280 MB** |
| Test suite time | 64.85 s | **5.61 s** |

This made deployment on AWS `t2.micro` (1 GB RAM) feasible.

### 7.3 Multi-Cloud Terraform IaC

Modular Terraform configs in `terraform/`:

| File | What it provisions |
|---|---|
| `aws_vpc.tf` | Aggregator VPC + isolated Bank 1 VPC + Bank 2 VPC |
| `aws_compute.tf` | EC2 Aggregator + Bank 1 + Bank 2 with Docker cloud-init |
| `aws_storage.tf` | Encrypted S3 buckets (SSE-AES256), IAM roles, instance profiles |
| `aws_monitoring.tf` | CloudWatch log groups, consistency drift alarms, latency alarms |
| `azure_compute.tf` | Azure Resource Group, VNet (`vnet-bank-3`), NSG, Linux VM (Bank 3) |
| `azure_storage.tf` | Azure Blob Storage container (`bank3data`) |
| `outputs.tf` | EC2 IPs, dashboard URLs, bucket names |

---

## 8. Live REST API Endpoints

Base URL: `http://<EC2-Public-IP>:8000` | API Docs: `.../docs`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Container health check |
| GET | `/api/metrics/summary` | Full benchmark results (all 3 methods) |
| GET | `/api/metrics/rounds` | Round-by-round training trajectory |
| GET | `/api/metrics/partitions` | Non-IID partition statistics |
| POST | `/api/federated/simulate-round` | Live aggregation math demonstration |
| POST | `/api/predict/risk` | Credit risk scoring + SHAP + 3-bank consensus |
| GET | `/api/aws/status` | Live AWS service health |
| POST | `/api/aws/s3/upload` | Upload models to S3 |

**Sample prediction response (real output):**

```json
{
  "default_probability_pct": 39.16,
  "risk_tier": "High (Subprime)",
  "decision": "Decline Recommended (High Default Risk)",
  "explanation_consistency": 0.8318,
  "feature_attributions": [
    {"feature": "int_rate", "shap_value": 0.600, "impact": "Increases Risk"},
    {"feature": "revol_util", "shap_value": 0.144, "impact": "Increases Risk"}
  ],
  "client_perspectives": {
    "client_1": {"default_probability": 31.7},
    "client_2": {"default_probability": 35.07},
    "client_3": {"default_probability": 41.19}
  }
}
```

---

## 9. Test Suite

```
tests/test_consistency_score.py .....     [ 38%]
tests/test_service_api.py ........        [100%]
=================== 13 passed in 5.61s ===================
```

Tests cover: consistency score, cosine similarity, per-client agreement, all API routes, health check, prediction format, and metrics endpoints.

---

## 10. Remaining Work (25%)

| Task | Description |
|---|---|
| Azure VM (Bank 3) live deployment | Currently simulated via docker-compose; Terraform IaC is ready |
| CloudWatch visual dashboard | Build monitoring dashboard panel for demo |
| Final academic report | Full methodology, results tables, camera-ready figures |
| Conference-quality plots | ROC curves, consistency trajectory, gain sweep results |
| Viva / presentation preparation | Live demo script and Q&A preparation |

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| LightGBM over XGBoost/RF | 4x faster training on tabular data; native SHAP TreeExplainer support |
| Spearman (not Pearson) for consistency | Rank correlation is robust to SHAP magnitude differences across non-IID clients |
| Serialize models to disk | Eliminates 1.6 GB CSV reload on cloud startup; fits in 512 MB free tier |
| IAM Instance Profile over access keys | Zero secret rotation risk; no hardcoded credentials in Docker image |
| boto3 optional import in service | Service starts and serves predictions even if AWS unavailable; graceful degradation |

---

## 12. References

1. "Federated Learning Architectures for Credit Risk Assessment," IEEE, 2025.
2. "Federated Learning for Credit Risk Assessment," HICSS, 2023.
3. S. Zhang et al., "Effects of Data Imbalance Under Federated Learning for Credit Risk," arXiv:2401.07234, 2024.
4. "Federated SHAP: Privacy-Preserving and Consistent Explainability," Springer ML, 2025.
5. A. Bogdanova et al., "DC-SHAP for Consistent Explainability in Distributed ML," Human-Centric Intelligent Systems, 2023.
6. H. Y. Wong et al., "Stratify: Rethinking Federated Learning for Non-IID Data," arXiv:2504.13462, 2025.
7. "Interpretable AI in Credit Scoring: Comparative Survey of SHAP, LIME," R Discovery, 2025.
8. "A Privacy-Preserving Cloud Architecture for Distributed Machine Learning," arXiv:2512.10341, 2025.
