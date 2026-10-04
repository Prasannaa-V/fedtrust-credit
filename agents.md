# AGENTS.md — FedTrust-Credit Developer & Agent Guide

> **Welcome, AI Agent / Developer!**
> This repository contains **FedTrust-Credit**: A Federated Learning Framework with Explanation-Consistency-Aware Aggregation for Privacy-Preserving Credit Risk Assessment.
> Read this document first to understand the system architecture, documentation index, current implementation state, and the remaining roadmap to reach 100% project completion.

---

## 1. Project Overview & Core Innovation

* **Course**: CLOUD COMPUTING (BITE412L), Fall Semester 2026–27, SCORE, VIT Vellore
* **Team**: Prashaanth Raj J M (`23BIT0173`), Prasannaa V (`23BIT0041`), Haswanth K (`23BIT0359`)
* **Guide**: Dr. Siva Rama Krishnan S (Associate Professor Grade 1, VIT Vellore)
* **GitHub Repository**: [Prasannaa-V/fedtrust-credit](https://github.com/Prasannaa-V/fedtrust-credit)

### The Core Problem & Solution
Financial institutions cannot share raw borrower data due to strict banking regulations (GDPR, Fair Lending, CCPA). However, training credit risk models locally leads to distributional bias and poor generalization.
* **Our Solution**: A **Federated Learning** architecture where 3 simulated banks collaboratively train LightGBM credit risk models across isolated cloud networks without ever sharing raw customer data.
* **Our Novel Contribution**: **Explanation-Consistency-Aware Aggregation**. Instead of standard FedAvg (which weights clients solely on dataset size $n_i$), our central aggregator collects each client's **TreeSHAP explanation vectors** and computes pairwise cosine-similarity consensus (D-020). Clients whose explanations align with consensus receive an aggregation gain factor ($\lambda$), improving fairness and alignment under non-IID conditions without exposing raw data.

---

## 2. Documentation Directory Index (`docs/`)

All project specifications, benchmarks, and architectural decisions are cataloged in `docs/`. When working on any aspect of this repository, consult these documents:

| Document | Path | What It Contains | When to Read |
|---|---|---|---|
| **75% Review Report** | [docs/midterm_review_75pct.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/midterm_review_75pct.md) | **Primary Reference**: System architecture, benchmark results (Accuracy, AUC, Consistency, Fairness), live multi-cloud status, and the remaining 25% implementation specifications. | **Start here** for overall context & benchmarks. |
| **Standalone HTML Review** | [docs/midterm_review_standalone.html](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/midterm_review_standalone.html) | Self-contained, styled HTML version of the review report with embedded diagrams and benchmarks. | For browser presentation or export. |
| **Cloud Deployment Manual** | [docs/CLOUD_DEPLOYMENT.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/CLOUD_DEPLOYMENT.md) | Multi-cloud setup details: AWS EC2, Azure VM, S3 model registry, CloudWatch telemetry, SNS alerts, IAM roles, and Terraform scripts. | When working on cloud infrastructure or deployment. |
| **Architecture Decisions** | [docs/DECISIONS.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/DECISIONS.md) | Full log of architectural decisions (`D-001` through `D-025`), including rationale for LightGBM, cosine-similarity consistency, model serialization, and IAM profiles. | When designing or altering algorithms / pipelines. |
| **Progress Execution Log** | [docs/PROGRESS.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/PROGRESS.md) | Historical execution log tracking real terminal outputs, round-by-round federated training metrics, and test logs. | To verify past experiment numbers and milestones. |
| **Phased Build Plan** | [docs/BUILD_PLAN.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/BUILD_PLAN.md) | The structured build plan and milestone checklist for development. | To track milestone completion. |
| **Initial Review Report** | [docs/initial_review_report.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/docs/initial_review_report.md) | Literature review, mathematical problem formulation, and foundation of the paper. | For theoretical background and paper citations. |
| **Results Summary** | [results/RESULTS_SUMMARY.md](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/results/RESULTS_SUMMARY.md) | Formal empirical comparison between Plain FedAvg, FedTrust-Credit (Ours), and Centralized benchmark. | When analyzing experimental deltas. |

---

## 3. Codebase Architecture & File Map

```
fedtrust-credit/
├── AGENTS.md                     # You are here!
├── pytest.ini                    # Pytest configuration (targets tests/ directory)
├── requirements.txt              # Pinned Python dependencies
├── Dockerfile                    # Container definition (Python 3.11-slim + libgomp1)
├── docker-compose.yml            # Multi-network local topology
├── data/                         # Lending Club partitioned datasets
├── models/                       # Serialized LightGBM models (<0.5s cold boot)
│   ├── client_1.txt              # Bank 1 model (Grade A/B Prime)
│   ├── client_2.txt              # Bank 2 model (Grade C/D Standard)
│   ├── client_3.txt              # Bank 3 model (Grade E/G Subprime)
│   ├── global_model.txt          # Aggregated global credit model
│   └── feature_names.json        # 82-feature aligned space
├── results/                      # Evaluation JSONs, metrics, and publication plots
│   ├── full_metrics_summary.json # Combined execution benchmark metrics
│   └── plots/                    # ROC curves, accuracy, consistency, and fairness charts
├── src/                          # Application source code
│   ├── aggregation_strategy.py   # Explanation-consistency reweighting algorithm
│   ├── consistency_score.py      # Cosine-similarity consistency scoring functions
│   ├── shap_explanation.py       # TreeSHAP feature attribution generator
│   ├── local_training.py         # LightGBM federated client training logic
│   ├── data_partition.py         # Non-IID data partitioner (by Grade, State, Time)
│   ├── evaluation.py             # Accuracy, AUC, F1, and institutional fairness metrics
│   ├── export_models.py          # Trains and persists models to models/ directory
│   ├── fedavg_server.py          # Classical FedAvg baseline simulation server
│   ├── run_experiment.py         # Universal experiment runner (fedavg, ours, centralized)
│   ├── run_ours.py               # Runner for consistency-aware federated rounds
│   ├── run_centralized.py        # Centralized benchmark runner (upper bound)
│   ├── aws_integrations.py       # AWS S3, CloudWatch metric logger, and SNS risk alerts
│   ├── service.py                # FastAPI web service, risk engine, and REST endpoints
│   └── static/                   # Interactive Web Dashboard frontend
│       ├── index.html            # Dashboard UI (KPIs, Persona presets, Risk form, Cloud status)
│       ├── app.js                # Frontend JavaScript (API calls, live evaluations)
│       └── style.css             # Premium responsive banking theme styles
├── terraform/                    # Multi-cloud Infrastructure as Code
│   ├── aws_vpc.tf                # AWS VPCs (Aggregator + Bank 1 + Bank 2)
│   ├── aws_compute.tf            # EC2 instances with Docker cloud-init
│   ├── aws_storage.tf            # S3 bucket for model registry
│   ├── aws_monitoring.tf         # CloudWatch log groups, metric filters, and alarms
│   ├── azure_compute.tf          # Azure Resource Group, VNet, and VM for Bank 3
│   └── azure_storage.tf          # Azure Blob Storage container
└── tests/                        # Automated unit and API test suite
    ├── test_consistency_score.py # Mathematical sanity tests for consistency scoring
    └── test_service_api.py       # FastAPI endpoint tests and risk evaluation verification
```

---

## 4. Live Multi-Cloud Infrastructure

The system is deployed across a live hybrid-cloud topology:
1. **AWS EC2** (`us-east-1`): Hosts the Central Aggregator container and simulated Bank 1 & 2 containers on port 8000.
2. **Azure Linux VM** (`East US`): Hosts simulated Bank 3 (Subprime portfolio) in an isolated Azure VNet.
3. **AWS S3** (`fedtrust-models`): Cloud model registry where trained LightGBM models are uploaded and fetched.
4. **Amazon CloudWatch**: Receives real-time `DefaultProbability` telemetry under namespace `FedTrustCredit/FL`.
5. **Amazon SNS** (`fedtrust-risk-alerts`): Automatically dispatches high-risk borrower email alerts when `default_probability_pct >= 35.0%`.
6. **AWS IAM**: Role-based access via instance profiles; zero credentials hardcoded.

---

## 5. Current Status vs. Remaining Implementation Roadmap

### What is Completed (~75%)
- Non-IID data partitioning on Lending Club (1.34M cleaned loans).
- 20-round federated training simulation across all 3 variants (FedAvg, Ours, Centralized).
- TreeSHAP vector calculation and pairwise cosine-similarity consistency reweighting.
- Full evaluation metrics (Accuracy: 85.34%, AUC: 0.6574, Consistency: 0.8836, +1.31% communication overhead).
- Model serialization reducing cold-boot time from 68.3s to 0.45s (<280MB RAM).
- FastAPI backend and interactive browser dashboard with live cloud status monitoring.
- AWS and Azure cloud deployment with live S3, CloudWatch, and SNS integrations.
- Comprehensive 75% Midterm Review document in `docs/midterm_review_75pct.md`.

---

### What Remains for 100% Completion (The Remaining 25%)

To finish the project at maximum quality, implement the following items:

#### Task 1: Differential Privacy ($\epsilon$-DP) Integration
* **Objective**: Provide formal $(\epsilon, \delta)$-DP guarantees for client model updates.
* **Implementation**:
  - Add gradient/parameter sensitivity clipping (threshold $C$) in [src/local_training.py](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/src/local_training.py).
  - Add calibrated Gaussian noise injection $\mathcal{N}(0, \sigma^2)$ parameterized by privacy budget $\epsilon \in [0.1, 10.0]$ and $\delta = 10^{-5}$.
  - Create runner script `src/run_differential_privacy.py` to sweep $\epsilon$ values and generate the Privacy vs. Utility degradation plot (`results/plots/differential_privacy_tradeoff.png`).

#### Task 2: Adversarial Robustness & Byzantine Client Defense
* **Objective**: Demonstrate that Explanation-Consistency-Aware Aggregation naturally isolates malicious/poisoned clients.
* **Implementation**:
  - Create `src/byzantine_attack.py` to simulate:
    - **Label Flipping Attack**: Bank 3 flips default labels ($1 \to 0$) to deceive the global model.
    - **Weight Poisoning**: Corrupting parameter updates.
  - Show that while FedAvg's accuracy degrades under attack, FedTrust-Credit's TreeSHAP consensus score for the attacker drops, causing the aggregator to down-weight its contribution to near-zero.
  - Generate comparison plot `results/plots/byzantine_robustness.png`.

#### Task 3: Cross-Dataset Generalization (UCI German Credit)
* **Objective**: Validate that the aggregation mechanism generalizes beyond Lending Club.
* **Implementation**:
  - Implement a loader for the UCI German Credit dataset (1,000 cases).
  - Partition across 3 institutions based on credit history.
  - Run the 3-way federated comparison (Centralized vs FedAvg vs FedTrust-Credit) and output `results/plots/german_credit_benchmark.png`.

#### Task 4: AWS CloudWatch Visual Dashboard
* **Objective**: Create a ready-to-view visual dashboard widget grid in the AWS console.
* **Implementation**:
  - Add `aws_cloudwatch_dashboard` resource in [terraform/aws_monitoring.tf](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/terraform/aws_monitoring.tf).
  - Provide helper script `src/deploy_cloudwatch_dashboard.py` (via `boto3`) to deploy or refresh dashboard widgets (DefaultProbability time series, consistency score, request latency, and SNS alarm states).

#### Task 5: Interactive Web Dashboard Upgrades
* **Objective**: Enhance the UI in [src/static/index.html](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/src/static/index.html) and [src/static/app.js](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/src/static/app.js) for live demonstrations:
  - Add a **Byzantine Attack Simulation toggle** in the UI: watch Bank 3's consistency score drop and its weight get penalized in real time.
  - Add a **Differential Privacy slider** for $\epsilon$: observe noise injection on predictions.

#### Task 6: Final 100% Submission Report & Viva Defense Kit
* **Objective**: Prepare academic final submission documents:
  - Create `docs/final_project_report_100pct.md` (and standalone HTML) incorporating all final results (Lending Club + German Credit + Byzantine defense + Differential Privacy).
  - Create `docs/VIVA_PRESENTATION_GUIDE.md`: 10-minute presentation guide and top 15 anticipated professor questions with answers.

---

## 6. Developer & Testing Workflow

### 1. Environment Setup
The project uses Python 3.10+ (tested up to Python 3.14). A virtual environment is pre-configured at `.venv/`.
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Automated Tests
Run pytest from the repository root (configured via `pytest.ini`):
```bash
.venv/bin/pytest
```
*Current status*: **13 passed in ~5.6s**.

### 3. Launching the Local Service & Dashboard
```bash
.venv/bin/uvicorn src.service:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` for the Web Dashboard, or `http://localhost:8000/docs` for the interactive Swagger API documentation.

### 4. Deploying Updates to Cloud Instances
* **AWS EC2 Aggregator** (`13.218.90.189`):
  ```bash
  ssh ubuntu@13.218.90.189
  cd ~/fedtrust-credit && git pull
  sudo docker cp src/service.py fedtrust-dashboard:/app/src/service.py
  sudo docker cp src/static/ fedtrust-dashboard:/app/src/
  sudo docker restart fedtrust-dashboard
  ```
* **Azure VM Bank 3** (`4.247.30.164`):
  ```bash
  ssh azureuser@4.247.30.164
  cd ~/fedtrust-credit && git pull
  sudo docker cp src/service.py fedtrust-azure-dashboard:/app/src/service.py
  sudo docker restart fedtrust-azure-dashboard
  ```

---

## 7. Key Operational Rules for AI Agents

1. **Strictly Grounded Metrics**: Never fabricate benchmark numbers. All metrics reported in docs or summaries must originate from executed code runs in `src/` or `results/`.
2. **Maintain Fast Boot Optimization**: Never re-train models from raw CSV on web service startup. Always load from `models/` when available (sub-second startup is critical for low RAM instances).
3. **Pydantic Model Consistency**: When adding or updating fields in the credit application form ([src/static/index.html](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/src/static/index.html)), ensure the `LoanApplicantRequest` model in [src/service.py](file:///home/prasannaa/Projects/fed%20trust%20credit/fedtrust-credit/src/service.py) matches exactly to avoid silent field dropping.
4. **Percentage Representation**: Remember that `default_probability_pct` is represented as a percentage (e.g., `14.45` for 14.45%). CloudWatch and SNS alerts use this percentage directly without additional multiplication.
