School of Computer Science Engineering and Information Systems
Department of Information Technology
Fall Semester 2026-27
CLOUD COMPUTING – BITE412L
PROJECT REPORT
FEDTRUST-CREDIT
A Federated Learning Framework with Explanation-Consistency-Aware
Aggregation
for Privacy-Preserving Credit Risk Assessment
Submitted by
Prashaanth Raj J M – 23BIT0173
Prasannaa V – 23BIT0041
Haswanth K – 23BIT0359
Under the Guidance of
Siva Rama Krishnan S
Associate Professor Grade 1
VIT, Vellore.
Fall Semester 2026-27
1
TABLE OF CONTENTS
S.No. Contents Page No.
1 Abstract 3
2 Chapter 1 : Introduction 4
3 Chapter 2 : Literature Survey 6
4 Chapter 3 : Proposed Technique 11
5 Chapter 4 : Results and Discussion 15
6 Chapter 5 : Conclusion & Future Work 17
7 References 19
2
ABSTRACT
Financial institutions increasingly rely on machine learning for credit risk assessment, but regulatory
and competitive constraints prevent them from sharing raw customer data with one another. Federated
Learning (FL) addresses this by enabling collaborative model training without centralizing sensitive
data; however, when client data distributions are non-identical (non-IID), the explanations generated for
model decisions can diverge significantly across institutions, undermining the regulatory expectation
that credit decisions be consistently and reliably explainable. This project proposes FedTrust-Credit, a
cloud-deployed federated learning framework for credit risk assessment that introduces an explanation-
consistency-aware aggregation mechanism. Each simulated bank client trains a local credit risk model
on its own private data partition and generates local SHAP-based explanations; a central aggregator
combines model updates using a modified Federated Averaging strategy that jointly considers
prediction accuracy and cross-client explanation consistency. The system is deployed across isolated
AWS EC2/VPC environments and an Azure Virtual Machine to validate multi-cloud interoperability,
using the publicly available Lending Club Loan Dataset for evaluation. The framework is evaluated
against baseline FedAvg and centralized training on accuracy, fairness, communication overhead, and a
proposed explanation-consistency metric, with the goal of demonstrating that trustworthy, regulator-
aligned credit risk explanations can be preserved even under strict data-privacy constraints.
3
CHAPTER 1: INTRODUCTION
A bank's credit model is only as trustworthy as the reasons it can give for turning someone down.
Regulators and internal risk-governance teams increasingly expect not just an accurate risk score but a
defensible explanation behind it — which factors drove the decision, and why. That expectation runs
into a separate, unrelated constraint: banks generally cannot pool their customer data to train a shared
model, whether for competitive reasons or because doing so would breach data-protection obligations.
Federated learning solves the second problem by keeping raw data on-premises and sharing only model
updates. What it does not automatically solve is the first problem — when each bank's customers look
statistically different from the next bank's, the explanations each local model produces can quietly drift
apart even while the shared global model converges just fine. This project asks a narrower, more
practical question than “does federated learning work for credit risk”: whether the explanations a
federated credit model produces can be kept consistent across institutions, and whether that consistency
can be built into how the model is aggregated rather than checked afterward.
1.1 Motivation
Two things pushed us toward this particular project.
● The explainability gap. When a lender turns an applicant down, it is generally expected to be able
to point to specific, stable reasons — income, utilisation, credit history — rather than an
unexplained score. That expectation was built around centrally trained models. Nobody has really
stress-tested what happens to it once the model training itself is split across institutions that each
see a different slice of the customer population.
● The federated-explainability gap in the literature. Our own literature survey (Chapter 2) found
federated learning work on credit risk that measures accuracy but not explanation behaviour, and
federated-explainability work that treats consistency as a post-hoc analysis step rather than
something the aggregation algorithm itself optimises for. We wanted to close that gap directly, not
just note that it exists.
1.2 Scope
● What we built: a federated learning pipeline for credit risk classification across three simulated
bank clients, each training locally on its own private data partition, with per-client SHAP-based
explanation generation and a central aggregator that folds cross-client explanation agreement into
how it weights each round's model update.
● The explainability method: we use SHAP (SHapley Additive exPlanations) for local feature
attribution rather than proposing a new explanation method of our own, since SHAP is already the
closest thing to a standard in centralized credit-scoring explainability (Chapter 2 covers this) —
our contribution is in how those explanations are used during aggregation, not in generating them.
● The comparison: the same non-IID data split and the same client architecture is evaluated under
three aggregation regimes — our explanation-consistency-aware FedAvg, baseline FedAvg with
no consistency weighting, and a fully centralized model as a privacy-free upper-bound reference
— so the only variable that changes is the aggregation strategy itself.
4
1.3 Problem Statement
Banks evaluating credit risk with machine learning face two constraints that are usually treated
separately: they cannot centralize customer data across institutions, and they must be able to give
consistent, defensible explanations for automated credit decisions. Federated learning satisfies the first
constraint but, under realistic non-IID data splits, does nothing on its own to guarantee the second —
local explanations can diverge across clients even when the global model itself performs well. This
project sets out to build a working federated credit-risk pipeline and, built into its core aggregation step
rather than bolted on afterward, a mechanism that keeps cross-client explanations measurably
consistent.
1.4 Objectives
1. Build a federated learning pipeline for credit risk classification across three simulated bank clients,
without centralizing raw data.
2. Generate per-client SHAP-based local explanations alongside each round of local training.
3. Design and implement an explanation-consistency-aware aggregation mechanism that folds cross-
client SHAP agreement into the Federated Averaging weighting.
4. Deploy the pipeline on genuinely isolated, multi-cloud infrastructure — AWS EC2/VPC/S3/IAM for
two clients and an Azure Virtual Machine for the third — to validate provider-agnostic
interoperability.
5. Benchmark the framework against baseline FedAvg and centralized training on accuracy, fairness,
communication overhead and a proposed explanation-consistency metric, under realistic non-IID
data partitions.
1.5 Organization of the Report
● Chapter 1: Introduction – the motivation, problem statement and objectives.
● Chapter 2: Literature Survey – existing federated learning work on credit risk, federated and
centralized explainability research, and cloud-deployed federated learning architectures.
● Chapter 3: Proposed Technique – the system architecture, module design, the aggregation
algorithm, dataset and tools.
● Chapter 4: Results and Discussion – experimental setup, evaluation methodology and
performance metrics.
● Chapter 5: Conclusion – summary and future work.
5
CHAPTER 2: LITERATURE SURVEY
2.1 Overview
Federated learning has moved from a research curiosity to a practical answer for financial institutions
that want the benefit of a shared model without the liability of a shared dataset. Over roughly the same
period, explainable AI — SHAP in particular — has become close to a baseline expectation for credit
scoring, since a rejected applicant is generally owed a specific reason, not just a score. What is harder to
find is work that sits at the intersection of the two: federated learning evaluated specifically on whether
the explanations it produces hold together across clients under realistic, non-identical data, rather than
only on whether the shared model converges. We looked at sixteen papers spanning three broad clusters
— federated learning applied to credit risk, explainability under both federated and centralized settings,
and cloud-deployed federated learning infrastructure — to establish where that gap actually sits.
2.2 Review of Existing Methods
Federated Learning Applied to Credit Risk (Papers 1–6)
Federated Learning Architectures for Credit Risk Assessment: A Comparative Analysis of VFL, HFL
and FTL (2025) compares vertical, horizontal and federated transfer learning architectures for credit risk
in FinTech, benchmarking the strengths and limitations of each. It is a useful map of which FL variant
fits which data-sharing scenario, but it does not consider an explainability layer at all, and never asks
how data heterogeneity affects the interpretability of what the resulting model outputs.
Federated Learning for Credit Risk Assessment (HICSS, 2023) builds an FL prototype that lets smaller
financial institutions cooperatively train credit risk models without sharing raw data, showing accuracy
gains for the data-poor institutions on a historical mortgage dataset. The prototype uses only a logistic-
regression baseline, though, and includes no explainability or fairness evaluation across clients.
The Effects of Data Imbalance Under a Federated Learning Approach for Credit Risk Forecasting
(2024) studies how client-level data imbalance affects FL model performance in credit risk forecasting.
It stays focused on accuracy under imbalance and does not ask whether the explanations a model gives
remain trustworthy or consistent across clients whose data is imbalanced relative to one another.
A Dynamic Receptive Field and Improved Feature Fusion Approach for Federated Learning in
Financial Credit Risk Assessment (2024) proposes a dynamic receptive-field, feature-fusion mechanism
built on squeeze-excitation networks to handle non-IID financial default data more effectively inside
FL. It measurably improves accuracy under heterogeneity, at the cost of added computational overhead,
but adds no interpretability component of its own.
Federated Learning Techniques Applied to Credit Risk Management: A Systematic Literature Review
(2023) reviews FL-for-credit-risk work published between 2018 and 2022, cataloguing the architectures
and applications in use. Its most useful finding for us is almost a confirmation of our own starting
premise: most existing FL-credit-risk work relies on simple models such as logistic regression and has
no explainability integration at all.
A Novel Federated Learning Approach with Knowledge Transfer for Credit Scoring (2024) introduces
knowledge transfer between federated clients to improve credit-scoring accuracy under heterogeneous
data, using knowledge distillation combined with federated transfer learning. It improves how well
6
accuracy transfers across clients but does not evaluate whether explanation stability improves — or even
survives — across the same federation.
Federated and Centralized Explainability for Credit Decisions (Papers 7–9, 11–14)
Federated SHAP: Privacy-Preserving and Consistent Post-hoc Explainability in Federated Learning
(2025) is the closest prior work to what we build. It constructs representative background datasets for
SHAP without sharing raw client data, meaningfully reducing the discrepancy between federated and
centrally computed SHAP values. It is validated mainly on general tabular and image benchmarks rather
than credit-risk data specifically, and — more importantly for us — its consistency scoring sits
downstream of training as an analysis step; it is never fed back into how the aggregation itself weights
client updates.
DC-SHAP: A Method for Consistent Explainability in Privacy-Preserving Distributed Machine
Learning (2023) proposes a distributed-consistent SHAP variant, built on a Data Collaboration
transformation, that produces stable explanations across distributed data partitions. It requires that data-
collaboration transformation step up front and has not been evaluated under realistic non-IID financial
client splits.
A Personalized and Explainable Federated Learning Framework (2025) integrates SHAP and LIME
within a personalized FL framework, addressing client heterogeneity through adaptive, client-level
personalization. It is applied to a fitness-recommendation domain rather than finance, and its
explainability and personalization mechanisms are not jointly optimised through the aggregation
function itself — they run alongside it rather than inside it.
Interpretable AI in Credit Scoring: A Comparative Survey of SHAP, LIME, and Hybrid Approaches
(2025) surveys XAI methods — SHAP, LIME, and hybrid interpretable ensembles — applied to
centralized credit-risk modelling. It is a useful map of the interpretability toolbox, but the setting is
entirely centralized; it does not touch how explanations behave once the model is trained across siloed,
privacy-preserving clients.
Explainable AI for Credit Assessment in Banks (2022) combines LightGBM with SHAP for consumer
loan default prediction at a Norwegian bank, with the LightGBM-plus-SHAP pipeline outperforming
the bank's existing logistic-regression model. The study is single-institution and centralized throughout,
so it does not address cross-institution privacy constraints or federated explanation consistency at all.
SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default Model (2025) is the
closest centralized precedent for the specific question we ask federally: it directly studies how stable
SHAP explanations are under retraining and data shifts in a credit-default model. That stability analysis
stays centralized, though, and is never extended to a federated, multi-client environment.
Explainable AI for Interpretable Credit Scoring (2021) benchmarks several interpretable-by-design and
post-hoc XAI techniques — BRCG, GIRP, SHAP, Anchors and ProtoDash — on the HELOC and
Lending Club datasets, the same public dataset family we use. The evaluation is centralized only, with
no treatment of privacy-preserving or federated deployment, or of cross-client explanation agreement.
Non-IID Data Handling in Federated Learning (Paper 10)
Stratify: Rethinking Federated Learning for Non-IID Data through Balanced Sampling (2025) proposes
stratified label scheduling and label-aware client selection to address non-IID data at its root cause,
7
rather than correcting for it after the fact, using homomorphic encryption to keep the client-selection
step itself privacy-preserving. It is a useful precedent for treating non-IID data as a first-class design
constraint — which we borrow — but the objective it optimises for is purely accuracy and convergence;
it does not consider explanation consistency as something the aggregation step should also account for.
Cloud-Deployed Federated Learning Architecture (Papers 15–16)
Prototype of Deployment of Federated Learning with IoT Devices (2023) demonstrates a real cloud-
deployed FL architecture using an AWS EC2 instance as the aggregator and AWS IoT Core to connect
edge clients over MQTT. It is a useful precedent for our own EC2-hosted aggregator pattern, but it is
validated for IoT and edge use cases rather than financial data, and does not discuss IAM-based per-
client data isolation — something a financial deployment cannot skip.
A Privacy-Preserving Cloud Architecture for Distributed Machine Learning at Scale (2025) proposes a
multi-cloud federated learning architecture that combines differential privacy, verifiable compliance
and adaptive governance for regulated, multi-institution ML deployment, built on hybrid Kubernetes
clusters. It is the closest precedent we found for a multi-cloud, compliance-aware deployment pattern,
applied to healthcare rather than finance — and its governance framework, thorough as it is, does not
treat explanation consistency as a first-class metric alongside its compliance and privacy guarantees.
2.3 Comparative Analysis
Paper Title & Year Technique Used Domain / Dataset Advantages Limitations
FL Architectures for
Credit Risk: VFL, HFL,
FTL (2025)
Comparative
benchmark of VFL /
HFL / FTL
FinTech credit risk
Clear map of which FL
variant fits which
scenario
No explainability layer;
heterogeneity-
interpretability link
unstudied
FL for Credit Risk
Assessment, HICSS
(2023)
FedAvg, logistic
regression
Historical mortgage
data
Accuracy gains for
data-poor institutions
Logistic regression
only; no
explainability/fairness
evaluation
Effects of Data
Imbalance Under FL
for Credit Risk
Forecasting (2024)
Federated Averaging,
imbalance-aware
sampling
Credit risk
forecasting
Isolates imbalance as a
distinct FL performance
factor
Accuracy-only;
explanation
trustworthiness under
imbalance untested
Dynamic Receptive
Field + Feature Fusion
for FL Credit Risk
(2024)
Feature fusion,
squeeze-excitation
networks
Financial default
data
Improved accuracy
under non-IID
heterogeneity
Added compute
overhead; no
interpretability
component
FL Techniques for
Credit Risk
Management:
Systematic Review
(2023)
Literature review
across FL variants
(2018–2022)
Credit risk
management
Confirms explainability
is largely absent from
prior FL-credit work
Review only — no
implementation or
benchmark of its own
Novel FL Approach
with Knowledge
Transfer for Credit
Scoring (2024)
Knowledge distillation,
Federated Transfer
Learning
Credit scoring
Improves accuracy
transfer under
heterogeneity
No evaluation of
explanation stability
across the federation
Federated SHAP:
Privacy-Preserving,
Consistent
Explainability (2025)
SHAP, federated
background-dataset
construction
General tabular /
image benchmarks
Reduces federated-vs-
centralized SHAP
discrepancy
Not credit-specific;
consistency not tied
into aggregation itself
DC-SHAP: Consistent Data Collaboration General distributed Stable explanations Requires DC transform;
8
Paper Title & Year Technique Used Domain / Dataset Advantages Limitations
Explainability in
Distributed ML (2023) (DC), SHAP ML across distributed
partitions
untested on non-IID
financial splits
A Personalized and
Explainable FL
Framework (2025)
SHAP, LIME,
personalized FL
Fitness
recommendation
Adaptive client-level
personalization
Explainability &
personalization not
jointly optimised via
aggregation
Stratify: Rethinking FL
for Non-IID Data
(2025)
Stratified label
scheduling,
homomorphic
encryption
General FL
benchmarks
Addresses non-IID data
at the root cause Accuracy/convergence
only; no explanation-
consistency objective
Interpretable AI in
Credit Scoring: SHAP,
LIME, Hybrid Survey
(2025)
SHAP, LIME,
interpretable
ensembles
Centralized credit
scoring
Broad comparative
survey of XAI methods Entirely centralized;
federated explanation
behaviour untouched
Explainable AI for
Credit Assessment in
Banks (2022)
LightGBM, SHAP Norwegian bank
loan defaults
Outperforms bank's
own logistic-regression
baseline
Single-institution,
centralized; no cross-
institution privacy angle
SHAP Stability in
Credit Risk
Management: Case
Study (2025)
SHAP stability
analysis
Credit card default
data Directly studies SHAP
stability under
retraining/shift
Centralized only; not
extended to federated
multi-client settings
Explainable AI for
Interpretable Credit
Scoring (2021)
SHAP, BRCG, GIRP,
Anchors, ProtoDash
HELOC, Lending
Club Broad benchmark of
interpretable-by-design
& post-hoc XAI
Centralized evaluation
only; no
federated/privacy
treatment
Prototype Deployment
of FL with IoT Devices
(2023)
AWS EC2, AWS IoT
Core, MQTT
IoT / edge devices Real, working cloud-
deployed FL aggregator
pattern
IoT-focused; no
financial data or per-
client IAM isolation
Privacy-Preserving
Cloud Architecture for
Distributed ML at Scale
(2025)
FL, differential
privacy, hybrid
Kubernetes
Healthcare (multi-
institution)
Multi-cloud,
governance- and
compliance-aware
architecture
Healthcare domain;
explanation-consistency
not a governance metric
2.4 Research Gap
What is missing: the federated-learning-for-credit-risk papers we found (papers 1–6) focus almost
entirely on predictive accuracy under privacy constraints, with little to no treatment of whether the
resulting model's explanations are trustworthy or consistent across the participating institutions.
Federated-explainability research (papers 7–9) treats explanation generation and consistency as a post-
hoc step layered on top of a standard FedAvg aggregation — the aggregation function itself does not
optimise for explanation consistency, only for model accuracy. The paper studying non-IID data
handling in FL (paper 10) addresses accuracy and convergence degradation caused by data
heterogeneity but does not examine how that same heterogeneity causes explanations to diverge across
clients. Centralized explainable credit-scoring research (papers 11–14), including direct studies of
SHAP stability, is extensive but entirely assumes a single, centralized dataset — none of it addresses the
federated, multi-institution setting where this stability problem is most severe and most regulator-
relevant. Cloud-deployed FL architecture papers (papers 15–16) demonstrate real infrastructure patterns
— AWS EC2 aggregators, IoT Core, multi-cloud governance — but are validated on IoT or healthcare
use cases, not on financial data with per-client IAM-based isolation and explanation auditing.
9
What we are doing about it: FedTrust-Credit tries to close these gaps at once. We integrate explanation-
consistency measurement directly into the aggregation objective, rather than treating it as a downstream
analysis step; we evaluate this specifically under realistic non-IID credit-risk data splits, partitioned by
loan grade, region and time period rather than the artificially balanced splits common in prior federated
credit-risk studies; and we validate the complete pipeline on real, isolated multi-cloud infrastructure
(AWS and Azure) with per-client access control, rather than a purely simulated or single-machine setup.
10
CHAPTER 3: PROPOSED TECHNIQUE
3.1 System Architecture
Figure 3.1: Architecture of the FedTrust-Credit Federated Learning System
The system is built around three isolated bank clients, each running local training and local explanation
generation, and a single Federated Aggregator Server that combines their updates. Figure 3.1 shows the
three pieces:
1. Bank Clients. Three simulated bank clients each hold their own private, encrypted data partition —
two on Amazon S3 buckets isolated inside their own AWS VPC, one on Azure Blob Storage behind
an Azure Virtual Machine. Each client trains a local credit risk classifier using the Flower federated
learning framework and, once local training for the round finishes, computes SHAP values for its
own model's predictions. Only the resulting model weight update and the local SHAP explanation
vector ever leave the client — the underlying credit data never does. AWS IAM enforces per-client
access policies so that no client, and nothing outside it, can reach another client's environment or
credentials.
2. Federated Aggregator Server. The aggregator receives model weights and SHAP vectors from all
three clients each round. Its Modified Federated Averaging component computes the usual size-
proportional client weighting, and its Explanation Consistency Scorer separately measures how
11
closely the clients' SHAP vectors agree with one another; the consistency signal feeds back into the
aggregation weighting rather than being computed only for reporting (Section 3.3 gives the full
algorithm). The aggregator writes the updated model to a versioned Global Model Store and
broadcasts it back to all three clients to start the next round.
3. Monitoring and Evaluation. Amazon CloudWatch tracks per-round training metrics and
communication overhead across every client and the aggregator. The Evaluation Dashboard reads
from the Global Model Store and the consistency scores to report accuracy, fairness,
communication overhead and the explanation-consistency metric across training rounds, and is
what the comparison in Chapter 4 is built on.
3.2 Module-wise Description
3.2.1 Local Training Module
Runs on each bank client. Trains a credit risk classifier on the client's own private, encrypted Lending
Club data partition using the Flower federated learning framework, and produces the local model weight
update for the current round.
3.2.2 Local Explanation (SHAP) Module
Also runs on each client, immediately after local training. Computes SHAP values for the client's local
model, producing a per-feature attribution vector that describes which inputs drove that client's risk
decisions for the round.
3.2.3 Secure Transmission Module
Handles everything that leaves a client: the model weight update and the local SHAP explanation vector,
never raw data. AWS IAM policies enforce that no client can read another client's storage or credentials,
and transmission runs over the same isolated network boundary each client's VPC (or, for Client 3,
Azure network security group) already establishes.
3.2.4 Explanation-Consistency-Aware Aggregation Module
The core novelty of the project, and the only place the three clients' updates are combined. Runs on the
Federated Aggregator Server; takes in all three clients' model weights and SHAP vectors, scores cross-
client explanation agreement, and uses that score to adjust the standard FedAvg weighting before
producing the new global model. Section 3.3 gives the pseudocode.
3.2.5 Global Model Broadcast Module
Stores the newly aggregated model in a versioned Global Model Store and broadcasts it back to all three
clients so the next federated round can begin.
3.2.6 Monitoring and Evaluation Module
Amazon CloudWatch collects per-round metrics — training time, communication volume, per-client
health — across every client and the aggregator. The Evaluation Dashboard consumes these metrics
alongside the consistency scores to track accuracy, fairness, communication overhead and explanation
consistency across rounds, and to compare the framework against the baseline FedAvg and centralized
reference runs in Chapter 4.
12
3.3 Algorithms Used
The contribution we wanted to make measurable is the aggregation step itself, so the algorithm below is
deliberately not a black box: it keeps the standard FedAvg size-proportional weighting as its base and
adjusts it with an explanation-agreement term computed directly from the clients' SHAP vectors, rather
than replacing FedAvg with a new opaque mechanism. That keeps it auditable — a regulator or an
internal risk team can trace exactly why a given client's update was weighted the way it was for a given
round.
FUNCTION explanation_consistency_score(shap_vectors):
# shap_vectors: one mean-absolute feature-attribution vector per client
pairwise_scores = []
FOR EACH PAIR (i, j) IN CLIENTS, i != j:
pairwise_scores.APPEND(COSINE_SIMILARITY(shap_vectors[i], shap_vectors[j]))
RETURN MEAN(pairwise_scores) # 1.0 = fully consistent, 0 = orthogonal
FUNCTION explanation_consistency_aware_aggregate(client_weights, client_sizes,
shap_vectors):
global_consistency = explanation_consistency_score(shap_vectors)
FOR EACH client i:
base_weight[i] = client_sizes[i] / SUM(client_sizes) #
standard FedAvg term
agreement[i] = MEAN(COSINE_SIMILARITY(shap_vectors[i], shap_vectors[j])
FOR EACH client j != i)
adjusted_weight[i] = base_weight[i] * (1 + CONSISTENCY_GAIN * agreement[i])
NORMALIZE adjusted_weight SO THAT SUM(adjusted_weight) == 1
global_model = SUM(adjusted_weight[i] * client_weights[i] FOR EACH client i)
RETURN global_model, global_consistency
CONSISTENCY_GAIN is a single tunable hyperparameter that controls how strongly explanation
agreement can shift a client's influence away from plain FedAvg's data-size weighting; setting it to zero
recovers standard FedAvg exactly, which is also how the baseline comparison in Chapter 4 is produced
from the same codebase. The global_consistency value returned each round is logged directly as the
explanation-consistency metric used throughout the evaluation.
3.4 Dataset Details
Attribute Description Type
loan_id Unique identifier for each loan application String
loan_amnt Requested loan amount Float
term Loan repayment term — 36 or 60 months String
int_rate Interest rate assigned to the loan Float
grade / sub_grade Lending Club's internal risk grade for the loan String
annual_inc Borrower's self-reported annual income Float
dti Debt-to-income ratio Float
home_ownership Rent / own / mortgage status String
purpose Stated purpose of the loan String
13
Attribute Description Type
addr_state Borrower's US state — used as one non-IID partitioning axis String
issue_d Loan issue date — used as the time-period partitioning axis Datetime
loan_status Fully paid / charged off / default — the prediction target String
Dataset source: the publicly available Lending Club Loan Dataset. Rather than splitting it evenly across
the three simulated clients, we partition it by loan grade, borrower region (addr_state) and issue-date
period so that each client sees a genuinely different customer population — similar to how three real
regional banks would each see a different risk mix — which is what makes the non-IID condition
realistic rather than artificial. This partitioning scheme is what the explanation-consistency mechanism
in Section 3.3 is actually being tested against.
3.5 Tools and Technologies
Layer Technology
Federated learning framework Flower (flwr)
Client compute — Bank 1 & Bank 2 AWS EC2, each inside its own isolated Amazon VPC
Client compute — Bank 3 Azure Virtual Machine
Network isolation Amazon VPC (per-client boundary, AWS side)
Data storage Amazon S3 (encrypted, per-client bucket) / Azure Blob Storage
Access control AWS IAM — per-client, least-privilege policies
Explainability SHAP (SHapley Additive exPlanations)
Monitoring Amazon CloudWatch (training-round metrics, communication overhead)
Managed ML hosting (optional) Amazon SageMaker, as a managed alternative to a raw EC2 aggregator if timeline
permits
Dataset Lending Club Loan Dataset (public)
Programming language Python 3.10+ (Flower, SHAP, model training and evaluation scripts)
14
CHAPTER 4: RESULTS AND DISCUSSION
4.1 Experimental Setup
System configuration: Bank Client 1 and Bank Client 2 run on AWS EC2 instances in the same AWS
region, each inside its own VPC with its own S3 bucket, so that cross-client network latency inside AWS
is not a confounding variable. Bank Client 3 runs on an Azure Virtual Machine, provisioned in a
geographically comparable region, specifically so that the multi-cloud comparison reflects a genuine
provider difference rather than a regional one. The Federated Aggregator Server runs on its own AWS
EC2 instance, separate from any client, so its resource usage does not compete with a client's training
workload.
Tools used: Flower for the federated learning protocol on all three clients and the aggregator; SHAP for
local explanation generation; Amazon S3, AWS IAM and Amazon VPC for the two AWS-hosted
clients; Azure Blob Storage for the Azure-hosted client; and Amazon CloudWatch for monitoring
throughout.
4.2 Evaluation Methodology
All three variants — our explanation-consistency-aware aggregation, baseline FedAvg (the same
codebase with CONSISTENCY_GAIN set to zero), and a centralized model trained on the pooled data
as a privacy-free reference — are run on the identical non-IID client split described in Section 3.4, for
the same fixed number of federated rounds. At the end of every round we record each client's local
accuracy, the pooled test-set accuracy of the current global model, the bytes transmitted between clients
and the aggregator, and the explanation-consistency score. Running all three variants from one shared
codebase, differing only in the aggregation weighting, is what keeps the comparison fair — the same
discipline CloudRelief and the broader literature (Jiang et al., 2021, in a different context) treat as a
precondition for a controlled comparison, rather than three separately-tuned pipelines that happen to be
compared afterward.
4.3 Performance Metrics
Four metrics anchor the comparison:
● Predictive performance. Accuracy, AUC-ROC and F1 on a held-out test split, reported both per-
client and on the pooled test set, so a gain that only shows up for one client's local population is
visible rather than averaged away.
● Explanation-Consistency Score. The metric defined in Section 3.3 — mean pairwise cosine
similarity of the clients' SHAP vectors — logged every round for all three variants, including the
centralized reference (computed there by splitting the centralized model's predictions back out
along the same client boundaries, purely for comparison, since a centralized model has no real
notion of per-client explanations).
● Communication overhead. Total bytes transmitted per round between clients and the aggregator
— model weights plus, for our variant, the SHAP vectors — compared against baseline FedAvg's
model-weights-only traffic, since the SHAP vectors are the one thing our approach adds to the
wire.
● Fairness across client populations. Since the public Lending Club dataset carries no protected
demographic attributes, fairness here is measured at the institutional level — the spread in approval
15
rate and error rate across the three clients' regional populations — rather than across race or gender,
which the data does not contain and which we do not attempt to infer.
4.4 Analysis Framework
Once the runs described above have produced real numbers, here is roughly how we plan to read them:
● If our explanation-consistency-aware aggregation raises the Explanation-Consistency Score
meaningfully above baseline FedAvg without a material drop in pooled accuracy, that supports the
central claim of this project — that trustworthy, consistent explanations do not have to come at the
cost of predictive performance.
● If communication overhead only grows modestly over baseline FedAvg, that supports real-world
feasibility, since SHAP vectors are far smaller than a full model weight update; a large increase
would call the practicality of the approach into question regardless of how well it does on the other
metrics.
● If the centralized reference's accuracy sits close to our federated variant's, that indicates privacy is
not costing much predictive power — consistent with what papers 1–6 in Chapter 2 report for
federated credit-risk models generally.
● Any consistent gap in per-round completion time between the AWS-hosted clients and the Azure-
hosted client would show up as an asymmetry in the per-client timing logs, and is worth watching
given the cloud-architecture precedent in papers 15–16 — a genuine multi-cloud deployment is
one of the things that separates this project from a single-provider simulation.
We are writing this section in the conditional because, honestly, it has to stay that way until the federated
rounds described in Section 4.2 have actually been run and there are real numbers in front of us to
interpret, rather than a plan for how we would interpret them.
16
CHAPTER 5: CONCLUSION AND FUTURE WORK
5.1 Conclusion
5.1.1 Summary of Work Done
This project designs FedTrust-Credit, a federated learning framework for credit risk assessment in
which three simulated bank clients train locally on their own private, non-IID partitions of the Lending
Club Loan Dataset and generate local SHAP-based explanations alongside each round of training. What
sets it apart from a standard FedAvg build is that cross-client explanation agreement is measured and fed
directly into how the aggregator weights each client's update, through the explanation-consistency-
aware aggregation mechanism described in Chapter 3, rather than being checked as a separate analysis
step after training finishes. The pipeline is deployed on genuinely isolated, multi-cloud infrastructure —
two clients on AWS EC2 inside their own VPCs with IAM-enforced access control, one client on an
Azure Virtual Machine — with a controlled benchmarking setup built specifically to compare it against
baseline FedAvg and a centralized reference under identical non-IID conditions.
5.1.2 Key Findings
Based on the evaluation framework set out in Chapter 4, this project is built to produce actual measured
evidence — not an assumption carried over from the accuracy-only federated-credit-risk literature
reviewed in Chapter 2 — on whether an aggregation mechanism that is explicitly aware of cross-client
explanation agreement can hold that agreement measurably higher than plain FedAvg, without giving
up meaningful predictive accuracy, communication efficiency, or cross-client fairness. The specific
numbers belong in the results table once the federated rounds described in Section 4.2 have actually
been run, read against the framework laid out in Section 4.4.
5.2 Future Work
5.2.1 Possible Enhancements
A few extensions seem worth doing later. Extending the explanation-consistency comparison beyond
SHAP to LIME or Anchors would test whether the aggregation mechanism generalises across
explanation methods, building on the method comparison in papers 9 and 14 (Chapter 2). Layering
differential privacy on top of the current IAM-based isolation, in the spirit of the compliance-aware
architecture in paper 16, would push the privacy guarantee from access-control alone to a formal,
provable bound. Scaling from three simulated clients to a larger federation would test whether the
consistency score — and the CONSISTENCY_GAIN hyperparameter tuned for three clients — still
behaves sensibly as the number of participating institutions grows. And a client-specific, personalized
weighting scheme, closer to what paper 9 explored for a different domain, could let the aggregator
account for a client whose explanations are legitimately different because its customer population is
different, rather than penalising it as simply inconsistent.
5.2.2 Real-World Deployment Scope
Taking this beyond a course project would mean validating the framework with real participating banks
under an actual regulatory review, and running a formal fairness audit against real protected-attribute
data under proper consent and compliance frameworks — something the public Lending Club dataset
used here does not permit, since it carries no protected demographic fields. Extending the multi-cloud
deployment to a third provider, such as Google Cloud, would also turn the current two-provider
17
comparison into a genuinely three-way one. Both are outside what we could reasonably cover here, but
the aggregation mechanism itself is not tied to credit risk specifically; any federation of institutions that
needs both privacy-preserving training and cross-client explanation consistency could reuse the same
approach.
18
REFERENCES
[1] “Federated Learning Architectures for Credit Risk Assessment: A Comparative Analysis of Vertical,
Horizontal, and Transfer Learning Approaches,” IEEE Conference Publication, 2025.
[2] “Federated Learning for Credit Risk Assessment,” Proceedings of the Hawaii International Conference
on System Sciences (HICSS), 2023.
[3] S. Zhang et al., “The Effects of Data Imbalance Under a Federated Learning Approach for Credit Risk
Forecasting,” arXiv:2401.07234, 2024.
[4] “A Dynamic Receptive Field and Improved Feature Fusion Approach for Federated Learning in Financial
Credit Risk Assessment,” Scientific Reports, 2024.
[5] “Federated Learning Techniques Applied to Credit Risk Management: A Systematic Literature Review,”
EDPACS, vol. 68, no. 1, 2023.
[6] Z. Wang et al., “A Novel Federated Learning Approach with Knowledge Transfer for Credit Scoring,”
2024.
[7] “Federated SHAP: Privacy-Preserving and Consistent Post-hoc Explainability in Federated Learning,”
Machine Learning, Springer, 2025.
[8] A. Bogdanova, A. Imakura, T. Sakurai, “DC-SHAP Method for Consistent Explainability in Privacy-
Preserving Distributed Machine Learning,” Human-Centric Intelligent Systems, vol. 3, no. 3, pp. 197–
210, 2023.
[9] “A Personalized and Explainable Federated Learning Framework,” DiVA Academic Archive, 2025.
[10] H. Y. Wong, C. K. Lim, C. S. Chan, “Stratify: Rethinking Federated Learning for Non-IID Data through
Balanced Sampling,” arXiv:2504.13462, 2025.
[11] “Interpretable AI in Credit Scoring: A Comparative Survey of SHAP, LIME, and Hybrid Approaches,”
R Discovery, 2025.
[12] “Explainable AI for Credit Assessment in Banks,” Journal of Risk and Financial Management (MDPI),
vol. 15, no. 12, 2022.
[13] L. Lin, Y. Wang, “SHAP Stability in Credit Risk Management: A Case Study in Credit Card Default
Model,” arXiv:2508.01851, 2025.
[14] “Explainable AI for Interpretable Credit Scoring,” arXiv:2012.03749, 2021.
[15] “Prototype of Deployment of Federated Learning with IoT Devices,” arXiv:2311.14401, 2023.
[16] “A Privacy-Preserving Cloud Architecture for Distributed Machine Learning at Scale,”
arXiv:2512.10341, 2025.
19
FedTrust-Credit
Federated Learning with
Explanation‑Consistency‑Aware Aggregation for
Privacy‑Preserving Credit Risk Assessment
Cloud Computing Project — Initial Review
Prashaanth Raj J M (23BIT0173) · Prasannaa V (23BIT0041) · Haswanth K (23BIT0359)
Under : Dr. Siva Rama Krishnan S
Problem Statement
Financial institutions cannot share raw customer data due to privacy
regulations and competitive concerns — yet a shared credit risk model
would improve accuracy and robustness.
Federated Learning helps
Enables collaborative model
training without exchanging raw
data (weights-only
communication).
New challenge
When client populations differ
(non‑IID), model explanations
(e.g., SHAP) may diverge —
undermining trust and regulatory
compliance.
Regulatory risk
Inconsistent explanations across banks conflict with “right to
explanation” requirements and auditability.
Motivation & Novelty
Credit risk systems must be accurate and explainable to be legally and
ethically deployable. Current work separates federated optimization
(FedAvg) from explainability (SHAP), offering no guarantee that
explanations align across clients.
Gap
Federated + XAI treated
independently → potential
explanation drift across
clients.
Contribution
A novel aggregation
mechanism that jointly
optimizes model weights and
explanation consistency
across non‑IID clients.
Impact
Improves regulatory compliance, auditability, and cross‑institution
trust — publication and patent potential.
Objectives
 Design and implement a federated learning pipeline for credit risk prediction across simulated multi‑bank clients.
 Develop an explanation‑consistency‑aware aggregation algorithm (core novel contribution).
 Deploy and validate on multi‑cloud infrastructure (AWS + Azure) for practical feasibility.
 Evaluate against baselines on accuracy, fairness, privacy, communication cost, and explanation‑consistency.
Dataset & Partitioning
Lending Club (~2.2M records) is the primary dataset with loan
outcome labels; German Credit (UCI) used for cross‑dataset
generalization validation. Both are public, pre‑labeled, and
real‑world.
1. Partition strategy: split by loan grade, region, and time period
to simulate non‑IID bank populations (no random shuffling).
2. No synthetic data — preserves real distributional differences
between institutions.
3. Privacy controls: per‑client encrypted S3 buckets and IAM
policies in the cloud deployment.
System Architecture
Three simulated bank clients (isolated VPCs on AWS EC2 + Azure VM) train locally on private data, sharing only model weights and SHAP explanation vectors
with a central aggregator. The aggregator performs explanation‑consistency‑aware FedAvg and broadcasts the updated global model.
Diagram: isolated client compute → local model + SHAP vectors → federated aggregator → explanation consistency scorer → global model store and
monitoring (CloudWatch & evaluation dashboard).
Broadcast
Model
Aggregate
& Score
Compute
SHAP
Local Train
Workflow summary: (1) clients train classifiers locally; (2) clients compute SHAP explanation vectors for their validation
predictions; (3) aggregator fuses weights and scores explanation alignment across clients to adjust aggregation weights;
(4) updated model is broadcast and the cycle repeats for N rounds. Tooling: Flower, SHAP, scikit-learn / XGBoost.
Cloud Infrastructure &
Security
 AWS EC2: client and aggregator compute instances
with isolated VPCs per simulated bank.
 AWS S3: encrypted per‑client data stores; IAM policies
enforce least privilege.
 Amazon CloudWatch: training round metrics, latency
and communication monitoring.
 Azure VM: one client on Azure to validate multi‑cloud
interoperability and cross‑provider communication.
Network isolation and strict IAM policies simulate
realistic institutional boundaries while enabling
controlled federated experiments.
Expected Outcomes & Evaluation
Plan
We will compare: (a) our explanation‑consistency-aware aggregation, (b)
FedAvg baseline, and (c) centralized training (upper bound).
Metrics
Prediction accuracy, fairness across client groups,
explanation‑consistency score (novel), communication overhead,
and privacy indicators.
Expected result
Accuracy close to centralized training while achieving higher
explanation consistency than FedAvg baseline.
Outcomes
Target: conference/journal submission and patent filing pending
novelty confirmation of the aggregation method.
Timeline
Weeks Focus
1–2 Cloud environment provisioning, dataset
ingestion, partitioning to simulate non‑IID clients
3–5 Implement baseline FedAvg + SHAP pipeline;
initial evaluations
6–8 Design and implement
explanation‑consistency‑aware aggregation; run
full evaluations
9–10 Results analysis, documentation, and write‑up
for publication
Next steps: finalize aggregation math, instrument explanation‑consistency
metric, and begin controlled experiments on multi‑cloud setup.