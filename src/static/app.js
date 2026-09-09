// FedTrust-Credit Interactive Dashboard Client Logic

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPlotGallery();
  initSimulator();
  initRiskForm();
  // Trigger initial simulation and initial risk scoring
  runSimulation();
  evaluateLoanRisk();
  // Load real benchmark data from API
  loadRealMetrics();
  loadServiceHealth();
});

/* ── 0. Load Real Benchmark Metrics from API ─────────────────────────────────── */

async function loadRealMetrics() {
  try {
    const res = await fetch("/api/metrics/summary");
    if (!res.ok) return;
    const data = await res.json();

    // Update KPI cards with real data
    const kpiCards = document.querySelectorAll(".kpi-value");
    if (kpiCards.length >= 4 && data.pooled_metrics) {
      const ours = data.pooled_metrics.ours;
      const fedavg = data.pooled_metrics.fedavg;
      const cent = data.pooled_metrics.centralized;

      // Accuracy card
      kpiCards[0].textContent = (ours.accuracy * 100).toFixed(2) + "%";
      // AUC card
      kpiCards[1].textContent = ours.auc.toFixed(4);
      // Consistency card
      kpiCards[2].textContent = ours.final_consistency.toFixed(4);
    }

    // Update data source indicator
    const statusText = document.querySelector(".status-text");
    if (statusText) {
      statusText.textContent = "Live — Real Data";
      statusText.style.color = "#4ade80";
    }
  } catch (err) {
    console.log("Metrics load deferred:", err.message);
  }
}

async function loadServiceHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) return;
    const data = await res.json();

    // Update status indicator based on model readiness
    const statusText = document.querySelector(".status-text");
    if (statusText && data.model_initialized) {
      statusText.textContent = "Live — " + (data.data_source || "Real Data");
      statusText.style.color = "#4ade80";
    } else if (statusText) {
      statusText.textContent = "API Online (Models Loading)";
      statusText.style.color = "#f59e0b";
    }
  } catch (err) {
    const statusText = document.querySelector(".status-text");
    if (statusText) {
      statusText.textContent = "Service Offline";
      statusText.style.color = "#ef4444";
    }
  }
}

/* ── 1. Tab Switching ────────────────────────────────────────────────────────── */

function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const contents = document.querySelectorAll(".tab-content");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      contents.forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetEl = document.getElementById(targetId);
      if (targetEl) targetEl.classList.add("active");
    });
  });
}

/* ── 2. Benchmark Plots Gallery ─────────────────────────────────────────────── */

const PLOT_CAPTIONS = {
  consistency_comparison: "Explanation Consistency Progression: FedTrust-Credit maintains elevated cross-institutional attribution agreement throughout 20 federated communication rounds.",
  accuracy_comparison: "Pooled Model Accuracy: High classification accuracy (~82.57%) is fully preserved with zero regression compared to plain FedAvg.",
  auc_comparison: "ROC-AUC Convergence: Discriminative capability converges stably across all federated communication rounds.",
  per_client_f1: "Per-Client Macro F1 Score: Performance grouped by bank silo under extreme non-IID loan demographic skew.",
  communication_overhead: "Communication Overhead Comparison: Explanation vectors incur only 1,008 bytes per client per round — an imperceptible +0.0316% bandwidth overhead.",
  fairness_spread: "Institutional Fairness Spread: Quantifying performance disparity across distinct lending silos under non-IID conditions."
};

function initPlotGallery() {
  const plotButtons = document.querySelectorAll(".plot-btn");
  const plotImg = document.getElementById("activePlotImg");
  const plotCaption = document.getElementById("activePlotCaption");

  plotButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      plotButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const plotKey = btn.getAttribute("data-plot");
      plotImg.src = `/api/plots/${plotKey}?t=${Date.now()}`;
      plotCaption.innerHTML = `<strong>${btn.textContent}:</strong> ${PLOT_CAPTIONS[plotKey] || ""}`;
    });
  });
}

/* ── 3. Federated Aggregator Simulator ──────────────────────────────────────── */

function initSimulator() {
  const gainSlider = document.getElementById("gainSlider");
  const gainDisplay = document.getElementById("gainValueDisplay");
  const noiseSlider = document.getElementById("noiseSlider");
  const noiseDisplay = document.getElementById("noiseValueDisplay");
  const btnRun = document.getElementById("btnRunSim");

  gainSlider.addEventListener("input", (e) => {
    gainDisplay.textContent = parseFloat(e.target.value).toFixed(1);
  });

  noiseSlider.addEventListener("input", (e) => {
    noiseDisplay.textContent = parseFloat(e.target.value).toFixed(2);
  });

  btnRun.addEventListener("click", runSimulation);
  gainSlider.addEventListener("change", runSimulation);
  noiseSlider.addEventListener("change", runSimulation);
}

async function runSimulation() {
  const gain = parseFloat(document.getElementById("gainSlider").value);
  const noise = parseFloat(document.getElementById("noiseSlider").value);

  try {
    const res = await fetch("/api/federated/simulate-round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consistency_gain: gain,
        correlation_noise: noise,
        client_sizes: [627716, 287686, 81060],
      })
    });

    if (!res.ok) throw new Error("Simulation endpoint returned error");
    const data = await res.json();
    renderSimulationResults(data);
  } catch (err) {
    console.error("Simulation error:", err);
  }
}

function renderSimulationResults(data) {
  // Update consensus score
  const consEl = document.getElementById("simGlobalCons");
  if (consEl) consEl.textContent = data.global_consistency.toFixed(4);

  // Render client weight cards
  const grid = document.getElementById("clientWeightsGrid");
  if (grid) {
    grid.innerHTML = data.clients.map(c => {
      const deltaSign = c.weight_delta_pct >= 0 ? "+" : "";
      const deltaClass = c.weight_delta_pct >= 0 ? "success-accent" : "text-muted";
      return `
        <div class="weight-card">
          <h4>${c.name}</h4>
          <p class="client-sub">${c.client_id} &bull; ${c.size.toLocaleString()} loans</p>
          
          <div class="weight-stat-row">
            <span>Base FedAvg Weight:</span>
            <strong>${c.base_weight_pct.toFixed(2)}%</strong>
          </div>
          
          <div class="weight-stat-row">
            <span>Agreement Score (a<sub>i</sub>):</span>
            <strong>${c.agreement_score.toFixed(4)}</strong>
          </div>

          <div class="weight-stat-row">
            <span>Adjusted Weight (w<sub>i</sub>*):</span>
            <strong style="color: var(--accent-cyan); font-size: 1.05rem;">${c.adjusted_weight_pct.toFixed(2)}%</strong>
          </div>

          <div class="weight-stat-row">
            <span>Weight Shift (&Delta;w):</span>
            <strong class="${deltaClass}">${deltaSign}${c.weight_delta_pct.toFixed(3)}%</strong>
          </div>

          <div class="weight-bar-track">
            <div class="weight-bar-fill" style="width: ${c.adjusted_weight_pct}%"></div>
          </div>
        </div>
      `;
    }).join("");
  }

  // Render matrix
  const matrixContainer = document.getElementById("matrixContainer");
  if (matrixContainer && data.pairwise_similarity_matrix) {
    const clients = ["Bank 1", "Bank 2", "Bank 3"];
    let cellsHtml = "";
    data.pairwise_similarity_matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        cellsHtml += `
          <div class="matrix-cell">
            <div class="cell-labels">${clients[i]} &harr; ${clients[j]}</div>
            <div class="cell-val">${val.toFixed(4)}</div>
          </div>
        `;
      });
    });
    matrixContainer.innerHTML = cellsHtml;
  }
}

/* ── 4. Real-time Credit Risk Assessment & SHAP ──────────────────────────────── */

const PERSONA_PRESETS = {
  prime: {
    loan_amnt: 12000,
    term: 36,
    int_rate: 7.9,
    grade: "A",
    annual_inc: 110000,
    dti: 11.5,
    home_ownership: "MORTGAGE",
    purpose: "debt_consolidation"
  },
  balanced: {
    loan_amnt: 18000,
    term: 36,
    int_rate: 14.5,
    grade: "C",
    annual_inc: 68000,
    dti: 21.0,
    home_ownership: "RENT",
    purpose: "credit_card"
  },
  subprime: {
    loan_amnt: 28000,
    term: 60,
    int_rate: 24.8,
    grade: "E",
    annual_inc: 42000,
    dti: 38.5,
    home_ownership: "RENT",
    purpose: "small_business"
  }
};

function initRiskForm() {
  const form = document.getElementById("loanForm");
  const presetSelect = document.getElementById("personaSelect");

  presetSelect.addEventListener("change", (e) => {
    const pKey = e.target.value;
    if (PERSONA_PRESETS[pKey]) {
      const p = PERSONA_PRESETS[pKey];
      document.getElementById("loanAmnt").value = p.loan_amnt;
      document.getElementById("term").value = p.term;
      document.getElementById("intRate").value = p.int_rate;
      document.getElementById("grade").value = p.grade;
      document.getElementById("annualInc").value = p.annual_inc;
      document.getElementById("dti").value = p.dti;
      document.getElementById("homeOwnership").value = p.home_ownership;
      document.getElementById("purpose").value = p.purpose;
      evaluateLoanRisk();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    evaluateLoanRisk();
  });
}

async function evaluateLoanRisk() {
  const payload = {
    loan_amnt: parseFloat(document.getElementById("loanAmnt").value),
    term: parseInt(document.getElementById("term").value),
    int_rate: parseFloat(document.getElementById("intRate").value),
    grade: document.getElementById("grade").value,
    annual_inc: parseFloat(document.getElementById("annualInc").value),
    dti: parseFloat(document.getElementById("dti").value),
    home_ownership: document.getElementById("homeOwnership").value,
    purpose: document.getElementById("purpose").value,
    addr_state: document.getElementById("addrState").value,
  };

  try {
    const res = await fetch("/api/predict/risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Prediction request failed");
    const result = await res.json();
    renderRiskVerdict(result);
  } catch (err) {
    console.error("Risk prediction error:", err);
  }
}

function renderRiskVerdict(res) {
  // Score & Verdict
  const scoreNum = document.getElementById("scoreNum");
  const scoreCircle = document.getElementById("scoreCircle");
  const tierPill = document.getElementById("tierPill");
  const decisionText = document.getElementById("decisionText");
  const consNote = document.getElementById("consNote");

  const pd = res.default_probability_pct;
  scoreNum.textContent = `${pd.toFixed(1)}%`;

  tierPill.className = "tier-pill";
  if (pd < 15.0) {
    tierPill.classList.add("prime");
    tierPill.textContent = "Low Risk Tier (Prime)";
    scoreCircle.style.borderColor = "var(--accent-emerald)";
  } else if (pd < 30.0) {
    tierPill.classList.add("moderate");
    tierPill.textContent = "Moderate Risk Tier (Near-Prime)";
    scoreCircle.style.borderColor = "var(--accent-amber)";
  } else {
    tierPill.classList.add("subprime");
    tierPill.textContent = "High Risk Tier (Subprime)";
    scoreCircle.style.borderColor = "var(--accent-crimson)";
  }

  decisionText.textContent = res.decision;
  consNote.innerHTML = `Multi-institution explanation consensus: <strong>${res.explanation_consistency.toFixed(4)}</strong>`;

  // SHAP Waterfall/Bars
  const shapBars = document.getElementById("shapBars");
  if (shapBars && res.feature_attributions) {
    const maxAbs = Math.max(...res.feature_attributions.map(f => f.abs_importance), 0.001);

    shapBars.innerHTML = res.feature_attributions.map(f => {
      const isPos = f.shap_value > 0;
      const barClass = isPos ? "pos-risk" : "neg-risk";
      const pctWidth = Math.min(100, Math.max(6, (f.abs_importance / maxAbs) * 100));
      const readableName = formatFeatureName(f.feature);

      return `
        <div class="shap-row">
          <div class="shap-label" title="${f.feature}">${readableName}</div>
          <div class="shap-track">
            <div class="shap-fill ${barClass}" style="width: ${pctWidth}%"></div>
          </div>
          <div class="shap-val-text">${f.shap_value > 0 ? "+" : ""}${f.shap_value.toFixed(3)}</div>
        </div>
      `;
    }).join("");
  }

  // Institutional Perspectives
  const persGrid = document.getElementById("perspectivesGrid");
  if (persGrid && res.client_perspectives) {
    const clientNames = {
      client_1: "Bank 1 (Prime Silo)",
      client_2: "Bank 2 (East Coast Silo)",
      client_3: "Bank 3 (Subprime Silo)"
    };

    persGrid.innerHTML = Object.entries(res.client_perspectives).map(([cId, info]) => {
      return `
        <div class="perspective-card">
          <div class="bank-name">${clientNames[cId] || cId}</div>
          <div class="bank-pd">${info.default_probability.toFixed(1)}% PD</div>
          <div class="bank-driver">Top driver: <strong>${formatFeatureName(info.top_factor)}</strong></div>
        </div>
      `;
    }).join("");
  }
}

function formatFeatureName(raw) {
  const map = {
    int_rate: "Interest Rate",
    dti: "Debt-to-Income",
    annual_inc: "Annual Income",
    loan_amnt: "Loan Amount",
    grade_num: "Credit Grade",
    term: "Loan Term",
    home_ownership_rent: "Renting Home",
    home_ownership_own: "Owns Home",
    home_ownership_mortgage: "Mortgage",
    purpose_debt_consolidation: "Debt Consolidation",
    purpose_credit_card: "Credit Card Refi",
    purpose_small_business: "Small Business"
  };
  return map[raw] || raw;
}
