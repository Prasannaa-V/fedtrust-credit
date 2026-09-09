"""
data_partition.py — Non-IID Lending Club data loading and partitioning
FedTrust-Credit | Day 1-2

Partitioning strategy (Section 3.4):
  Client 1 (Bank 1): loan grade A or B
  Client 2 (Bank 2): addr_state in East Coast states
  Client 3 (Bank 3): issue_d in the most recent 2 years of the dataset

Assignment priority (DECISIONS.md D-001): grade A/B first, then East Coast, then recent period.
Records qualifying for none of the three are excluded from federated training.

Target encoding (DECISIONS.md D-004):
  loan_status "Fully Paid" -> 0
  loan_status "Charged Off" | "Default" -> 1
  All other statuses dropped.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────

# Required columns per Section 3.4 dataset schema
REQUIRED_COLS = [
    "loan_amnt", "term", "int_rate", "grade", "sub_grade",
    "annual_inc", "dti", "home_ownership", "purpose",
    "addr_state", "issue_d", "loan_status",
    "emp_length", "revol_bal", "revol_util",
    "total_acc", "open_acc", "pub_rec",
    "installment", "verification_status",
]

# Optional: loan_id if present (used for record-keeping only, not training)
ID_COL = "id"

# Binary target encoding (D-004)
STATUS_POSITIVE = {"Charged Off", "Default"}   # -> 1 (default)
STATUS_NEGATIVE = {"Fully Paid"}               # -> 0 (repaid)

# Client 1: grade partition (D-002 / Section 3.4)
CLIENT1_GRADES = {"A", "B"}

# Client 2: East Coast states (D-002)
CLIENT2_STATES = {
    "NY", "NJ", "PA", "MA", "CT", "VA", "NC", "SC",
    "GA", "FL", "MD", "DE", "NH", "VT", "ME", "RI",
}

# Train/test split ratio (D-009)
TEST_RATIO = 0.20
RANDOM_STATE = 42


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the Lending Club CSV, filter to required columns, encode the target,
    and return a clean DataFrame ready for partitioning.

    Args:
        csv_path: Path to the Lending Club CSV file.

    Returns:
        Clean DataFrame with columns: [REQUIRED_COLS..., 'target']
    """
    csv_path = Path(csv_path)
    print(f"[data_partition] Loading: {csv_path}")

    # Load — the full Lending Club file has ~150 columns; read in chunks if large
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=lambda c: c in set(REQUIRED_COLS + [ID_COL]),
    )
    print(f"[data_partition]   Raw rows: {len(df):,}")

    # ── Target encoding ─────────────────────────────────────────────────────
    # Keep only rows with a known outcome (Fully Paid or Charged Off/Default)
    df = df[df["loan_status"].isin(STATUS_POSITIVE | STATUS_NEGATIVE)].copy()
    df["target"] = df["loan_status"].apply(
        lambda s: 1 if s in STATUS_POSITIVE else 0
    )
    print(f"[data_partition]   After dropping unknown statuses: {len(df):,}")
    print(f"[data_partition]   Class distribution: {df['target'].value_counts(normalize=True).to_dict()}")

    # ── issue_d → datetime (support both 4-digit and 2-digit years) ──────
    dt1 = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    dt2 = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")
    df["issue_d"] = dt1.fillna(dt2)

    # ── int_rate: strip '%' if present ─────────────────────────────────────
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.rstrip("%").astype(float)

    # ── term: strip ' months' if present ───────────────────────────────────
    if df["term"].dtype == object:
        df["term"] = df["term"].str.strip().str.extract(r"(\d+)").astype(float)

    # ── emp_length: extract numeric years ("10+ years" → 10, "< 1 year" → 0)
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )
        df["emp_length"] = df["emp_length"].fillna(0)

    # ── revol_util: strip '%' if present ───────────────────────────────────
    if "revol_util" in df.columns and df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].str.rstrip("%").astype(float)

    # ── Drop rows with null in key columns ─────────────────────────────────
    key_cols = ["grade", "addr_state", "issue_d", "annual_inc", "dti", "loan_amnt"]
    before = len(df)
    df = df.dropna(subset=key_cols)
    print(f"[data_partition]   After dropping nulls in key cols: {len(df):,} (dropped {before - len(df):,})")

    # ── Engineered features ────────────────────────────────────────────────
    # Debt-to-income ratio interaction with interest rate
    if "dti" in df.columns and "int_rate" in df.columns:
        df["dti_x_int_rate"] = df["dti"] * df["int_rate"]

    # Installment as fraction of monthly income
    if "installment" in df.columns and "annual_inc" in df.columns:
        monthly_inc = df["annual_inc"] / 12.0
        df["installment_to_income"] = (
            df["installment"] / monthly_inc.clip(lower=1.0)
        )

    # ── One-hot encode categoricals ─────────────────────────────────────────
    cat_cols = ["term", "grade", "sub_grade", "home_ownership", "purpose", "addr_state", "verification_status"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    return df


# ── Partitioning ───────────────────────────────────────────────────────────────

def partition_noniid(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Assign records to three non-IID client partitions per Section 3.4.

    Priority (D-001):
      1. Client 1 (grade A/B)
      2. Client 2 (East Coast states)
      3. Client 3 (most recent 2 years)
      Remainder: excluded from federated training

    Returns:
        dict with keys "client_1", "client_2", "client_3" — each a DataFrame
    """
    # Identify time threshold for Client 3 (D-003: most recent 2 years)
    max_date = df["issue_d"].max()
    cutoff_date = max_date - pd.DateOffset(years=2)
    print(f"[data_partition] Client 3 time window: {cutoff_date.date()} to {max_date.date()}")

    # Assignment masks (strict priority, mutually exclusive)
    mask_client1 = df["grade"].str.upper().isin(CLIENT1_GRADES)
    mask_client2 = (~mask_client1) & df["addr_state"].str.upper().isin(CLIENT2_STATES)
    mask_client3 = (~mask_client1) & (~mask_client2) & (df["issue_d"] >= cutoff_date)

    clients = {
        "client_1": df[mask_client1].copy(),
        "client_2": df[mask_client2].copy(),
        "client_3": df[mask_client3].copy(),
    }

    excluded = len(df) - sum(len(v) for v in clients.values())
    print(f"[data_partition] Excluded (no partition match): {excluded:,}")

    return clients


# ── Feature Engineering ────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop non-feature columns and one-hot encode categoricals.
    If feature_cols is provided, align output to that exact column list
    (filling missing columns with 0). This ensures all clients share the
    same feature space.
    Returns (X, y) ready for model training.
    """
    drop_cols = ["loan_status", "issue_d", ID_COL]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y = feature_df.pop("target")

    # One-hot encode remaining categoricals
    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)

    # Fill NaN with column median
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    # Align to shared feature space if provided
    if feature_cols is not None:
        missing = [col for col in feature_cols if col not in feature_df.columns]
        if missing:
            zeros = pd.DataFrame(0, index=feature_df.index, columns=missing)
            feature_df = pd.concat([feature_df, zeros], axis=1)
        feature_df = feature_df[feature_cols]

    return feature_df, y


def build_shared_feature_columns(clients: dict[str, pd.DataFrame]) -> list[str]:
    """
    Compute the union of all feature columns across all client partitions.
    This ensures all clients train and evaluate on the same feature space,
    which is required for SHAP vector comparison (Section 3.3).
    Absent features in a partition are filled with 0.
    """
    all_cols: set[str] = set()
    for cdf in clients.values():
        X, _ = prepare_features(cdf)
        all_cols.update(X.columns.tolist())
    return sorted(all_cols)  # sorted for reproducibility


def get_client_splits(
    client_df: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns (X_train, X_test, y_train, y_test) for a single client partition.
    Stratified on target label (D-009).
    If feature_cols provided, aligns to shared feature space.
    """
    X, y = prepare_features(client_df, feature_cols=feature_cols)
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state, stratify=y)


# ── Stats & Reporting ──────────────────────────────────────────────────────────

def compute_partition_stats(
    clients: dict[str, pd.DataFrame],
    results_dir: str | Path = "results",
) -> dict:
    """
    Compute and save partition statistics to results/partition_stats.json.
    """
    stats = {}
    for name, cdf in clients.items():
        total = len(cdf)
        n_default = int(cdf["target"].sum())
        n_repaid = total - n_default
        pct_default = round(n_default / total * 100, 2) if total > 0 else 0.0
        grade_dist = cdf["grade"].value_counts().to_dict() if "grade" in cdf.columns else {}
        state_top5 = cdf["addr_state"].value_counts().head(5).to_dict() if "addr_state" in cdf.columns else {}

        stats[name] = {
            "total_rows": total,
            "n_default": n_default,
            "n_repaid": n_repaid,
            "pct_default": pct_default,
            "pct_repaid": round(100 - pct_default, 2),
            "grade_distribution": grade_dist,
            "top5_states": state_top5,
        }

        print(
            f"[data_partition] {name}: {total:,} rows | "
            f"default={n_default:,} ({pct_default:.1f}%) | "
            f"repaid={n_repaid:,} ({100-pct_default:.1f}%)"
        )

    results_path = Path(results_dir) / "partition_stats.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[data_partition] Saved stats to {results_path}")

    return stats


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_partition.py <path_to_lending_club_csv>")
        print("Example: python data_partition.py ~/data/lending_club/loan.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    results_dir = Path(__file__).parent.parent / "results"

    # Day 1: Load and clean
    df_clean = load_and_clean(csv_path)
    print(f"\n[data_partition] Clean dataset: {len(df_clean):,} rows, {len(df_clean.columns)} columns")
    print(f"[data_partition] Columns: {list(df_clean.columns)}")

    # Day 2: Non-IID partition
    clients = partition_noniid(df_clean)

    # Stats + save
    stats = compute_partition_stats(clients, results_dir)

    print("\n=== PARTITION SUMMARY ===")
    for name, s in stats.items():
        print(f"  {name}: {s['total_rows']:,} rows | {s['pct_default']:.1f}% default")

    print("\n[data_partition] Day 1-2 complete. Check results/partition_stats.json for full output.")
