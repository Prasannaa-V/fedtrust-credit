"""
local_training.py — Local training client for FedTrust-Credit
FedTrust-Credit | Day 1 (compressed schedule)

Pure Python class — no Flower import needed since we run a custom simulation loop
(D-014: flwr[simulation] requires Ray which is incompatible with Python 3.14).

The interface mirrors Flower's NumPyClient:
  fit(global_params, config) -> (local_params, n_examples, metrics)
  evaluate(params, config)   -> (loss, n_examples, metrics)

Weight representation (D-011): prediction probability vector on training set.
Aggregation: weighted average of these vectors (FedAvg in function space).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from shap_explanation import compute_shap_vector


class FederatedClient:
    """
    Local training client. Trains LightGBM on its partition each round.
    Returns parameters (prediction vector), num_examples, and metrics dict
    (including shap_vector for the aggregator).

    Mirrors fl.client.NumPyClient interface without importing flwr.
    """

    LGB_PARAMS = {
        "objective":         "binary",
        "num_leaves":        63,
        "learning_rate":     0.03,
        "n_estimators":      300,
        "n_jobs":            -1,
        "random_state":      42,
        "verbose":           -1,
        "min_child_samples": 30,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
    }

    def __init__(
        self,
        client_id: str,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: pd.Series,
        y_test:  pd.Series,
    ):
        self.client_id   = client_id
        self.X_train     = X_train
        self.X_test      = X_test
        self.y_train     = y_train
        self.y_test      = y_test
        self.n_features  = X_train.shape[1]
        self._is_fitted  = False
        self._soft_labels: np.ndarray | None = None
        self._current_init_score: float | None = None
        self.model: lgb.LGBMClassifier | None = None

    def _predict_prob(self, X: pd.DataFrame, init_score_val: float | None) -> np.ndarray:
        raw = self.model.predict(X, raw_score=True)
        if init_score_val is not None:
            raw = raw + init_score_val
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))

    # ── Federated interface ─────────────────────────────────────────────────

    def get_parameters(self) -> list[np.ndarray]:
        """Local model's predicted probabilities on training set."""
        if not self._is_fitted:
            return [np.full(len(self.X_train), 0.5, dtype=np.float32)]
        probs = self._predict_prob(self.X_train, self._current_init_score)
        return [probs.astype(np.float32)]

    def set_parameters(self, global_params: list[np.ndarray]) -> None:
        """Receive aggregated signal from server."""
        if global_params and len(global_params[0]) == len(self.X_train):
            self._soft_labels = global_params[0].astype(np.float32)
        elif global_params:
            # Scalar global signal (weighted mean probability)
            global_mean = float(global_params[0].flat[0])
            self._soft_labels = np.full(len(self.X_train), global_mean, dtype=np.float32)

    def fit(
        self,
        global_params: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """
        Local training round.
        Returns: (local_params, num_examples, metrics)
        """
        self.set_parameters(global_params)

        # Use global soft-label signal as init_score for knowledge distillation
        init_score = None
        if self._soft_labels is not None and self._is_fitted:
            # Convert probability to log-odds for LightGBM's init_score
            p = np.clip(self._soft_labels, 1e-6, 1.0 - 1e-6)
            scalar_p = float(np.mean(p))
            self._current_init_score = float(np.log(scalar_p / (1.0 - scalar_p)))
            init_score = np.full(len(self.X_train), self._current_init_score)
        else:
            self._current_init_score = None

        # Train fresh model each round
        self.model = lgb.LGBMClassifier(**self.LGB_PARAMS)
        self.model.fit(
            self.X_train,
            self.y_train,
            init_score=init_score,
        )
        self._is_fitted = True

        # SHAP vector (D-006, D-010)
        shap_vec = compute_shap_vector(self.model, self.X_train)

        # Metrics with init_score offset applied
        y_prob = self._predict_prob(self.X_test, self._current_init_score)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = float(accuracy_score(self.y_test, y_pred))
        auc = float(roc_auc_score(self.y_test, y_prob))
        f1  = float(f1_score(self.y_test, y_pred, zero_division=0))

        print(
            f"    [{self.client_id}] acc={acc:.4f} | auc={auc:.4f} | "
            f"f1={f1:.4f} | n={len(self.X_train):,}"
        )

        return (
            self.get_parameters(),
            len(self.X_train),
            {
                "client_id":   self.client_id,
                "accuracy":    acc,
                "auc":         auc,
                "f1":          f1,
                "shap_vector": shap_vec.tolist(),
                "n_features":  self.n_features,
            },
        )

    def evaluate(
        self,
        global_params: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        """Evaluate local model on local test set."""
        if not self._is_fitted:
            return 1.0, len(self.X_test), {"accuracy": 0.0, "auc": 0.5, "f1": 0.0}
        y_prob = self._predict_prob(self.X_test, self._current_init_score)
        y_pred = (y_prob >= 0.5).astype(int)
        loss = float(np.mean((y_prob - self.y_test.values.astype(float)) ** 2))
        acc  = float(accuracy_score(self.y_test, y_pred))
        auc  = float(roc_auc_score(self.y_test, y_prob))
        f1   = float(f1_score(self.y_test, y_pred, zero_division=0))
        return loss, len(self.X_test), {"accuracy": acc, "auc": auc, "f1": f1}


# Alias for backward compatibility with any code that used FlowerClient
FlowerClient = FederatedClient
