"""
dp_accountant.py — RDP composition for the Gaussian mechanism (improvement 1)

Closed-form Rényi DP accounting (Mironov 2017, Prop. 7): one Gaussian release
with L2 sensitivity C and noise std sigma satisfies, for any order α > 1,

    RDP(α) = α · C² / (2σ²).

n independent releases compose by addition. Convert total RDP to (ε,δ)-DP via

    ε(δ) = min_α [ RDP_total(α) + log(1/δ) / (α − 1) ],

minimized here over integer orders α ∈ [2, 64].

Honest scope: per-release σ comes from the analytic Gaussian mechanism
calibrated to (ε₀,δ); composition counts one release per client per round.
SHAP releases are counted only when dp_shap_epsilon is set.
"""

import math

_ORDERS = list(range(2, 65))


def rdp_gaussian(alpha: float, sigma: float, sensitivity: float = 1.0) -> float:
    """RDP of order alpha for one Gaussian release (Mironov Prop. 7)."""
    if alpha <= 1:
        raise ValueError("RDP order alpha must exceed 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rho = sensitivity ** 2 / (2.0 * sigma ** 2)
    return alpha * rho


def compose_rdp(per_release_rdp: float, n_releases: int) -> float:
    """Additive RDP composition over n independent releases."""
    if n_releases < 1:
        raise ValueError("n_releases must be >= 1")
    return per_release_rdp * n_releases


def rdp_to_dp(total_rdp_fn, delta: float) -> float:
    """Convert a total-RDP curve to (ε,δ)-DP, minimized over orders."""
    if not 0 < delta < 1:
        raise ValueError("delta must lie in (0, 1)")
    best = math.inf
    for alpha in _ORDERS:
        eps = total_rdp_fn(alpha) + math.log(1.0 / delta) / (alpha - 1)
        best = min(best, eps)
    return float(best)


def analytic_sigma(clip: float, epsilon: float, delta: float) -> float:
    """Analytic-Gaussian-mechanism σ used by FederatedClient (D-022)."""
    return clip * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def total_epsilon(per_release_eps: float, dp_delta: float, n_releases: int,
                  target_delta: float = 1e-5, clip: float = 1.0) -> dict:
    """Composed (ε,δ)-DP budget for n Gaussian releases.

    Derives σ from the per-release analytic calibration, then composes in
    RDP space and converts back at target_delta. Returns the full ledger.
    """
    sigma = analytic_sigma(clip, per_release_eps, dp_delta)
    rho = clip ** 2 / (2.0 * sigma ** 2)
    eps_total = rdp_to_dp(lambda a: compose_rdp(a * rho, n_releases),
                          target_delta)
    return {"per_release_epsilon": per_release_eps,
            "dp_delta": dp_delta,
            "target_delta": target_delta,
            "clip": clip,
            "sigma": sigma,
            "n_releases": n_releases,
            "composed_epsilon": eps_total}
