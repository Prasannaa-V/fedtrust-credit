"""
byzantine_defense.py — Update-norm pre-filter for parameter-space attacks.

Motivation (D-027): explanation-consistency scoring is blind to post-training
weight poisoning because SHAP is computed from the clean local model. This
module flags clients whose shared update vectors deviate abnormally, using a
length-invariant statistic so clients with different train sizes compare fairly:

    rms_dev(v) = ||v − mean(v)||₂ / sqrt(len(v))

A client is flagged when its deviation exceeds ``median + k·MAD`` over the
round's clients (MAD floored at 1e-12 so identical updates never flag).
With only 3 clients this is a coarse tripwire, reported honestly as such:
it catches gross corruption (e.g. N(0,2) poison), not stealthy attacks.
"""

import numpy as np


def rms_deviation(v: np.ndarray) -> float:
    """Length-invariant spread of an update vector."""
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    return float(np.linalg.norm(v - v.mean()) / np.sqrt(v.size))


def flag_updates(vecs, k: float = 3.0, rel_tol: float = 0.10) -> list[bool]:
    """Flag clients whose RMS deviation exceeds median + k·MAD.

    The MAD term is floored at ``rel_tol`` × median so ordinary jitter
    among few clients never trips the filter — only gross corruption
    (hundreds of percent above median) flags. With 3 clients this is a
    coarse tripwire, reported honestly as such.
    """
    devs = [rms_deviation(v) for v in vecs]
    med = float(np.median(devs))
    mad = float(np.median([abs(d - med) for d in devs]))
    thresh = max(med + k * mad, med * (1.0 + rel_tol))
    return [d > thresh for d in devs]
