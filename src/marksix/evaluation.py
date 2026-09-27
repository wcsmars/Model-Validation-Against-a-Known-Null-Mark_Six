"""Past-only probability evaluation, with whole-draw resampling uncertainty."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .probability import UNIFORM_LOGP, log_probability, log_weights, marginals
from .sparse import sparse_prediction, spike_slab_prediction

MODEL_NAMES = ("uniform", "sparse_single_ball", "spike_slab", "online_mixture")
MIXTURE_PRIOR = np.array([0.5, 0.25, 0.25])


def walk_forward(y, *, warmup=60, reset_index=None, reset_indices=()):
    """Score t using only preceding draws from the same machine generation.

    Every component forecasts a conditional-Poisson set law. Propensity
    conversion is an explicit projection, including for the single-ball model.
    The ensemble is an arithmetic probability mixture of these three laws.
    """
    if np.iscomplexobj(y):
        raise ValueError("Draw indicators must be real")
    y = np.asarray(y, dtype=float)
    if (y.ndim != 2 or y.shape[1] != 49 or not np.isfinite(y).all()
            or not np.isin(y, (0, 1)).all() or not np.all(y.sum(axis=1) == 6)):
        raise ValueError("Expected binary draws with shape (n, 49) and six mains per row")
    if (isinstance(warmup, bool) or not isinstance(warmup, (int, np.integer))
            or not 1 <= warmup < len(y)):
        raise ValueError("warmup must leave at least one training and one evaluation draw")
    if reset_index is not None and (isinstance(reset_index, bool)
            or not isinstance(reset_index, (int, np.integer))
            or not 0 <= reset_index <= len(y)):
        raise ValueError("reset_index must lie between zero and the draw count")
    try:
        boundaries = list(reset_indices)
    except TypeError as error:
        raise ValueError("reset_indices must be a sequence of draw indices") from error
    if reset_index is not None:
        boundaries.append(reset_index)
    if any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer))
           or not 0 <= b <= len(y) for b in boundaries):
        raise ValueError("reset_indices must contain valid draw indices")
    boundaries = sorted(set(boundaries))
    indices = np.arange(warmup, len(y))
    probabilities, all_marginals, mixture_history = [], [], []
    weights = MIXTURE_PRIOR.copy()
    for t in indices:
        if t in boundaries:
            weights = MIXTURE_PRIOR.copy()
        start = max((b for b in boundaries if b <= t), default=0)
        history = y[start:t]
        z = np.stack([np.zeros(49),
                      log_weights(sparse_prediction(history, strength=20, alt_mass=0.5)),
                      log_weights(spike_slab_prediction(history, strength=20, bias_probability=1/49))])
        component_marginals = marginals(z)
        lp = log_probability(z, y[t])
        # Score the uniform law exactly; the recurrence can round it by an ulp.
        lp[0] = UNIFORM_LOGP
        mixture_lp = float(logsumexp(np.log(weights) + lp))
        probabilities.append(np.r_[lp, mixture_lp])
        all_marginals.append(np.vstack([component_marginals, weights @ component_marginals]))
        mixture_history.append(weights.copy())
        # This update takes place strictly after the target is scored.
        posterior = weights * np.exp(0.25 * (lp - lp.max()))
        posterior /= posterior.sum()
        weights = 0.99 * posterior + 0.01 * MIXTURE_PRIOR
    return {"names": MODEL_NAMES, "indices": indices,
            "log_probabilities": np.array(probabilities),
            "marginals": np.array(all_marginals),
            "mixture_weights": np.array(mixture_history)}


def holm(pvalues):
    """Holm adjustment over exactly the supplied hypothesis family."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def block_uncertainty(gains, *, seed=20260914, replicates=2000, block_length=8):
    """Circular block-bootstrap mean intervals and centered one-sided p-values.

    Resample whole draws, retaining all models on the same sampled indices.
    These are exploratory uncertainty summaries, not a prospective test.
    """
    x = np.asarray(gains, dtype=float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("gains must be a nonempty finite matrix")
    if (not isinstance(replicates, (int, np.integer)) or isinstance(replicates, bool)
            or replicates < 100):
        raise ValueError("replicates must be an integer of at least 100")
    if (not isinstance(block_length, (int, np.integer)) or isinstance(block_length, bool)
            or block_length < 1):
        raise ValueError("block_length must be a positive integer")
    n = len(x)
    if n < max(32, 4 * block_length):
        raise ValueError("Bootstrap needs at least 32 draws and four blocks")
    length = block_length
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(replicates, (n + length - 1) // length))
    indices = ((starts[..., None] + np.arange(length)) % n).reshape(replicates, -1)[:, :n]
    sampled = x[indices].mean(axis=1)
    means = x.mean(axis=0)
    lo, hi = np.quantile(sampled, [0.025, 0.975], axis=0)
    p = (1 + ((sampled - means) >= means).sum(axis=0)) / (replicates + 1)
    return lo, hi, p


def summarize(result, y, *, seed=20260914):
    gains = result["log_probabilities"] - UNIFORM_LOGP
    targets = np.asarray(y)[result["indices"]]
    brier = np.mean((result["marginals"] - targets[:, None, :]) ** 2, axis=-1)
    enough = len(gains) >= 32
    if enough:
        lo, hi, p = block_uncertainty(gains, seed=seed)
        adjusted = np.r_[1., holm(p[1:])]
    rows = []
    for j, name in enumerate(MODEL_NAMES):
        rows.append({"model": name, "draws": int(len(gains)),
                     "total_log_gain": float(gains[:, j].sum()),
                     "mean_log_gain": float(gains[:, j].mean()),
                     "mean_95pct_block_ci": [float(lo[j]), float(hi[j])] if enough else ([0., 0.] if j == 0 else None),
                     "one_sided_bootstrap_p": float(p[j]) if enough else (1. if j == 0 else None),
                     "holm_p": float(adjusted[j]) if enough else (1. if j == 0 else None),
                     "uncertainty_status": "exploratory_block_bootstrap" if enough else "unavailable_fewer_than_32_evaluation_draws",
                     "mean_brier_gain": float(np.mean(brier[:, 0] - brier[:, j]))})
    return rows
