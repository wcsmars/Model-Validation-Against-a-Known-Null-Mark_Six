"""Exact probabilities for unordered draws of exactly ``k`` distinct items.

A conditional-Poisson distribution assigns a set S probability proportional to
``exp(sum(z[j] for j in S))``. Its normalizer is the degree-k elementary
symmetric polynomial in ``exp(z)``. Arrays use their last axis for items;
leading axes can represent batches.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.special import logsumexp

UNIFORM_LOGP = -math.log(math.comb(49, 6))


def _array(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers")
    try:
        out = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numbers") from error
    if out.ndim < 1:
        raise ValueError(f"{name} must have an item axis")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _draw_size(k: int, n: int) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, numbers.Integral):
        raise ValueError("k must be an integer")
    if not 0 <= k <= n:
        raise ValueError("k must lie between zero and the number of items")
    return int(k)


def elementary(weights, k=6):
    """Return elementary symmetric polynomials of degrees 0 through k.

    ``weights`` must be finite and nonnegative. The result has shape
    ``weights.shape[:-1] + (k + 1,)``. This raw-polynomial utility can exceed
    floating-point range; use :func:`log_probability` for log-weight scoring.
    """
    weights = _array(weights, "weights")
    k = _draw_size(k, weights.shape[-1])
    if (weights < 0).any():
        raise ValueError("weights must be nonnegative")
    result = np.zeros(weights.shape[:-1] + (k + 1,))
    result[..., 0] = 1
    with np.errstate(over="raise", invalid="raise"):
        for j in range(weights.shape[-1]):
            result[..., 1:] += weights[..., j, None] * result[..., :-1].copy()
    return result


def log_weights(q):
    """Convert Bernoulli propensities to centered conditional-Poisson logits.

    Inputs must lie in [0, 1]. The research convention clips them to
    [0.015, 0.60] before conversion, bounding influence of extreme estimates.
    The resulting exact-k inclusion marginals generally differ from ``q``;
    this conversion is a projection, not an exact moment-matching procedure.
    """
    q = _array(q, "q")
    if q.shape[-1] == 0 or ((q < 0) | (q > 1)).any():
        raise ValueError("q must have a nonempty item axis and lie in [0, 1]")
    clipped = np.clip(q, 0.015, 0.60)
    logits = np.log(clipped) - np.log1p(-clipped)
    return logits - logits.mean(axis=-1, keepdims=True)


def _center(z: np.ndarray) -> np.ndarray:
    # Adding a common log-weight does not change an exact-k distribution.
    # Centering and a log-space recurrence avoid exponentiating large logits.
    with np.errstate(over="raise", invalid="raise"):
        return z - z.max(axis=-1, keepdims=True)


def _log_elementary(z: np.ndarray, k: int) -> np.ndarray:
    result = np.full(z.shape[:-1] + (k + 1,), -np.inf)
    result[..., 0] = 0
    for j in range(z.shape[-1]):
        result[..., 1:] = np.logaddexp(
            result[..., 1:], z[..., j, None] + result[..., :-1]
        )
    return result


def log_probability(z, y, k=6):
    """Return log probabilities of binary exact-k outcomes ``y``.

    ``z`` contains finite log-weights and ``y`` contains zero/one indicators
    summing to k on the last axis. Item-axis lengths must match; leading axes
    may broadcast. The elementary-symmetric normalizer is computed exactly up
    to floating-point rounding, without enumerating the possible sets.
    """
    z, y = _array(z, "z"), _array(y, "y")
    k = _draw_size(k, z.shape[-1])
    if z.shape[-1] != y.shape[-1]:
        raise ValueError("z and y must have the same item-axis length")
    if not np.isin(y, [0.0, 1.0]).all() or not np.all(y.sum(axis=-1) == k):
        raise ValueError("y must contain binary outcomes with exactly k items")
    try:
        z, y = np.broadcast_arrays(z, y)
    except ValueError as error:
        raise ValueError("z and y batch dimensions must be broadcast-compatible") from error
    if k == 0 or k == z.shape[-1]:
        return np.zeros(z.shape[:-1])
    centered = _center(z)
    selected = np.where(y == 1, centered, 0).sum(axis=-1)
    result = selected - _log_elementary(centered, k)[..., k]
    # A deterministic set may round a few ulps above zero in the subtraction.
    return np.minimum(result, 0.0)


def marginals(z, k=6):
    """Return exact conditional-Poisson inclusion probabilities for each item.

    Forward and backward log-polynomial tables evaluate the normalizer with
    each item excluded. The returned probabilities sum to k within rounding.
    """
    z = _array(z, "z")
    n = z.shape[-1]
    k = _draw_size(k, n)
    if k == 0:
        return np.zeros_like(z)
    if k == n:
        return np.ones_like(z)
    centered = _center(z)
    shape = z.shape[:-1] + (n + 1, k + 1)
    forward = np.full(shape, -np.inf)
    backward = np.full(shape, -np.inf)
    forward[..., 0, 0] = backward[..., n, 0] = 0
    for j in range(n):
        forward[..., j + 1, :] = forward[..., j, :]
        forward[..., j + 1, 1:] = np.logaddexp(
            forward[..., j, 1:], centered[..., j, None] + forward[..., j, :-1]
        )
    for j in range(n - 1, -1, -1):
        backward[..., j, :] = backward[..., j + 1, :]
        backward[..., j, 1:] = np.logaddexp(
            backward[..., j + 1, 1:],
            centered[..., j, None] + backward[..., j + 1, :-1],
        )
    normalizer = forward[..., n, k]
    out = np.empty_like(z)
    for j in range(n):
        excluded = logsumexp(
            forward[..., j, :k] + backward[..., j + 1, :k][..., ::-1], axis=-1
        )
        out[..., j] = np.exp(np.minimum(centered[..., j] + excluded - normalizer, 0))
    return out
