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
UNIFORM_LOGP = -math.log(math.comb(49, 6))


def _array(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers")
    try:
        out = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
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


def _relative_normalizers(ordered_z: np.ndarray, k: int) -> np.ndarray:
    """Log normalizers relative to each prefix's most likely subset.

    Logits must be in descending order. For prefix j and size r, subtracting
    the energy of its first r items keeps the table between zero and log C(j,r).
    A max-centered unscaled log normalizer can still lose a factor such as
    log(2) when required low-weight items have logits near -1e20.
    """
    n = ordered_z.shape[-1]
    result = np.full(ordered_z.shape[:-1] + (n + 1, k + 1), -np.inf)
    result[..., :, 0] = 0
    for j in range(1, n + 1):
        width = min(j, k)
        # A difference beyond float range is a zero-probability branch at
        # available precision; all differences are nonpositive, never +inf.
        with np.errstate(over="ignore"):
            take = (ordered_z[..., j - 1, None] - ordered_z[..., :width]
                    + result[..., j - 1, :width])
        with np.errstate(under="ignore"):
            result[..., j, 1:width + 1] = np.logaddexp(
                result[..., j - 1, 1:width + 1], take
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
    ordered = np.sort(z, axis=-1)[..., ::-1]
    selected = np.sort(np.where(y == 1, z, -np.inf), axis=-1)[..., -k:][..., ::-1]
    # Pair selected logits with the corresponding best-k logits before
    # summation, so common huge energies cancel without erasing multiplicity.
    with np.errstate(over="ignore"):
        relative_energy = (selected - ordered[..., :k]).sum(axis=-1)
    result = relative_energy - _relative_normalizers(ordered, k)[..., -1, k]
    # A deterministic set may round a few ulps above zero in the subtraction.
    return np.minimum(result, 0.0)


def marginals(z, k=6):
    """Return exact conditional-Poisson inclusion probabilities for each item.

    A normalized log-polynomial table and backward conditional-inclusion
    recursion avoid cancellation for extreme logits. Probabilities sum to k
    within rounding.
    """
    z = _array(z, "z")
    n = z.shape[-1]
    k = _draw_size(k, n)
    if k == 0:
        return np.zeros_like(z)
    if k == n:
        return np.ones_like(z)
    order = np.argsort(z, axis=-1)[..., ::-1]
    ordered = np.take_along_axis(z, order, axis=-1)
    normalizers = _relative_normalizers(ordered, k)
    state = np.zeros(z.shape[:-1] + (k + 1,))
    state[..., k] = 1
    ordered_marginals = np.zeros_like(z)
    for j in range(n, 0, -1):
        width = min(j, k)
        with np.errstate(over="ignore"):
            log_take = (ordered[..., j - 1, None] - ordered[..., :width]
                        + normalizers[..., j - 1, :width])
        total = normalizers[..., j, 1:width + 1]
        with np.errstate(under="ignore"):
            take = np.exp(np.minimum(log_take - total, 0))
            skip = np.exp(np.minimum(normalizers[..., j - 1, 1:width + 1] - total, 0))
        included = state[..., 1:width + 1] * take
        ordered_marginals[..., j - 1] = included.sum(axis=-1)
        previous = np.zeros_like(state)
        previous[..., 0] = state[..., 0]
        previous[..., :width] += included
        previous[..., 1:width + 1] += state[..., 1:width + 1] * skip
        state = previous
    out = np.empty_like(z)
    np.put_along_axis(out, order, ordered_marginals, axis=-1)
    return out
