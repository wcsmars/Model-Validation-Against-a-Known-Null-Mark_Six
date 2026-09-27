"""Check exact set probabilities against enumeration, including batched inputs."""
import itertools
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from marksix.probability import UNIFORM_LOGP, elementary, log_probability, log_weights, marginals


def outcomes(n, k):
    rows = []
    for subset in itertools.combinations(range(n), k):
        row = np.zeros(n)
        row[list(subset)] = 1
        rows.append(row)
    return np.asarray(rows)


@pytest.mark.parametrize("n,k", [(5, 0), (5, 1), (5, 3), (5, 5), (7, 4)])
def test_distribution_matches_complete_enumeration(n, k):
    z = np.linspace(-1.7, 2.1, n)
    y = outcomes(n, k)
    raw = y @ z
    expected = raw - logsumexp(raw)
    actual = log_probability(z, y, k)
    assert_allclose(actual, expected, atol=2e-14)
    probabilities = np.exp(actual)
    assert_allclose(probabilities.sum(), 1, atol=1e-14)
    assert_allclose(marginals(z, k), probabilities @ y, atol=1e-14)
    assert_allclose(marginals(z, k).sum(), k, atol=1e-14)


def test_elementary_matches_enumerated_products_and_batching():
    weights = np.array([[0.2, 0.7, 2, 3], [0, 1, 4, 2]])
    expected = np.array([
        [sum(math.prod(row[list(s)]) for s in itertools.combinations(range(4), k))
         for k in range(5)] for row in weights
    ])
    assert_allclose(elementary(weights, k=4), expected)
    assert_allclose(elementary(np.array([]), k=0), [1])


def test_log_scores_and_marginals_are_invariant_to_large_common_shift():
    z = np.array([[-1, -0.3, 0.2, 1.1, 2], [0.1, 0.2, -0.8, 1.5, 0.4]])
    y = np.array([[1, 0, 1, 0, 0], [0, 0, 1, 1, 0]])
    assert_allclose(log_probability(z + 10000, y, 2), log_probability(z, y, 2), atol=2e-12)
    assert_allclose(marginals(z + 10000, 2), marginals(z, 2), atol=1e-12)
    assert_allclose(marginals(z, 2).sum(axis=-1), 2)


def test_extreme_log_weights_do_not_require_exponentiation():
    z = np.array([-1000, -700, 200, 600, 1000.0])
    y = outcomes(5, 2)
    probabilities = log_probability(z, y, 2)
    expected = y @ z - logsumexp(y @ z)
    assert np.isfinite(probabilities).all()
    assert_allclose(probabilities, expected, atol=1e-12)
    assert_allclose(marginals(z, 2), np.exp(expected) @ y, atol=1e-12)


def test_uniform_mark_six_reference():
    y = np.r_[np.ones(6), np.zeros(43)]
    assert_allclose(log_probability(np.zeros(49), y), UNIFORM_LOGP)
    assert_allclose(marginals(np.zeros(49)), np.full(49, 6 / 49))


def test_mark_six_scale_matches_exact_integer_polynomials():
    # Integer weights give exact elementary symmetric polynomials for n=49.
    weights = np.random.default_rng(5).integers(1, 6, size=49)
    table = [1] + [0] * 6
    for w in weights.tolist():
        for k in range(6, 0, -1):
            table[k] += w * table[k - 1]
    z = np.log(weights)
    subsets = np.random.default_rng(6).permuted(np.tile(np.arange(49), (20, 1)), axis=1)[:, :6]
    y = np.zeros((20, 49))
    np.put_along_axis(y, subsets, 1, axis=1)
    expected = [math.log(math.prod(weights[s].tolist())) - math.log(table[6]) for s in subsets]
    assert_allclose(log_probability(z, y), expected, rtol=0, atol=1e-13)
    excluded = []
    for i in range(49):
        rest = [1] + [0] * 5
        for w in np.delete(weights, i).tolist():
            for k in range(5, 0, -1):
                rest[k] += w * rest[k - 1]
        excluded.append(weights[i] * rest[5] / table[6])
    assert_allclose(marginals(z), excluded, rtol=1e-13)


def test_broadcasted_batches_match_individual_calls():
    z = np.array([[-1, 0.1, 0.4, 1.2], [0.2, 0.7, -0.2, -1.4]])
    y = outcomes(4, 2)
    scores = log_probability(z[:, None, :], y[None, :, :], 2)
    assert scores.shape == (2, 6)
    for i in range(2):
        assert_allclose(scores[i], log_probability(z[i], y, 2))


def test_log_weights_documents_clipping_and_projection():
    q = np.array([0, 0.2, 0.4, 1])
    clipped = np.clip(q, 0.015, 0.60)
    expected = np.log(clipped) - np.log1p(-clipped)
    assert_allclose(log_weights(q), expected - expected.mean())
    assert_allclose(marginals(log_weights(q), 2).sum(), 2)


@pytest.mark.parametrize("k", [-1, 5, 1.2, True])
def test_invalid_draw_sizes(k):
    for operation in (elementary, marginals):
        with pytest.raises(ValueError):
            operation(np.ones(4), k)
    with pytest.raises(ValueError):
        log_probability(np.ones(4), [1, 1, 0, 0], k)


@pytest.mark.parametrize("bad", [[0, np.nan], [0, np.inf], [0, -np.inf], [1 + 1j, 0], 2.0, [10**400, 0]])
def test_nonfinite_complex_and_scalar_inputs_rejected(bad):
    for operation in (elementary, marginals):
        with pytest.raises(ValueError):
            operation(bad, 1)
    with pytest.raises(ValueError):
        log_weights(bad)
    with pytest.raises(ValueError):
        log_probability(bad, [1, 0], 1)


@pytest.mark.parametrize("bad", [[-0.1, 0.2], [0.1, 1.1], []])
def test_invalid_propensities_rejected(bad):
    with pytest.raises(ValueError):
        log_weights(bad)


@pytest.mark.parametrize("bad_y", [[1, 1, 0], [1, 0.5, 0], [1, np.nan, 0], [1, -1, 1], [1, 0]])
def test_invalid_outcomes_rejected(bad_y):
    with pytest.raises(ValueError):
        log_probability(np.zeros(3), bad_y, 1)


def test_negative_polynomial_weights_and_incompatible_batches_rejected():
    with pytest.raises(ValueError):
        elementary([-1, 2], 1)
    with pytest.raises(ValueError):
        log_probability(np.zeros((2, 3)), np.tile([1, 0, 0], (4, 1)), 1)
