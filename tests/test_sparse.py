"""Validate sparse priors, symmetry, limits, and input boundaries."""
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import beta as beta_distribution

from marksix.probability import log_weights, marginals
from marksix.sparse import P0, _log_survival, sparse_prediction, spike_slab_prediction


def training_draws(n=80):
    generator = np.random.default_rng(901)
    y = np.zeros((n, 49))
    for row in y:
        row[generator.choice(49, 6, replace=False)] = 1
    return y


@pytest.mark.parametrize("hot_only", [False, True])
def test_sparse_empty_training_preserves_symmetric_prior(hot_only):
    q, diagnostics = sparse_prediction(np.empty((0, 49)), hot_only=hot_only, return_diagnostics=True)
    assert_allclose(q, np.full(49, P0), atol=1e-14)
    assert_allclose(diagnostics["posterior_null_probability"], 0.5)
    assert_allclose(diagnostics["posterior_alternative_probabilities"], np.full(49, 0.5 / 49))
    assert_allclose(diagnostics["log_bayes_factors"], 0, atol=1e-14)
    assert_allclose(diagnostics["log_mixture_evidence_gain_over_fair"], 0, atol=1e-14)
    assert diagnostics["training_draws"] == 0


def test_spike_slab_empty_training_distinguishes_untruncated_and_hot_prior():
    empty = np.empty((0, 49))
    q, diagnostics = spike_slab_prediction(empty, return_diagnostics=True)
    assert_allclose(q, np.full(49, P0))
    assert_allclose(diagnostics["posterior_bias_probabilities"], np.full(49, 1 / 49))
    hot_q = spike_slab_prediction(empty, hot_only=True)
    assert (hot_q > P0).all()
    # The composite propensity vector is not a vector of joint inclusion moments.
    assert hot_q.sum() > 6
    assert_allclose(marginals(log_weights(hot_q)), np.full(49, P0))


@pytest.mark.parametrize("hot_only", [False, True])
def test_zero_alternative_prior_stays_uniform_despite_uneven_training(hot_only):
    y = np.tile(np.r_[np.ones(6), np.zeros(43)], (150, 1))
    q, diagnostics = sparse_prediction(y, alt_mass=0, hot_only=hot_only, return_diagnostics=True)
    assert_allclose(q, np.full(49, P0))
    assert diagnostics["posterior_null_probability"] == 1
    assert_allclose(diagnostics["posterior_alternative_probabilities"], 0)
    q = spike_slab_prediction(y, bias_probability=0, hot_only=hot_only)
    assert_allclose(q, np.full(49, P0))


@pytest.mark.parametrize("predict", [sparse_prediction, spike_slab_prediction])
@pytest.mark.parametrize("hot_only", [False, True])
def test_ball_labels_and_training_order_do_not_affect_predictions(predict, hot_only):
    y = training_draws()
    permutation = np.random.default_rng(33).permutation(49)
    q = predict(y, hot_only=hot_only)
    assert_allclose(predict(y[:, permutation], hot_only=hot_only), q[permutation], atol=1e-14)
    assert_allclose(predict(y[::-1], hot_only=hot_only), q, atol=1e-14)


@pytest.mark.parametrize("alt_mass", [0.1, 0.5, 1])
@pytest.mark.parametrize("hot_only", [False, True])
def test_single_ball_mixture_remains_coherent(alt_mass, hot_only):
    q, diagnostics = sparse_prediction(
        training_draws(), alt_mass=alt_mass, hot_only=hot_only, return_diagnostics=True
    )
    assert np.isfinite(q).all()
    assert ((q >= 0) & (q <= 1)).all()
    assert_allclose(q.sum(), 6, atol=1e-12)
    total_mass = diagnostics["posterior_null_probability"] + sum(diagnostics["posterior_alternative_probabilities"])
    assert_allclose(total_mass, 1)
    if alt_mass == 1:
        assert diagnostics["posterior_null_probability"] == 0


def test_repeated_inclusions_provide_evidence_without_encoding_ball_identity():
    y = np.tile(np.r_[np.ones(6), np.zeros(43)], (100, 1))
    q, diagnostics = sparse_prediction(y, return_diagnostics=True)
    assert q[:6].min() > q[6:].max()
    assert diagnostics["posterior_null_probability"] < 1e-20
    assert_allclose(q[:6], q[0])
    assert_allclose(q[6:], q[6])
    assert_allclose(q.sum(), 6)


@pytest.mark.parametrize("hot_only", [False, True])
def test_tiny_strength_stays_finite_when_a_ball_is_always_drawn(hot_only):
    y = np.tile(np.r_[np.ones(6), np.zeros(43)], (100, 1))
    q = sparse_prediction(y, strength=1e-100, hot_only=hot_only)
    assert np.isfinite(q).all()
    assert_allclose(q.sum(), 6)


def test_full_spike_slab_alternative_has_beta_posterior_mean():
    # With no point mass, beta-Bernoulli conjugacy gives this analytic limit.
    y = training_draws(12)
    strength = 30
    expected = (strength * P0 + y.sum(axis=0)) / (strength + len(y))
    q, diagnostics = spike_slab_prediction(
        y, strength=strength, bias_probability=1, return_diagnostics=True
    )
    assert_allclose(q, expected)
    assert_allclose(diagnostics["posterior_bias_probabilities"], 1)


def test_hot_tail_underflow_stays_finite_for_large_training_sample():
    # Zero inclusions in 10,000 trials makes the beta survival smaller than a
    # normal float; the rescaled integration fallback must still be usable.
    y = np.tile(np.r_[np.ones(6), np.zeros(43)], (10000, 1))
    q, diagnostics = sparse_prediction(y, hot_only=True, return_diagnostics=True)
    assert np.isfinite(q).all()
    assert np.isfinite(diagnostics["log_bayes_factors"]).all()
    assert np.isfinite(diagnostics["conditional_biased_ball_posterior_means"]).all()
    assert_allclose(q.sum(), 6, atol=1e-10)


@pytest.mark.parametrize("predict", [sparse_prediction, spike_slab_prediction])
@pytest.mark.parametrize("bad", [np.zeros(49), np.zeros((2, 48)), np.zeros((2, 49)), np.full((1, 49), 6 / 49), np.full((1, 49), np.nan), np.full((1, 49), np.inf), np.ones((1, 49), dtype=complex) * (1 + 1j)])
def test_invalid_training_draws_rejected(predict, bad):
    with pytest.raises(ValueError):
        predict(bad)


@pytest.mark.parametrize("predict", [sparse_prediction, spike_slab_prediction])
@pytest.mark.parametrize("strength", [0, -1, 1e9, np.nan, np.inf, True, [20], "20"])
def test_invalid_strength_rejected(predict, strength):
    with pytest.raises(ValueError):
        predict(np.empty((0, 49)), strength=strength)


@pytest.mark.parametrize("predict,keyword", [(sparse_prediction, "alt_mass"), (spike_slab_prediction, "bias_probability")])
@pytest.mark.parametrize("mass", [-0.1, 1.1, np.nan, np.inf, True, [0.5], "0.5"])
def test_invalid_prior_mass_rejected(predict, keyword, mass):
    with pytest.raises(ValueError):
        predict(np.empty((0, 49)), **{keyword: mass})


@pytest.mark.parametrize("predict", [sparse_prediction, spike_slab_prediction])
@pytest.mark.parametrize("keyword,value", [("hot_only", "false"), ("return_diagnostics", 1)])
def test_nonboolean_flags_rejected(predict, keyword, value):
    with pytest.raises(ValueError):
        predict(np.empty((0, 49)), **{keyword: value})


@pytest.mark.parametrize("hot_only", [False, True])
def test_log_bayes_factors_match_quadrature(hot_only):
    # Alternative likelihood relative to the null, integrated over the beta prior.
    y = training_draws(60)
    strength = 20
    _, diagnostics = sparse_prediction(y, strength=strength, hot_only=hot_only, return_diagnostics=True)
    prior = beta_distribution(strength * P0, strength * (1 - P0))
    lower = P0 if hot_only else 0
    expected = []
    for count in y.sum(axis=0):
        ratio = lambda p: (p / P0) ** count * ((1 - p) / (1 - P0)) ** (len(y) - count) * prior.pdf(p)
        expected.append(np.log(quad(ratio, lower, 1, epsabs=0, epsrel=1e-11)[0] / prior.sf(lower)))
    assert_allclose(diagnostics["log_bayes_factors"], expected, rtol=0, atol=1e-8)


def test_survival_fallback_matches_incomplete_beta():
    pairs = [(2.45, 117.55), (0.5, 40.0), (3.0, 1000.0), (12.45, 160.0)]
    a, b = np.array(pairs).T
    direct = _log_survival(a, b)
    with patch("marksix.sparse.betaincc", return_value=np.zeros(len(pairs))):
        integrated = _log_survival(a, b)
    assert_allclose(integrated, direct, rtol=1e-11)


def test_hot_posterior_mean_is_smooth_through_subnormal_tail():
    # Near n=5750 the untruncated survival is subnormal for a never-drawn ball.
    # E[p | p > P0] must stay above P0 and track the exponential-tail limit.
    means = []
    for n in range(5700, 5820, 15):
        y = np.zeros((n, 49))
        for i, row in enumerate(y):
            row[[(6 * i + j) % 48 for j in range(6)]] = 1
        mean = spike_slab_prediction(y, hot_only=True, bias_probability=1)[48]
        alpha, beta = 20 * P0, 20 * (1 - P0) + n
        rate = (beta - 1) / (1 - P0) - (alpha - 1) / P0
        assert 0.999 < (mean - P0) * rate < 1
        means.append(mean)
    assert np.all(np.diff(means) < 0)
