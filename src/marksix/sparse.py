"""Multiplicity-aware hypotheses in which very few numbered balls are biased.

The single-ball mixture is a coherent Bayesian model for six distinct mains:
the null chooses a uniform subset; alternative j includes ball j with probability
p_j and chooses all other balls uniformly conditional on that inclusion. Its
binomial inclusion likelihood is exact despite dependence within a six-ball draw.

The functions receive *training draws only*. Call with y[regime_start:t] for a
machine reset. No ball number, observed current result, or fitted hyperparameter
is embedded here. Converting returned inclusion probabilities to conditional-Poisson weights
is an approximation and does not preserve this exact mixture law.
"""
from __future__ import annotations

import numbers

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, betaincc, expit, logsumexp

N_BALLS = 49
DRAW_SIZE = 6
P0 = DRAW_SIZE / N_BALLS


def _validate(y, strength: float, mass: float) -> tuple[np.ndarray, int, float, float]:
    if np.iscomplexobj(y):
        raise ValueError("Training draws must contain real indicators")
    try:
        y = np.asarray(y, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Training draws must contain real indicators") from error
    if y.ndim != 2 or y.shape[1] != N_BALLS:
        raise ValueError("Training draws must have shape (n, 49)")
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Draw indicators must be finite zeros and ones")
    if not np.all(y.sum(axis=1) == DRAW_SIZE):
        raise ValueError("Each training draw must contain exactly six distinct main balls")
    # Above 1e8, beta-function differences lose more than about 1e-7 nats;
    # far below 1e-300 the prior parameters leave the normal float range.
    # Range checks also reject NaN and infinity without converting to float.
    if (isinstance(strength, (bool, np.bool_)) or not isinstance(strength, numbers.Real)
            or not 1e-300 <= strength <= 1e8):
        raise ValueError("strength must lie between 1e-300 and 1e8")
    if (isinstance(mass, (bool, np.bool_)) or not isinstance(mass, numbers.Real)
            or not 0 <= mass <= 1):
        raise ValueError("Prior alternative probability must lie in [0, 1]")
    return y.sum(axis=0), len(y), float(strength), float(mass)


def _validate_flags(hot_only, return_diagnostics):
    if not isinstance(hot_only, (bool, np.bool_)):
        raise ValueError("hot_only must be boolean")
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("return_diagnostics must be boolean")


def _log_survival(a, b, threshold=P0) -> np.ndarray:
    """Log P(Beta(a,b)>threshold), including rare survival underflow.

    The ordinary calculation uses the complement incomplete beta directly.
    For extreme tails, integrate a rescaled beta density from the threshold;
    the scale puts its boundary-decay width near one and avoids exp(-1000).
    Values below 1e-300 also take the integral path: subnormal results keep
    too few bits, and SciPy 1.13 loses about six digits just above them.
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    survival = betaincc(a, b, threshold)
    result = np.full(a.shape, -np.inf, dtype=float)
    valid = survival >= 1e-300
    result[valid] = np.log(survival[valid])
    for index in zip(*np.where(~valid)) if a.ndim else ([()] if not valid else []):
        aa, bb = float(a[index]), float(b[index])
        # Underflow implies a decreasing upper tail for the beta priors here.
        rate = max((bb-1)/(1-threshold)-(aa-1)/threshold, 1.0)
        width = 1/rate
        log_peak = (aa-1)*np.log(threshold)+(bb-1)*np.log1p(-threshold)-betaln(aa, bb)
        limit = min((1-threshold)/width, 100.0)

        def relative_density(u):
            x = threshold + width*u
            if x >= 1:
                return 0.0
            relative = ((aa-1)*np.log(x/threshold)
                        +(bb-1)*np.log((1-x)/(1-threshold)))
            return np.exp(relative)

        integral, _ = quad(relative_density, 0, limit, epsabs=1e-12, epsrel=1e-10)
        if integral <= 0:
            raise FloatingPointError("Could not resolve truncated-beta survival probability")
        result[index] = log_peak + np.log(width*integral)
    return result


def _evidence(counts: np.ndarray, n: int, strength: float, hot_only: bool):
    alpha, beta = strength*P0, strength*(1-P0)
    # Grouping n-counts first keeps post_beta positive when beta is tiny.
    post_alpha, post_beta = alpha+counts, beta+(n-counts)
    # Combinatorial terms in the binomial count law cancel between models.
    null_log_likelihood = counts*np.log(P0)+(n-counts)*np.log1p(-P0)
    log_bf = betaln(post_alpha, post_beta)-betaln(alpha, beta)-null_log_likelihood
    posterior_mean = post_alpha/(post_alpha+post_beta)
    if hot_only:
        prior_log_survival = float(_log_survival(alpha, beta))
        posterior_log_survival = _log_survival(post_alpha, post_beta)
        log_bf = log_bf + posterior_log_survival-prior_log_survival
        # E[p 1(p>P0)] = a/(a+b) * P(Beta(a+1,b)>P0).
        posterior_mean = posterior_mean*np.exp(
            _log_survival(post_alpha+1, post_beta)-posterior_log_survival)
    return log_bf, posterior_mean


def sparse_prediction(
    y,
    strength: float = 20,
    hot_only: bool = False,
    alt_mass: float = 0.5,
    return_diagnostics: bool = False,
):
    """Predict inclusion probabilities under null + 49 single-ball alternatives.

    The fixed prior assigns 1-alt_mass to a completely fair draw and alt_mass/49
    to each possible biased-ball identity. Alternative probability has a beta
    prior centered at 6/49 with concentration strength. hot_only conditions that
    beta prior on p>6/49, including its normalizing constant in the Bayes factor.

    Under alternative j, all remaining balls have inclusion (6-p_j)/48. The
    posterior model average therefore has exactly six expected main inclusions.
    Return q, or (q, diagnostics) if return_diagnostics=True.
    """
    _validate_flags(hot_only, return_diagnostics)
    counts, n, strength, alt_mass = _validate(y, strength, alt_mass)
    log_bf, posterior_mean = _evidence(counts, n, strength, hot_only)
    null_log_mass = np.log1p(-alt_mass) if alt_mass < 1 else -np.inf
    alternative_log_mass = np.log(alt_mass/N_BALLS) if alt_mass > 0 else -np.inf
    log_model_mass = np.r_[null_log_mass, alternative_log_mass+log_bf]
    log_evidence = float(logsumexp(log_model_mass))
    weights = np.exp(log_model_mass-log_evidence)
    null_weight, alternative_weights = weights[0], weights[1:]
    others = (DRAW_SIZE-posterior_mean)/(N_BALLS-1)
    common = null_weight*P0+alternative_weights@others
    probabilities = common+alternative_weights*(posterior_mean-others)
    if not return_diagnostics:
        return probabilities
    return probabilities, {
        "training_draws": n,
        "strength": float(strength),
        "hot_only": bool(hot_only),
        "prior_alternative_mass": float(alt_mass),
        "prior_probability_per_biased_ball": float(alt_mass/N_BALLS),
        "posterior_null_probability": float(null_weight),
        "posterior_alternative_probabilities": alternative_weights.tolist(),
        "log_bayes_factors": log_bf.tolist(),
        "conditional_biased_ball_posterior_means": posterior_mean.tolist(),
        "log_mixture_evidence_gain_over_fair": log_evidence,
        "probability_interpretation": "Exact posterior inclusion moments of a mutually exclusive single-biased-ball mixture; CP conversion is approximate.",
    }


def spike_slab_prediction(
    y,
    strength: float = 20,
    hot_only: bool = False,
    bias_probability: float = 1/49,
    return_diagnostics: bool = False,
):
    """Exploratory per-ball spike/slab shrinkage with fixed sparsity prior.

    Each one-ball inclusion likelihood is exact, but treating the 49 posterior
    probabilities independently is a composite approximation: six-main outcomes
    make ball indicators dependent. Returned q are Bernoulli propensity inputs,
    not jointly coherent inclusion moments, and need not sum to six. For set
    scoring, convert them to an exact-six conditional-Poisson distribution
    with :func:`marksix.probability.log_weights`.
    No empirical-Bayes tuning is performed on the observed sample.
    """
    _validate_flags(hot_only, return_diagnostics)
    counts, n, strength, bias_probability = _validate(y, strength, bias_probability)
    log_bf, posterior_mean = _evidence(counts, n, strength, hot_only)
    if bias_probability == 0:
        posterior_bias = np.zeros(N_BALLS)
    elif bias_probability == 1:
        posterior_bias = np.ones(N_BALLS)
    else:
        prior_log_odds = np.log(bias_probability)-np.log1p(-bias_probability)
        posterior_bias = expit(prior_log_odds+log_bf)
    probabilities = P0+posterior_bias*(posterior_mean-P0)
    if not return_diagnostics:
        return probabilities
    return probabilities, {
        "training_draws": n, "strength": float(strength), "hot_only": bool(hot_only),
        "prior_bias_probability_per_ball": float(bias_probability),
        "posterior_bias_probabilities": posterior_bias.tolist(),
        "log_bayes_factors": log_bf.tolist(),
        "conditional_biased_ball_posterior_means": posterior_mean.tolist(),
        "probability_interpretation": "Independent composite spike/slab propensity; project through exact-six CP distribution before scoring.",
    }
