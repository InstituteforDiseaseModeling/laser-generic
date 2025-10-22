# """
# Copilot prompt:
# Please write tests for each of the distributions in laser_generic.distributions comparing the results of the functions there with appropriate parameterization - see distributions.ipynb for sample values - and a KS test.
# There should be a shared Numba compiled function - again, see distributions.ipynb - that returns multiple samples from the parameterized distribution.
# +distributions.ipynb
# +distributions.py
# +test_distributions.py
# """

import unittest
from itertools import product

import numba as nb
import numpy as np
from scipy.stats import beta as beta_ref
from scipy.stats import binom
from scipy.stats import expon
from scipy.stats import gamma as gamma_ref
from scipy.stats import ks_2samp
from scipy.stats import logistic as logistic_ref
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import uniform as uniform_ref
from scipy.stats import weibull_min

import laser_generic.distributions as dist


# Shared Numba sampling functions
@nb.njit(parallel=True, nogil=True, cache=True)
def sample_float(fn, count):
    result = np.empty(count, dtype=np.float32)
    for i in nb.prange(count):
        result[i] = fn()
    return result


@nb.njit(parallel=True, nogil=True, cache=True)
def sample_int(fn, count):
    result = np.empty(count, dtype=np.int32)
    for i in nb.prange(count):
        result[i] = fn()
    return result


NSAMPLES = 100_000
KS_THRESHOLD = 0.02  # Acceptable KS statistic for similarity


class TestDistributions(unittest.TestCase):
    def test_beta(self):
        params = [(0.5, 0.5), (5.0, 1.0), (1.0, 3.0), (2.0, 2.0), (2.0, 5.0)]
        for a, b in params:
            fn = dist.beta(a, b)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = beta_ref.rvs(a, b, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Beta({a},{b}) KS={stat}"

    def test_binomial(self):
        params = [(20, 0.5), (20, 0.7), (40, 0.5)]
        for n, p in params:
            fn = dist.binomial(n, p)
            samples = sample_int(fn, NSAMPLES)
            ref_samples = binom.rvs(n, p, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Binomial({n},{p}) KS={stat}"

    def test_constant_float(self):
        values = [0.0, 1.0, -1.0, 3.14159, 2.71828]
        for value in values:
            fn = dist.constant_float(value)
            samples = sample_float(fn, NSAMPLES)
            assert np.all(samples == np.float32(value)), f"Constant Float({value}) failed"

    def test_constant_int(self):
        values = [0, 1, -1, 42, 100]
        for value in values:
            fn = dist.constant_int(value)
            samples = sample_int(fn, NSAMPLES)
            assert np.all(samples == np.int32(value)), f"Constant Int({value}) failed"

    def test_exponential(self):
        params = [0.5, 1.0, 1.5]
        for lam in params:
            scale = 1 / lam
            fn = dist.exponential(scale)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = expon.rvs(scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Exponential({scale}) KS={stat}"

    def test_gamma(self):
        params = [(1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (5.0, 1.0), (9.0, 0.5), (7.5, 1.0), (0.5, 1.0)]
        for shape, scale in params:
            fn = dist.gamma(shape, scale)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = gamma_ref.rvs(shape, scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Gamma({shape},{scale}) KS={stat}"

    def test_logistic(self):
        params = [(5, 2), (9, 3), (9, 4), (6, 2), (2, 1)]
        for loc, scale in params:
            fn = dist.logistic(loc, scale)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = logistic_ref.rvs(loc=loc, scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Logistic({loc},{scale}) KS={stat}"

    def test_lognormal(self):
        params = [(0, 1), (0, 0.5), (0, 0.25)]
        for mean, sigma in params:
            fn = dist.lognormal(mean, sigma)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = lognorm.rvs(sigma, scale=np.exp(mean), size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Lognormal({mean},{sigma}) KS={stat}"

    def test_negative_binomial(self):
        params = product([1, 2, 3, 4, 5], [1 / 2, 1 / 3, 1 / 4, 1 / 5])
        for r, p in params:
            fn = dist.negative_binomial(r, p)
            samples = sample_int(fn, NSAMPLES)
            ref_samples = np.random.negative_binomial(r, p, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Negative Binomial({r},{p}) KS={stat}"

    def test_normal(self):
        params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]
        for mu, sigmasq in params:
            sigma = np.sqrt(sigmasq)
            fn = dist.normal(mu, sigma)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = norm.rvs(loc=mu, scale=sigma, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Normal({mu},{sigma}) KS={stat}"

    def test_poisson(self):
        params = [1, 4, 10]
        for lam in params:
            fn = dist.poisson(lam)
            samples = sample_int(fn, NSAMPLES)
            ref_samples = poisson.rvs(mu=lam, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Poisson({lam}) KS={stat}"

    def test_uniform(self):
        params = [(0.0, 1.0), (0.25, 1.25), (0.0, 2.0), (-1.0, 1.0), (2.71828, 3.14159), (1.30, 4.20)]
        for low, high in params:
            fn = dist.uniform(low, high)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = uniform_ref.rvs(loc=low, scale=high - low, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Uniform({low},{high}) KS={stat}"

    def test_weibull(self):
        params = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (5.0, 1.0)]
        for a, lam in params:
            fn = dist.weibull(a, lam)
            samples = sample_float(fn, NSAMPLES)
            ref_samples = weibull_min.rvs(a, scale=lam, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Weibull({a},{lam}) KS={stat}"


if __name__ == "__main__":
    unittest.main()
