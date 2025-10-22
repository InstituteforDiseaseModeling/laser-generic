import numba as nb
import numpy as np


# beta(a, b, size=None)
def beta(a, b):
    @nb.njit(nogil=True, cache=True)
    def _beta():
        return np.float32(np.random.beta(a, b))

    return _beta


# binomial(n, p, size=None)
def binomial(n, p):
    @nb.njit(nogil=True, cache=True)
    def _binomial():
        return np.int32(np.random.binomial(n, p))

    return _binomial


def constant_float(value):
    @nb.njit(nogil=True, cache=True)
    def _constant():
        return np.float32(value)

    return _constant


def constant_int(value):
    @nb.njit(nogil=True, cache=True)
    def _constant():
        return np.int32(value)

    return _constant


# exponential(scale=1.0, size=None)
def exponential(scale):
    @nb.njit(nogil=True, cache=True)
    def _exponential():
        return np.float32(np.random.exponential(scale))

    return _exponential


# gamma(shape, scale=1.0, size=None)
def gamma(shape, scale):
    @nb.njit(nogil=True, cache=True)
    def _gamma():
        return np.float32(np.random.gamma(shape, scale))

    return _gamma


# logistic(loc=0.0, scale=1.0, size=None)
def logistic(loc, scale):
    @nb.njit(nogil=True, cache=True)
    def _logistic():
        return np.float32(np.random.logistic(loc, scale))

    return _logistic


# lognormal(mean=0.0, sigma=1.0, size=None)
def lognormal(mean, sigma):
    @nb.njit(nogil=True, cache=True)
    def _lognormal():
        return np.float32(np.random.lognormal(mean, sigma))

    return _lognormal


# # multinomial(n, pvals, size=None)
# def multinomial(n, pvals):
#     @nb.njit(nogil=True, cache=True)
#     def _multinomial():
#         return np.int32(np.random.multinomial(n, pvals))
#
#     return _multinomial


# negative_binomial(n, p, size=None)
def negative_binomial(n, p):
    @nb.njit(nogil=True, cache=True)
    def _negative_binomial():
        return np.int32(np.random.negative_binomial(n, p))

    return _negative_binomial


# normal(loc=0.0, scale=1.0, size=None)
def normal(loc, scale):
    @nb.njit(nogil=True, cache=True)
    def _normal():
        return np.float32(np.random.normal(loc, scale))

    return _normal


# poisson(lam=1.0, size=None)
def poisson(lam):
    @nb.njit(nogil=True, cache=True)
    def _poisson():
        return np.int32(np.random.poisson(lam))

    return _poisson


# uniform(low=0.0, high=1.0, size=None)
def uniform(low, high):
    @nb.njit(nogil=True, cache=True)
    def _uniform():
        return np.float32(np.random.uniform(low, high))

    return _uniform


# weibull(a, size=None)
def weibull(a, lam):
    @nb.njit(nogil=True, cache=True)
    def _weibull():
        return np.float32(lam * np.random.weibull(a))

    return _weibull
