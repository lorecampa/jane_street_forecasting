# pylint: disable=C0103

import numpy as np
from scipy.optimize import root


def get_p(
    m_t,
    w_tm1,
    n_tm1,
):
    """
    Compute experts probabilities.
    """
    w_tilde = w_tm1 * np.exp(n_tm1 * m_t)
    return (n_tm1 * w_tilde) / np.dot(n_tm1, w_tilde)


def get_r(
    l_t,
    p_t,
):
    """
    Compute experts regrets.
    """
    return np.dot(p_t, l_t) - l_t


def get_m(
    l_tm1,
    n_tm1,
    w_tm1,
    K,
):
    """
    Compute experts regrets estimates.
    """
    alpha = get_alpha(l_tm1, n_tm1, w_tm1, K)
    return alpha - l_tm1


def get_alpha(
    l_tm1,
    n_tm1,
    w_tm1,
    K,
):
    """
    Compute alpha value to compute experts regrets estimates.
    """

    def w_tilde_tm1_k(a, k):
        return w_tm1[k] * np.exp(n_tm1[k] * (a - l_tm1[k]))

    def p_t_k(a, k):
        return (n_tm1[k] * w_tilde_tm1_k(a, k)) / np.sum(
            [n_tm1[i] * w_tilde_tm1_k(a, i) for i in range(K)]
        )

    def fun(a):
        return np.sum([p_t_k(a, k) * l_tm1[k] for k in range(K)]) - a

    return root(fun, 1)["x"]


def upd_n(
    cum_err,
    K,
):
    """
    Updates experts learning rates.
    """
    return np.minimum(np.full(K, 1 / 4), np.sqrt(np.log(K) / (1 + cum_err)))


def upd_w(
    w_tm1,
    n_tm1,
    n_t,
    r_t,
    m_t,
    K,
):
    """
    Updates experts weights.
    """
    w_t = np.power(
        w_tm1 * np.exp(n_tm1 * r_t - (n_tm1 * (r_t - m_t)) ** 2),
        n_t / n_tm1,
    )
    if np.all(w_t == np.zeros(K)):
        w_t = np.full(K, 1 / K)
    return w_t
