"""
This module defines functions and classes for generating user true preference
data.

"""
import gc
import sys
import logging
from abc import ABCMeta, abstractmethod
import math
from inspect import signature

import numpy as np
import numba
from numba import njit
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from .base import BaseModel


_log = logging.getLogger(__name__)


def generate_uniform(nusers, nitems, lam):
    """
    A uniform generator

    Args:
        nusers (int): The number of users
        nitems (int): The number of items
        lam (float): The average number of items liked by a user

    Returns:
        pd.DataFrame: The user-item preference pairs.

    """
    # generate consumed items for each user uniformly over all the items.
    # prevent the number of consumed items from being greater than the total
    # number of items
    user_like_counts = np.zeros(nusers, dtype=np.int64)
    outliers = (user_like_counts > nitems) | (user_like_counts == 0)
    while any(outliers):
        # compute the total number of consumed items for each user as
        # a poisson random variable
        ndraws = np.count_nonzero(outliers)
        user_like_counts[outliers] = np.random.poisson(lam, ndraws)
        outliers = (user_like_counts > nitems) | (user_like_counts == 0)
    items = map(lambda x: np.random.choice(nitems, size=x, replace=False),
                user_like_counts)
    items = list(items)
    return records_to_frame(items)


def generate_ibp(nusers, alpha, c=1, sigma=0):
    """
    A generator of Indian Buffet Process with power law behavior

    Args:
        nusers (int): The number of users
        alpha (float): The mass parameter. It's the mean of Poisson random
            variable, and controls the number of new items tried by each
            user.
        c (float): The concentration parameter, c > -sigma is required.
            It controls the number of users that try each item.
        sigma (float): The stability exponent parameter.
            It controls the power-law behavior.

    Returns:
        pd.DataFrame: The user-item preference pairs.

    """
    from scipy.special import gammaln

    _check_ibp_params(alpha, c, sigma)
    # the number of items picked
    nitems_picked = 0
    # the number of users who tried for each item (m_k)
    npicks = np.ones(0, dtype=np.int64)
    # item list of picked items for each user
    items = []
    log_gamma_a = gammaln(1 + c)
    log_gamma_b = gammaln(c + sigma)
    for u in range(nusers):
        # pick from previously-used items with probability prob.
        # (len(prob) == nitems_picked)
        prob = (npicks - sigma) / (u + c)
        pick = np.random.binomial(1, prob, nitems_picked) > 0
        repicked_items = np.arange(nitems_picked, dtype=np.int64)[pick]
        # update item pick counts
        npicks[pick] = npicks[pick] + 1
        # pick new items
        lam = alpha * np.exp(log_gamma_a +
                             gammaln(u + c + sigma) -
                             gammaln(u + 1 + c) -
                             log_gamma_b)
        assert not np.isnan(lam)
        nnew = np.random.poisson(lam)
        new_items = np.arange(nitems_picked, nitems_picked + nnew,
                              dtype=np.int64)
        # add new items to picked ones
        if nnew > 0:
            npicks = np.hstack((npicks, np.ones(nnew, dtype=np.int64)))
            nitems_picked += nnew
            items.append(np.hstack((repicked_items, new_items)))
        else:
            items.append(repicked_items)

    return records_to_frame(items)


def _check_ibp_params(alpha, c, sigma):
    for k, v in (('alpha', alpha), ('c', c), ('sigma', sigma)):
        if v is None or np.isnan(v):
            raise ValueError(f'{k} cannot be set to None')

    if alpha <= 0:
        raise ValueError('alpha must be greater than 0')

    if c < (-sigma):
        raise ValueError('c must be greater than -sigma')

    if (sigma < 0) or (sigma >= 1):
        raise ValueError('sigma must be from 0 (included) to 1 (excluded)')


@njit(parallel=True)
def _generate_uniform_nbp(nusers, nitems, lam):
    """
    Numba parallel implementation of drawing `nitems` items uniformly for each
    user, where `nitems` is drawn from Poisson distribution with `lam`.

    Args:
        nusers (int32): The number of users
        nitems (int32): The number of items
        lam (float): The average number of items liked by a user

    Returns:
        list of numpy.array: The list of items the users like.

    """
    items = [np.empty(0, dtype=np.int32)] * nusers
    nlikes = _draw_poisson_nbp(lam, nusers, nitems)
    for i in numba.prange(nusers):
        item_ind = np.random.choice(nitems, nlikes[i], replace=False)
        items[i] = item_ind.astype(np.int32)
    return items


def generate_uniform_csr(nusers, nitems, lam):
    """
    A uniform preference generator

    Args:
        nusers (int): The number of users
        nitems (int): The number of items
        lam (float): The average number of items liked by a user

    Returns:
        scipy.sparse.csr_matrix: The user-item preference pairs.

    """
    items = _generate_uniform_nbp(nusers, nitems, lam)
    return records_to_csr(items)


@njit
def generate_ibp_nb(nusers, alpha, c=1, sigma=0, debug=False, seed=1):
    """
    A numba implementation of Indian Buffet Process with power law behavior

    Args:
        nusers (int): The number of users
        alpha (float): The mass parameter. It's the mean of Poisson random
            variable, and controls the number of new items tried by each
            user.
        c (float): The concentration parameter, c > -sigma is required.
            It controls the number of users that try each item.
        sigma (float): The stability exponent parameter.
            It controls the power-law behavior.
        debug (bool): Whether or not to use `seed` to reset to random generator.
        seed (int): The seed for the numpy random generator if `debug` is
            `True`. None value is not supported by numba.

    Returns:
        :obj:`list` of :obj:`numpy.ndarray`: The list of item preferences
            for each user.

    """
    if debug:
        np.random.seed(seed)
    # the number of items picked
    nitems_picked = 0
    # the number of users who tried for each item (m_k)
    npicks = np.ones(0, dtype=np.int64)
    # item list of picked items for each user
    items = []
    log_gamma_a = math.lgamma(1 + c)
    log_gamma_b = math.lgamma(c + sigma)
    for u in range(nusers):
        # pick from previously-used items with probability prob.
        # (len(prob) == nitems_picked)
        prob = (npicks - sigma) / (u + c)
        pick = np.empty_like(prob, dtype=np.bool_)
        for i, p in np.ndenumerate(prob):
            pick[i] = np.random.binomial(np.int(1), p) > 0
        repicked_items = np.arange(nitems_picked)[pick]
        # update item pick counts
        npicks[pick] = npicks[pick] + 1
        # pick new items
        lam = alpha * math.exp(log_gamma_a +
                               math.lgamma(u + c + sigma) -
                               math.lgamma(u + 1 + c) -
                               log_gamma_b)
        assert not np.isnan(lam)
        nnew = np.random.poisson(lam)
        new_items = np.arange(nitems_picked, nitems_picked + nnew)
        # add new items to picked ones
        if nnew > 0:
            npicks = np.hstack((npicks, np.ones(nnew, dtype=np.int64)))
            nitems_picked += nnew
            items.append(np.hstack((repicked_items, new_items)))
        else:
            items.append(repicked_items)
    return items


def _draw_poisson_rvs(nusers, alpha, c=1.0, sigma=0.0):
    from scipy.special import gammaln
    log_gamma_a = gammaln(1 + c)
    log_gamma_b = gammaln(c + sigma)
    users = np.arange(nusers)
    lam = alpha * np.exp(log_gamma_a + gammaln(users + c + sigma) -
                         gammaln(users + 1 + c) - log_gamma_b)
    nnew = np.random.poisson(lam)
    return nnew


@njit(parallel=True)
def _draw_poisson_rvs_nbp(nusers, alpha, c=1.0, sigma=0.0):
    log_gamma_a = math.lgamma(1 + c)
    log_gamma_b = math.lgamma(c + sigma)
    nnew = np.empty(nusers, dtype=np.int64)
    for u in numba.prange(nusers):
        lam = alpha * math.exp(log_gamma_a + math.lgamma(u + c + sigma) -
                               math.lgamma(u + 1 + c) - log_gamma_b)
        nnew[u] = np.random.poisson(lam)
    return nnew


def _draw_nnew_ibp(nusers, alpha, c=1.0, sigma=0.0):
    """
    This function draws new items for `nusers` users from the poisson
    distribution.

    Args:
        nusers (int): The number of users
        alpha (float): The mass parameter. It's the mean of Poisson random
            variable, and controls the number of new items tried by each
            user.
        c (float): The concentration parameter, c > -sigma is required.
            It controls the number of users that try each item.
        sigma (float): The stability exponent parameter.
            It controls the power-law behavior.

    Returns:
        numpy.ndarray: The number of new items each user will try
        numpy.ndarray: The start index
        numpy.ndarry: The end index

    """
    nnew = _draw_poisson_rvs(nusers, alpha, c, sigma)
    npicked_end = np.cumsum(nnew)
    npicked_start = npicked_end - nnew
    return nnew, npicked_start, npicked_end


@njit(parallel=True)
def _draw_nnew_ibp_nbp(nusers, alpha, c=1.0, sigma=0.0):
    nnew = _draw_poisson_rvs_nbp(nusers, alpha, c, sigma)
    npicked_end = np.cumsum(nnew)
    npicked_start = npicked_end - nnew
    return nnew, npicked_start, npicked_end


def _draw_bernoulli(prob):
    pick = np.random.binomial(1, prob) > 0
    return pick


@njit(parallel=True)
def _draw_bernoulli_nbp(prob):
    ntrial = np.int(1)
    n = len(prob)
    pick = np.empty_like(prob, dtype=np.bool_)
    for i in numba.prange(n):
        pick[i] = np.random.binomial(ntrial, prob[i]) > 0
    return pick


@njit
def generate_ibp_nbp(nusers, alpha, c=1.0, sigma=0.0, debug=False, seed=1):
    """
    A numba parallel implementation of Indian Buffet Process with power law
    behavior

    Args:
        nusers (int): The number of users
        alpha (float): The mass parameter. It's the mean of Poisson random
            variable, and controls the number of new items tried by each
            user.
        c (float): The concentration parameter, c > -sigma is required.
            It controls the number of users that try each item.
        sigma (float): The stability exponent parameter.
            It controls the power-law behavior.
        debug (bool): Whether or not to use `seed` to reset to random generator.
        seed (int): The seed for the numpy random generator if `debug` is
            `True`. None value is not supported by numba.

    Returns:
        :obj:`list` of :obj:`numpy.ndarray`: The list of item preferences
            for each user.

    """
    if debug:
        np.random.seed(seed)
    # the number of users who tried for each item (m_k)
    npicks = np.ones(0, dtype=np.int64)
    # item list of picked items for each user
    items = []
    # draw news items ahead for each user
    nnew, npicked_st, npicked_sp = _draw_nnew_ibp_nbp(nusers, alpha, c, sigma)
    for u in range(nusers):
        # pick from previously-used items with probability prob.
        # (len(prob) == nitems_picked)
        prob = (npicks - sigma) / (u + c)
        pick = _draw_bernoulli_nbp(prob)
        repicked_items = np.arange(npicked_st[u])[pick]
        # update item pick counts
        npicks[pick] = npicks[pick] + 1
        # pick new items
        new_items = np.arange(npicked_st[u], npicked_sp[u])
        # add new items to picked ones
        if nnew[u] > 0:
            npicks = np.hstack((npicks, np.ones(nnew[u], dtype=np.int64)))
            items.append(np.hstack((repicked_items, new_items)))
        else:
            items.append(repicked_items)
    return items


@njit
def _draw_users(nusers, first_uid, c=1.0, sigma=0.0):
    """
    Draws users for an item.

    Args:
        nusers (int): The total number of users.
        first_uid (int): The user id of the first user coming across the
            item.
        c (float): The concentration parameter, c > -sigma is required.
            It controls the number of users that try the item.
        sigma (float): The stability exponent parameter.
            It controls the power-law behavior.

    Returns:
        :obj:`list`: The list of user ids who tried the item.

    """
    # The user id of the first user who tried the item.
    users = [first_uid]
    npicks = 1  # The number of users who have tried the item.
    ntrial = np.int(1)
    # draw users from the sub-users.
    second_uid = first_uid + 1
    for u in range(second_uid, nusers):
        prob = (npicks - sigma) / (u + c)
        pick = np.random.binomial(ntrial, prob) > 0
        if pick:
            users.append(u)
            npicks += 1
    return np.array(users, dtype=np.int64)


# TODO: Test this function using the same first_uids and run multiple times
#  to see if the subsequent lines' mean equal to the update formula.
@njit(parallel=True)
def _draw_users_foreach(nusers, first_uids, c=1.0, sigma=0.0):
    """
    Draws users for each item.

    Args:
        nusers:
        first_uids:
        c:
        sigma:

    Returns:

    """
    nitems = len(first_uids)
    # allocate maximum memory
    # users = [np.empty(nusers, dtype=np.int64) for i in range(nitems)]
    users = [np.empty(0, dtype=np.int64)] * nitems
    for i in numba.prange(nitems):
        first_uid = first_uids[i]
        users[i] = _draw_users(nusers, first_uid, c, sigma)
    return users


def generate_ibp_itemwise(nusers, alpha, c=1.0, sigma=0.0):
    """
    The Indian Buffet Process generated item-wisely.

    Args:
        nusers:
        alpha:
        c:
        sigma:

    Returns:
        :obj:`list` of :obj:`numpy.array`: The list of users who tried for each
            item.

    """
    # draw the number of new items for each user.
    nnew = _draw_poisson_rvs_nbp(nusers, alpha, c, sigma)
    first_uids = np.repeat(np.arange(nusers), nnew)
    users = _draw_users_foreach(nusers, first_uids, c, sigma)
    _log.debug(f'generate {len(users)} items with parameters: '
               f'nusers={nusers}, alpha={alpha}, c={c}, sigma={sigma}')

    return records_to_frame(users, None, 'item', 'user').reindex(
        columns=['user', 'item'])


def generate_ibp_itemwise_csr(nusers, alpha, c=1.0, sigma=0.0):
    """
    The Indian Buffet Process generated item-wisely.

    Args:
        nusers:
        alpha:
        c:
        sigma:

    Returns:
        :obj:`list` of :obj:`numpy.array`: The list of users who tried for each
            item.

    """
    # draw the number of new items for each user.
    nnew = _draw_poisson_rvs_nbp(nusers, alpha, c, sigma)
    first_uids = np.repeat(np.arange(nusers), nnew)
    users = _draw_users_foreach(nusers, first_uids, c, sigma)
    # _log.debug(f'generate {len(users)} items with parameters: '
    #            f'nusers={nusers}, alpha={alpha}, c={c}, sigma={sigma}')

    return records_to_csr(users, is_uid=True)


def records_to_frame(data, index=None, id_label='user', record_label='item'):
    """
    Converts a sequence of records (items/users) to a pandas dataframe.

    Args:
        data (:obj:`iterable` of :obj:`array-like`): A sequence of records
        index (array-like, None): The indices of the records. If index is
            None, the index is created from 0 to nrecords.
        id_label (str): The label of the ids.
        record_label (str): The label of the records.

    Returns:
        pandas.DataFrame: The dataframe with columns `id_label` and
            `record_label`.
    """
    nrecords = [len(d) for d in data]
    if index is None:
        index = range(len(nrecords))
    ids = np.repeat(index, nrecords)
    records = np.hstack(data)
    return pd.DataFrame({id_label: ids, record_label: records})


def records_to_coo_unique(record, index=None, value=None, is_uid=False):
    """
    Converts a sequence of records (items/users) to a pandas dataframe with
    unique user-item pairs.

    Args:
        record (:obj:`iterable` of :obj:`array-like`): A sequence of records.
        index (array-like, None): The indices of the records. If index is
            None, the index is created from 0 to the length of record.
        value (:obj:`iterable` of :obj:`array-like`): A sequence of ratings.
            It should be align with data. If value is None, all the rating
            are ones.
        is_uid (bool): Whether the records are user ids or item ids.

    Returns:
        scipy.sparse.coo_matrix: The sparse rating matrix.

    """
    nrecords = [len(r) for r in record]
    if index is None:
        index = range(len(nrecords))
    ids = np.repeat(index, nrecords)
    records = np.hstack(record)
    if value is None:
        values = np.ones_like(records, dtype=np.float_)
    else:
        values = np.hstack(value)

    unique_pairs = np.column_stack((ids, records))
    del ids, records
    gc.collect()
    unique_pairs, indices = np.unique(unique_pairs, return_index=True,
                                      axis=0)
    values = values[indices]
    ids, records = unique_pairs[:, 0], unique_pairs[:, 1]
    if is_uid:
        return coo_matrix((values, (records, ids)))
    else:
        return coo_matrix((values, (ids, records)))


def records_to_coo(record, index=None, value=None, is_uid=False):
    """
    Converts a sequence of records (items/users) to a scipy.sparse.coo_matrix.

    Args:
        record (:obj:`iterable` of :obj:`array-like`): A sequence of records.
        index (array-like, None): The indices of the records. If index is
            None, the index is created from 0 to the length of record.
        value (:obj:`iterable` of :obj:`array-like`): A sequence of ratings.
            It should be align with data. If value is None, all the rating
            are ones.
        is_uid (bool): Whether the records are user ids or item ids.
    Returns:
        scipy.sparse.coo_matrix: The sparse rating matrix.
    """
    nrecords = [len(r) for r in record]
    if index is None:
        index = range(len(nrecords))
    ids = np.repeat(index, nrecords)
    records = np.hstack(record)
    if value is None:
        values = np.ones_like(records, dtype=np.float_)
    else:
        values = np.hstack(value)
    if is_uid:
        return coo_matrix((values, (records, ids)))
    else:
        return coo_matrix((values, (ids, records)))


def records_to_csr(record, index=None, value=None, is_uid=False):
    """
    Converts a sequence of records (items/users) to a scipy.sparse.coo_matrix.

    Args:
        record (:obj:`iterable` of :obj:`array-like`): A sequence of records.
        index (array-like, None): The indices of the records. If index is
            None, the index is created from 0 to the length of record.
        value (:obj:`iterable` of :obj:`array-like`, None): A sequence of
            ratings. It should have the same dimensions as record if it's
            not None. If value is None, all the rating are ones.
        is_uid (bool): Whether the records are user ids or item ids.
    Returns:
        scipy.sparse.csr_matrix: The sparse rating matrix.
    """
    nrecords = [len(r) for r in record]
    if index is None:
        index = range(len(nrecords))
    ids = np.repeat(index, nrecords)
    records = np.hstack(record)
    del index, record
    if value is None:
        values = np.ones_like(records, dtype=np.float_)
    else:
        values = np.hstack(value)
    del value
    if is_uid:
        return csr_matrix((values, (records, ids)))
    else:
        return csr_matrix((values, (ids, records)))


def generate_pf(nusers, nitems, k=100, cap=1, a=0.3, c=0.3,
                aprime=0.3, bprime=1, cprime=0.3, dprime=1):
    """
    A correlated preference model that uses Hierarchical Poisson Factorization.

    Args:
        nusers:
        nitems:
        k:
        cap: The cap of preference scale.
        a:
        c:
        aprime:
        bprime:
        cprime:
        dprime:

    Returns:
        pandas.DataFrame: The user preference

    """
    # draw preference features
    theta = _draw_hidden_features(nusers, k, a, aprime, bprime)
    # draw attribute features
    beta = _draw_hidden_features(nitems, k, c, cprime, dprime)
    # draw preferences
    lam = theta[:, :, None] * beta[:, None, :]
    z = np.random.poisson(lam)
    assert z.shape == (k, nusers, nitems)
    y = np.sum(z, axis=0)
    assert y.shape == (nusers, nitems)
    # truncate rating scale at cap
    y[y > cap] = cap
    user, item = np.nonzero(y)
    values = y[user, item]
    return pd.DataFrame({'user': user, 'item': item, 'rating': values})


def _draw_hidden_features(n, k, a, aprime, bprime):
    """
    Draws feature variables for Poisson factorization
    Args:
        n:
        k:
        a:
        aprime:
        bprime:

    Returns:

    """
    scale = bprime / aprime
    feature_scale = 1 / np.random.gamma(aprime, scale, n)
    features = np.random.gamma(a, feature_scale, size=(k, n))
    return features


@njit(parallel=True)
def _draw_multinomial_nbp(n, pvals):
    nsamples = len(n)
    npvals = pvals.shape[1]
    samples = np.empty((nsamples, npvals), dtype=np.int32)
    for i in numba.prange(nsamples):
        samples[i] = multinomial(n[i], pvals[i])
    return samples


@njit(nogil=True)
def multinomial(n, pvals):
    out = np.zeros_like(pvals, dtype=np.int32)
    npvals = len(pvals)
    psum = 1.0
    nsamples = n
    for i in range(npvals - 1):
        prob = max(min(pvals[i] / psum, 1), 0)  # bound probability
        sample = np.random.binomial(nsamples, prob)
        out[i] = sample
        nsamples -= sample
        psum -= pvals[i]
        if nsamples <= 0:
            break
    out[-1] = nsamples
    return out


@njit(parallel=True)
def _non_zero_nbp(a):
    """Returns indices and values of nonzero entries in a 2-d array"""

    row, col = a.nonzero()
    nrows = len(row)
    a_nonzeros = np.empty(nrows, dtype=a.dtype)
    for i in numba.prange(nrows):
        a_nonzeros[i] = a[row[i], col[i]]
    return row.astype(np.int32), col.astype(np.int32), a_nonzeros


@njit(parallel=True)
def _draw_lda_user_item(n, beta, theta):
    # draw topic counts for each of k topics for each user's n items
    nz = _draw_multinomial_nbp(n, theta)  # (nusers, k)
    u, z, nz = _non_zero_nbp(nz)  # users, topics, topic counts
    nrows = len(z)
    items = [np.empty(0, dtype=np.int32)] * nrows
    ratings = [np.empty(0, dtype=np.int32)] * nrows
    for i in numba.prange(nrows):
        zbeta = beta[z[i]]  # topic vector
        w = multinomial(nz[i], zbeta)
        item = w.nonzero()[0].astype(np.int32)
        items[i] = item
        ratings[i] = w[item].astype(np.int32)
    return u, items, ratings


@njit(parallel=True)
def _draw_poisson_nbp(lam, size, cap):
    out = np.empty(size, dtype=np.int32)
    for i in numba.prange(size):
        sample = np.random.poisson(lam)
        while (sample > cap) | (sample == 0):
            sample = np.random.poisson(lam)
        out[i] = sample
    return out


def generate_lda_nbp(nusers, nitems, k, lam, a, b):
    """
    Generates lda preferences.

    Args:
        nusers (int):
        nitems (int):
        k:
        lam:
        a:
        b:

    Returns:
        numpy.array: The user ids. User id can have duplicates if the user
            likes multiple topics.
        :obj:`list` of :obj:`numpy.array`: The Item ids for each user.
        :obj:`list` of :obj:`numpy.array`: The preference values for each item.

    """
    # draw the number of preferences for each user
    n = _draw_poisson_nbp(lam, nusers, nitems)
    # draw priors
    alpha = np.full(k, a)
    pbeta = np.full(nitems, b)
    # draw item latent features
    beta = np.random.dirichlet(pbeta, k)  # (k, nitems)
    # draw user latent features
    theta = np.random.dirichlet(alpha, nusers)  # (nusers, k)
    return _draw_lda_user_item(n, beta, theta)


def generate_lda_csr(nusers, nitems, k, lam, a, b):
    users, items, ratings = generate_lda_nbp(nusers, nitems, k, lam, a, b)
    # user item pairs can have duplicates
    return records_to_csr(items, users, ratings)


def generate_lda_df(nusers, nitems, k, lam, a, b):
    users, items, ratings = generate_lda_nbp(nusers, nitems, k, lam, a, b)
    pref = records_to_frame(items, users)
    del users, items
    gc.collect()
    pref['rating'] = np.hstack(ratings)
    return pref.groupby(['user', 'item'], as_index=False).sum()


def generate_lda(nusers, nitems, k, lam, a, b):
    # _log.debug(f'generating lda with parameters '
    #            f'{inspect.getargvalues(inspect.currentframe()).locals}')
    # draw the number of preferences for each user
    n = np.random.poisson(lam, nusers)
    outliers = (n > nitems) | (n == 0)
    while any(outliers):
        # compute the total number of consumed items for each user as
        # a poisson random variable
        ndraws = np.count_nonzero(outliers)
        n[outliers] = np.random.poisson(lam, ndraws)
        outliers = (n > nitems) | (n == 0)
    # draw priors
    alpha = np.full(k, a)
    pbeta = np.full(nitems, b)
    # draw item latent features
    beta = np.random.dirichlet(pbeta, k)  # (k, nitems)
    # draw user latent features
    theta = np.random.dirichlet(alpha, nusers)  # (nusers, k)
    # draw topic counts for each of k topics for each user's n items
    nz = np.array(list(map(np.random.multinomial, n, theta)))  # (nusers, k)
    u, z = np.nonzero(nz)
    nz = nz[u, z]
    # slice topic vectors for each user
    zbeta = beta[z]  # unique topics for each user
    # draw items, items can have duplicates
    w = np.array(list(map(np.random.multinomial, nz, zbeta)))
    it, items = np.nonzero(w)
    ratings = w[it, items]
    users = u[it]
    pref = pd.DataFrame({'user': users, 'item': items, 'rating': ratings})
    return pref.groupby(['user', 'item'], as_index=False).sum()


class PreferenceModel(BaseModel, metaclass=ABCMeta):
    """
    The base class for preference models.

    """

    @abstractmethod
    def generate(self):
        """
        Generate user true preferences using parameters initialized in the
        initializer.

        Returns:
            pandas.DataFrame: The user true preferences with columns: user,
            item, and/or rating.

        """
        raise NotImplementedError()


class UniformPreference(PreferenceModel):
    """
    The uniform preference model that generates the user's binary preference
    on each item with an equal probability.

    """

    def __init__(self, nusers, nitems, lam):
        """
        The initialization. The parameters define the size and density of
        the user preference matrix.

        Args:
            nusers (int): The number of users
            nitems (int): The number of items
            lam (float): The average number of items liked by a user

        """
        self.nusers = nusers
        self.nitems = nitems
        self.lam = lam

    def generate(self):
        """
        A uniform generator

        Returns:
            pandas.DataFrame: The user-item preference pairs.

        """
        return generate_uniform(self.nusers, self.nitems, self.lam)


class IndianBuffetProcess(PreferenceModel):
    """
    The preference model that uses the Indian Buffet Process (IBP) with
    power law behavior.

    """

    def __init__(self, nusers, alpha, c=1, sigma=0):
        """
        The initialization of indian buffet process model.

        Setting `c`=1 and `sigma`=0 makes the model be reduced to indian buffet
        process without power law behavior. For indian buffet process,
        the average number of liked user-item pairs can be computed by
        `nusers` * `alpha`. The average number of items can be approximated
        by alpha * (log(`nusers`) + `gamma`), where `gamma` is Euler's constant.

        Args:
            nusers (int): The number of users
            alpha (float): The mass parameter. It's the mean of Poisson random
                variable, and controls the number of new items tried by each
                user.
            c (float): The concentration parameter, c > -sigma is required.
                It controls the number of users that try each item.
            sigma (float): The stability exponent parameter.
                It controls the power-law behavior.

        """
        _check_ibp_params(alpha, c, sigma)
        self.nusers = nusers
        self.alpha = alpha
        self.c = c
        self.sigma = sigma

    def set_params(self, **kwargs):
        alpha = kwargs.get('alpha', self.alpha)
        c = kwargs.get('c', self.c)
        sigma = kwargs.get('sigma', self.sigma)
        _check_ibp_params(alpha, c, sigma)
        super().set_params(**kwargs)

    def generate(self):
        """

        Returns:
            pandas.DataFrame: The user-item preference pairs.

        """
        return generate_ibp_itemwise(self.nusers, self.alpha,
                                     self.c, self.sigma)


class LatentDirichletAllocation(PreferenceModel):
    """
    Latent Dirichlet Allocation using symmetric dirichlet distribution.

    """

    def __init__(self, nusers, nitems, k, lam, a, b):
        """

        Args:
            nusers (int): The number of users
            nitems (int): The number of items
            k (int): The number of latent features.
            lam (float): The mean of Poisson distribution which produces user
                profile size.
            a (float): The alpha parameter of dirichlet distribution. a > 0.
            b (float): The beta parameter of dirichlet distribution. b > 0.

        """
        local_vars = locals()
        sig = signature(self.__init__)
        for param in sig.parameters.values():
            pname = param.name
            setattr(self, pname, local_vars.get(pname))

    def generate(self):
        pref = generate_lda_df(self.nusers, self.nitems, self.k, self.lam,
                               self.a, self.b)
        return pref.groupby(['user', 'item'], as_index=False).sum()


class UniformPreferenceCSR(PreferenceModel):
    """
    The uniform preference model that generates the user's binary preference
    on each item with an equal probability.

    """

    def __init__(self, nusers, nitems, lam):
        """
        The initialization. The parameters define the size and density of
        the user preference matrix.

        Args:
            nusers (int): The number of users
            nitems (int): The number of items
            lam (float): The average number of items liked by a user

        """
        self.nusers = nusers
        self.nitems = nitems
        self.lam = lam

    def generate(self):
        """
        A uniform generator

        Returns:
            scipy.sparse.csr_matrix: The user-item preference pairs.

        """
        return generate_uniform_csr(self.nusers, self.nitems, self.lam)


class LatentDirichletAllocationCSR(PreferenceModel):
    """
    Latent Dirichlet Allocation using symmetric dirichlet distribution.

    """

    def __init__(self, nusers, nitems, k, lam, a, b):
        """

        Args:
            nusers (int): The number of users
            nitems (int): The number of items
            k (int): The number of latent features.
            lam (float): The mean of Poisson distribution which produces user
                profile size.
            a (float): The alpha parameter of dirichlet distribution. a > 0.
            b (float): The beta parameter of dirichlet distribution. b > 0.

        """
        local_vars = locals()
        sig = signature(self.__init__)
        for param in sig.parameters.values():
            pname = param.name
            setattr(self, pname, local_vars.get(pname))

    def generate(self):
        pref = generate_lda_csr(self.nusers, self.nitems, self.k, self.lam,
                                self.a, self.b)
        # sum duplicate entries
        pref.sum_duplicates()
        return pref


class IndianBuffetProcessCSR(PreferenceModel):
    """
    The preference model that uses the Indian Buffet Process (IBP) with
    power law behavior.

    """

    def __init__(self, nusers, alpha, c=1, sigma=0):
        """
        The initialization of indian buffet process model.

        Setting `c`=1 and `sigma`=0 makes the model be reduced to indian buffet
        process without power law behavior. For indian buffet process,
        the average number of liked user-item pairs can be computed by
        `nusers` * `alpha`. The average number of items can be approximated
        by alpha * (log(`nusers`) + `gamma`), where `gamma` is Euler's constant.
        For indian buffet process with power law behavior (sigma == 0),
        the average number of items can be approximated by alpha * `nusers` ^
        `sigma` for the large number of items.

        Args:
            nusers (int): The number of users
            alpha (float): The mass parameter. It's the mean of Poisson random
                variable, and controls the number of new items tried by each
                user. alpha > 0.
            c (float): The concentration parameter, c > -sigma is required.
                It controls the number of users that try each item.
            sigma (float): The stability exponent parameter.
                It controls the power-law behavior. sigma belongs to [0, 1).

        """
        _check_ibp_params(alpha, c, sigma)
        self.nusers = nusers
        self.alpha = alpha
        self.c = c
        self.sigma = sigma

    def set_params(self, **kwargs):
        alpha = kwargs.get('alpha', self.alpha)
        c = kwargs.get('c', self.c)
        sigma = kwargs.get('sigma', self.sigma)
        _check_ibp_params(alpha, c, sigma)
        super().set_params(**kwargs)

    def generate(self):
        """

        Returns:
            scipy.sparse.csr_matrix: The user-item preference pairs.

        """
        return generate_ibp_itemwise_csr(self.nusers, self.alpha,
                                         self.c, self.sigma)
