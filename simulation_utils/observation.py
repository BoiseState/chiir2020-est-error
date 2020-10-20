"""
The module that defines different sampling functions to generate observation data.

"""
import logging
from abc import ABCMeta, abstractmethod

import numba
import numpy as np
from numba import njit
from scipy.stats import pareto

from simulation_utils.preference import multinomial, records_to_csr
from .base import BaseModel

_logger = logging.getLogger(__name__)


class ObservationModel(BaseModel, metaclass=ABCMeta):
    """
    The base class for observation sampling models.

    """

    @abstractmethod
    def sample(self, preferences):
        """
        Sample observations from the given user preferences
        Args:
            preferences (pandas.DataFrame, scipy.sparse.coo_matrix): The data
            frame containing user-item pairs. It usually has columns: user,
            item and/or rating.

        Returns:
            pandas.DataFrame, scipy.spars.coo_matrix: The sampled user
            consumption data with columns: user, item, and/or rating.

        """
        raise NotImplementedError()


class UniformObservation(ObservationModel):
    """
    The uniform observation model that samples each item with equal
    probability for each user.

    """

    def __init__(self, dist_func, cap=True):
        """

        Args:
            frac (float): The fraction of items to be sampled for each user.

        """
        self.dist_func = dist_func
        self.cap = cap

    def sample(self, preferences, **kwargs):
        self.cap = kwargs.pop('cap', self.cap)
        return sample_uniform_n(preferences, self.dist_func, self.cap, **kwargs)


class UniformObservationCSR(ObservationModel):
    """
    The uniform observation model that samples each item with equal
    probability for each user.

    """

    def __init__(self, dist_func):
        """

        Args:
            dist_func (callable): The function to draw the profile size.

        """
        self.dist_func = dist_func

    def sample(self, preferences):
        items, ratings = sample_uniform_csr_nbp(preferences, self.dist_func)
        return records_to_csr(items, value=ratings)


class PopularityObservation(ObservationModel):
    """
    The popularity-based observation model.

    This model draws a user profile size n from a given distribution for each
    user, then n items are drew based on their popularity.

    """

    def __init__(self, dist_func, cap=True):
        """

        Args:
            dist_func (Callable): The distribution function used to draw user
                profile size from. Any numpy.random distributions or
                BaseModel with `__call__` implemented can be used.

        """
        self.dist_func = dist_func
        self.cap = cap

    def sample(self, preferences, **kwargs):
        """

        Args:
            preferences (pandas.DataFrame): The user preferences with columns:
                user, item, and/or rating

        Returns:
            pandas.DataFrame: The sampled user consumption data with columns:
                user, item, and/or rating

        """
        self.cap = kwargs.pop('cap', self.cap)
        return sample_popular_n(preferences, self.dist_func, self.cap, **kwargs)


class PopularityObservationCSR(ObservationModel):
    """
    The popularity-based observation model.

    This model draws a user profile size n from a given distribution for each
    user, then n items are drew based on their popularity.

    """

    def __init__(self, dist_func):
        """

        Args:
            dist_func (Callable): The distribution function used to draw user
                profile size from.

        """
        self.dist_func = dist_func

    def sample(self, preferences):
        items, ratings = sample_popular_csr_nbp(preferences, self.dist_func)
        return records_to_csr(items, value=ratings)


def sample_uniform(ratings, frac):
    col_names = ratings.columns
    ds = ratings.rename(columns={col_names[0]: 'user',
                                 col_names[1]: 'item'})
    return (ds.groupby('user')
            .apply(lambda x: x.sample(frac=frac))
            .reset_index(drop=True))


def sample_popular(ratings, frac):
    col_names = ratings.columns
    ds = ratings.rename(columns={col_names[0]: 'user', col_names[1]: 'item'})
    popularity = ds[['user', 'item']].groupby('item', as_index=False).count()
    popularity['prob'] = popularity['user'] / popularity['user'].sum()
    popularity.drop('user', axis=1, inplace=True)
    merged = ds.merge(popularity, on='item', how='left')
    return (merged.groupby('user')
            .apply(lambda x: x.sample(frac=frac, weights='prob'))
            .reset_index(drop=True)
            .drop('prob', axis=1))


def sample_uniform_n(ratings, dist_func, use_cap=True, **kwargs):
    col_names = ratings.columns
    if not {'user', 'item'}.issubset(col_names):
        ratings = ratings.rename(columns={col_names[0]: 'user',
                                          col_names[1]: 'item'})
    # compute user preference stats
    pref_stats = (ratings[['user', 'item']].groupby('user', as_index=False)
                  .count())
    nusers = len(pref_stats)
    npref = pref_stats['item'].values
    # sample user profile size
    nprofile = _generate_profile_size(nusers, npref, dist_func, use_cap,
                                      **kwargs)
    pref_stats['nprofile'] = nprofile
    pref_stats.drop('item', axis=1, inplace=True)   # columns: user, nprofile
    pref_stats_dict = dict(pref_stats.values)
    return (ratings.groupby('user')
            .apply(lambda x: x.sample(n=pref_stats_dict.get(x.name)))
            .reset_index(drop=True))


def sample_uniform_coo(coo, dist_func):
    """

    Args:
        coo (scipy.sparse.coo_matrix): The sparse rating matrix.
        dist_func (callable): The distribution function that generates user
            profile size. It takes one array that caps profile size for each
            user.

    Returns:
        scipy.sparse.coo_matrix: The sampled matrix.

    """
    csr = coo.tocsr()
    # get user profile size.
    caps = csr.getnnz(axis=1)
    nprofile = dist_func(caps)
    nrows, ncols = csr.shape
    indices = csr.indices
    indptr = csr.indptr
    data = csr.data
    samples = [np.empty(0, dtype=np.int32)] * nrows
    ratings = [np.empty(0, dtype=np.float_)] * nrows
    for i in range(nrows):
        item = indices[indptr[i]:indptr[i+1]]
        rating = data[indptr[i]:indptr[i+1]]
        sample_ind = np.random.choice(caps[i], nprofile[i], replace=False)
        samples[i], ratings[i] = item[sample_ind], rating[sample_ind]
    return samples, ratings


@njit(parallel=True)
def _sample_uniform_foreach(data, indices, indptr, nrows, caps, nprofile):
    samples = [np.empty(0, dtype=indices.dtype)] * nrows
    ratings = [np.empty(0, dtype=data.dtype)] * nrows
    for i in numba.prange(nrows):
        item = indices[indptr[i]:indptr[i+1]]
        rating = data[indptr[i]:indptr[i+1]]
        sample_ind = np.random.choice(caps[i], nprofile[i], replace=False)
        samples[i], ratings[i] = item[sample_ind], rating[sample_ind]
    return samples, ratings


def sample_uniform_coo_nbp(coo, dist_func):
    """

    Args:
        coo (scipy.sparse.coo_matrix): The sparse rating matrix.
        dist_func (callable): The distribution function that generates user
            profile size. It takes one array that caps profile size for each
            user.

    Returns:
        scipy.sparse.coo_matrix: The sampled matrix.

    """
    csr = coo.tocsr()
    caps = csr.getnnz(axis=1)
    nprofile = dist_func(caps)
    nrows, _ = csr.shape
    return _sample_uniform_foreach(csr.data, csr.indices, csr.indptr, nrows,
                                   caps, nprofile)


def sample_uniform_csr_nbp(csr, dist_func):
    """

    Args:
        csr (scipy.sparse.csr_matrix): The sparse rating matrix.
        dist_func (callable): The distribution function that generates user
            profile size. It takes one array that caps profile size for each
            user.

    Returns:
        scipy.sparse.coo_matrix: The sampled matrix.

    """
    caps = csr.getnnz(axis=1)
    nprofile = dist_func(caps)
    nrows, _ = csr.shape
    return _sample_uniform_foreach(csr.data, csr.indices, csr.indptr, nrows,
                                   caps, nprofile)


@njit(nogil=True)
def random_choice(p, size):
    """
    Randomly chooses n items without replacement given p.

    Size must be smaller than the length of p

    Args:
        p (numpy.array): The sampling probability of each item. If it doesn't
            sum to 1, it will be normalized.
        size (int): The sample size.

    Returns:
        numpy.array: The samples.
    """
    out = np.empty(size, dtype=np.int32)
    pvals = p / p.sum()
    for i in range(size):
        sample, = multinomial(1, pvals).nonzero()[0]
        out[i] = sample
        pvals[sample] = 0
        pvals = pvals / pvals.sum()
    return out


def sample_popular_n(ratings, dist_func, use_cap=True, **kwargs):
    """
    A function to sample observations from user true preferences.

    This function draws a user profile size n from `dist_func` for each user,
    then n items are drew based on their popularity.

    Args:
        ratings (pandas.DataFrame): The data frame containing user
            consumption data.
        dist_func (function): The distribution function used to draw user
            profile size from. Any numpy.random distributions can be used.
        use_cap (bool): If True, use preference cap as user profile size;
            if False, redraw user profile size
        **kwargs: Keyword arguments used by dist_func except the `size`
            parameter.

    Returns:
        pandas.DataFrame: The sampled user consumption data with columns:
            user, item.

    """
    col_names = ratings.columns
    if not {'user', 'item'}.issubset(col_names):
        ratings = ratings.rename(columns={col_names[0]: 'user',
                                          col_names[1]: 'item'})
    # compute user preference stats
    pref_stats = ratings[['user', 'item']].groupby('user', as_index=False).count()
    nusers = len(pref_stats)
    npref = pref_stats['item'].values
    # sample user profile size
    nprofile = _generate_profile_size(nusers, npref, dist_func, use_cap,
                                      **kwargs)
    pref_stats['nprofile'] = nprofile
    pref_stats.drop('item', axis=1, inplace=True)   # columns: user, nprofile
    pref_stats_dict = dict(pref_stats.values)
    popularity = ratings[['user', 'item']].groupby('item', as_index=False).count()
    popularity['prob'] = popularity['user'] / popularity['user'].sum()
    popularity.drop('user', axis=1, inplace=True)   # columns: item, prob

    merged = ratings.merge(popularity, on='item', how='left')

    return (merged.groupby('user')
            .apply(lambda x: x.sample(n=pref_stats_dict.get(x.name),
                                      weights='prob'))
            .reset_index(drop=True)
            .drop('prob', axis=1))


def sample_popular_coo(coo, dist_func):
    """
    Samples items based on item popularity

    Args:
        coo (scipy.sparse.coo_matrix): The rating matrix.
        dist_func (callable): The function to draw user profile size.

    Returns:

    """
    csr = coo.tocsr()
    # profile size
    caps = csr.getnnz(axis=1)
    nprofile = dist_func(caps)
    popularity = coo.tocsc().getnnz(axis=0).astype(np.float_)
    nrows, ncols = csr.shape
    indices = csr.indices
    indptr = csr.indptr
    data = csr.data
    samples = [np.empty(0, dtype=np.int32)] * nrows
    ratings = [np.empty(0, dtype=np.float_)] * nrows
    for i in range(nrows):
        item = indices[indptr[i]:indptr[i+1]]
        rating = data[indptr[i]:indptr[i+1]]
        prob = popularity[item] / popularity[item].sum()
        sample_ind = np.random.choice(caps[i], nprofile[i], False,
                                      prob)
        samples[i], ratings[i] = item[sample_ind], rating[sample_ind]
    return samples, ratings


@njit(parallel=True)
def _sample_popular_foreach(data, indices, indptr, nrows, nprofile,
                            popularity):
    samples = [np.empty(0, dtype=indices.dtype)] * nrows
    ratings = [np.empty(0, dtype=data.dtype)] * nrows
    for i in numba.prange(nrows):
        item = indices[indptr[i]:indptr[i+1]]
        rating = data[indptr[i]:indptr[i+1]]
        prob = popularity[item]
        sample_ind = random_choice(prob, nprofile[i])
        samples[i], ratings[i] = item[sample_ind], rating[sample_ind]
    return samples, ratings


def sample_popular_coo_nbp(coo, dist_func):
    """
    Samples items based on item popularity

    Args:
        coo (scipy.sparse.coo_matrix): The rating matrix.
        dist_func (callable): The function to draw user profile size.

    Returns:

    """
    csr = coo.tocsr(copy=False)
    # profile size
    caps = csr.getnnz(axis=1)
    nprofile = dist_func(caps)
    popularity = coo.tocsc(copy=False).getnnz(axis=0)
    nrows, _ = csr.shape
    indices = csr.indices
    indptr = csr.indptr
    data = csr.data
    return _sample_popular_foreach(data, indices, indptr, nrows, nprofile,
                                   popularity)


def sample_popular_csr_nbp(csr, dist_func):
    """
    Samples items based on item popularity

    Args:
        csr (scipy.sparse.csr_matrix): The rating matrix.
        dist_func (callable): The function to draw user profile size.

    Returns:
        :obj:`list` of :obj:`numpy.array`: The list of items for each user
        :obj:`list` of :obj:`numpy.array`: The list of ratings for each user

    """
    # profile size
    caps = csr.getnnz(axis=1)
    nprofile = dist_func(caps)
    popularity = csr.getnnz(axis=0)
    nrows, _ = csr.shape
    indices = csr.indices
    indptr = csr.indptr
    data = csr.data
    return _sample_popular_foreach(data, indices, indptr, nrows, nprofile,
                                   popularity)


# TODO: Come up with a better solution
def _generate_profile_size(nusers, npref, dist_func, use_cap=True, **kwargs):
    nprofile = np.zeros(nusers, dtype=np.int64)
    outliers = np.full(nprofile.shape, True, dtype=bool)
    while any(outliers):
        # draw the total number of consumed items for each user from `dist_func`
        ndraws = np.count_nonzero(outliers)
        nprofile[outliers] = dist_func(size=ndraws, **kwargs)
        outliers = (nprofile == 0)
        if not use_cap:
            outliers |= (nprofile > npref)
       #  _logger.debug("%s %d re-draws", dist_func.__name__, ndraws)
       #  _logger.debug("parameters: %s", kwargs)
    outliers = (nprofile > npref)
    nprofile[outliers] = npref[outliers]
    return nprofile


def _generate_profile_size_cap(nusers, npref, dist_func, **kwargs):
    nprofile = np.zeros(nusers, dtype=np.int64)
    outliers = (nprofile == 0)
    # re-draw nprofile when it's 0
    while any(outliers):
        # draw the total number of consumed items for each user from `dist_func`
        ndraws = np.count_nonzero(outliers)
        nprofile[outliers] = dist_func(size=ndraws, **kwargs)
        outliers = (nprofile == 0)
        _logger.debug("%d re-draws", ndraws)
    # use npref as cap when nprofile > npref
    outliers = (nprofile > npref)
    nprofile[outliers] = npref[outliers]
    return nprofile


@njit(nogil=True)
def _trunc_pareto_p(m, alpha, p):
    u = np.random.uniform(0, p)
    y = m / ((1 - u) ** (1 / alpha))
    return int(y)


@njit(parallel=True)
def _trunc_pareto_profile_reject(m, alpha, p, caps):
    n = len(caps)
    out = np.empty(n, dtype=np.int32)
    for i in numba.prange(n):
        icap = caps[i]
        if icap == 0:
            out[i] = 0
            continue
        sample = _trunc_pareto_p(m, alpha, p)
        while (sample > icap) | (sample == 0):
            sample = _trunc_pareto_p(m, alpha, p)
        out[i] = sample
    return out


@njit(parallel=True)
def _trunc_pareto_profile_cap(m, alpha, p, caps):
    n = len(caps)
    out = np.empty(n, dtype=np.int32)
    for i in numba.prange(n):
        icap = caps[i]
        if icap == 0:
            out[i] = 0
            continue
        sample = _trunc_pareto_p(m, alpha, p)
        while sample == 0:
            sample = _trunc_pareto_p(m, alpha, p)
        out[i] = min(sample, icap)
    return out


def trunc_pareto_profile(m, alpha, upper, caps, use_cap=True):
    p = pareto.cdf(upper, alpha, scale=m)
    if use_cap:
        return _trunc_pareto_profile_cap(m, alpha, p, caps)
    else:
        return _trunc_pareto_profile_reject(m, alpha, p, caps)


class TruncParetoProfile(BaseModel):
    """
    Truncated Pareto distribution.

    """
    def __init__(self, m, alpha, upper, cap=True):
        """

        Args:
            m (float): The scale parameter. It's also the lower bound. `m` > 0.
            alpha (float): The shape parameter. alpha > 0
            upper (float): The upper bound of the distribution support.
                `upper` must be greater than m.
            cap (bool): Whether to cap the profile size. If True, use cap
                when sample is greater than cap. If False, redraw samples
                until the sample is below the cap.

        """
        self.m = m
        self.alpha = alpha
        self.upper = upper
        self.cap = cap

    def __call__(self, caps):
        return trunc_pareto_profile(self.m, self.alpha, self.upper,
                                    caps, self.cap)
