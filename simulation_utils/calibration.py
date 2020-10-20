"""
This module includes classes and functions for calibrating data simulations.

"""
from itertools import product
import logging

from joblib import delayed, Parallel
# from memory_profiler import profile
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import numpy as np
from numba import njit, numba
import pandas as pd

from simulation_utils.base import BaseModel


_log = logging.getLogger(__name__)
_STATSNAME_TO_COLNAME = {'user_activity': 'user', 'item_popularity': 'item'}


def _dict_product(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for value in product(*values):
        yield dict(zip(keys, value))


def _param_grid(*args):
    """
    Unpacks a list of dictionaries (param: list of values) to individual
    dictionary (param: value)

    """
    for params in args:
        for param in _dict_product(**params):
            yield param


def compute_dstats(data, icol=0, nmin=None, nmax=None, normed=False):
    """
    Computes the distribution of the user activity/item popularity.

    Args:
        data (numpy.ndarray, pandas.DataFrame): A 2-d numpy array that contains
            user-item interactions or 1-d numpy array that contains users/items.
        icol (int, optional): The index of the column to compute on.
        nmin (int, optional): The lower bound to truncate the result.
        nmax (int, optional): The upper bound to truncate the result.
        normed (bool, optional): Whether or not to normalize the distribution.

    Returns:
        pandas.Series: the distribution of the user activity/item popularity.

    """
    dims = len(data.shape)
    records = data[:, icol] if dims > 1 else data
    _, data_stats = np.unique(records, return_counts=True)
    # the distribution of the statistics
    stats, counts = np.unique(data_stats, return_counts=True)
    del data_stats
    out = pd.Series(data=counts, index=stats)
    if normed:
        out = out / out.sum()
    return out.loc[nmin:nmax]


def calibrate_data(data, pref_gen, obs_gen, pref_kwargs,
                   obs_kwargs, stats_name=('user_activity', 'item_popularity'),
                   similarity='kl_divergence', nthread=None):
    # TODO: Change to class to support more parameter search methods.
    """
    Calibrates simulated data set against an existing one using grid search.

    Args:
        data (pandas.DataFrame): The existing user consumption data.
            It has at least two columns: user, item.
        pref_gen (function): The preference generator function.
        obs_gen (function): The observation sampler function.
        pref_kwargs (dict): The dictionary of preference parameters to tune.
        obs_kwargs (dict, list of dict): The dictionary of observation
        parameters to tune.
        stats_name (str, list of str): The name of data statistics used to tune.
        similarity (str): The metric name used to compute similarity.
        nthread (int, None): The thread count. If None, use cpu_count() threads.

    Returns:
        pandas.DataFrame: The frame that contains similarity score for each pair of tuning
            parameters.

    """
    from multiprocessing import cpu_count
    from multiprocessing import Pool
    if nthread and nthread >= cpu_count():
        raise ValueError('nthread must be less than the system cpu count')
    # compute data statistics
    data_stats = dict()
    if isinstance(stats_name, str):
        stats_name = (stats_name,)
    for sn in stats_name:
        colname = _STATSNAME_TO_COLNAME[sn]
        data_stats[sn] = compute_dstats(data[colname].values)

    pref_params = _param_grid(pref_kwargs)
    if isinstance(obs_kwargs, dict):
        obs_kwargs = [obs_kwargs]
    obs_params = _param_grid(*obs_kwargs)
    model_params = product([data_stats], [pref_gen], [obs_gen], pref_params,
                           obs_params, [similarity])
    with Pool(nthread) as p:
        result = p.starmap(_eval_model, model_params)
    return result


def _eval_model(data_stats, pref_gen, obs_gen, pref_kwargs, obs_kwargs,
                similarity='kl_divergence'):
    out = dict()
    out['pref'] = pref_gen.__name__
    out['obs'] = obs_gen.__name__
    for k, v in pref_kwargs.items():
        pname = '_'.join(['pref', k])
        out[pname] = v
    for k, v in obs_kwargs.items():
        pname = '_'.join(['obs', k])
        out[pname] = v.__name__ if callable(v) else v

    # generate data
    pref = pref_gen(**pref_kwargs)
    obs = obs_gen(pref, **obs_kwargs)

    # compute data statistics
    for sn, dstats in data_stats.items():
        colname = _STATSNAME_TO_COLNAME[sn]
        obs_stats = compute_dstats(obs[colname].values)
        score = compute_similarity(dstats, obs_stats, similarity)
        metric_name = '_'.join([sn, similarity])
        out[metric_name] = score
    return out


def compute_similarity(x, y, metric='kl_divergence'):
    """
    Computes similarity of two distributions

    Args:
        x (pandas.Series): The first distribution. Index is the domain.
        y (pandas.Series): The second distribution.
        metric (str): `kl_divergence` or `pearson`

    Returns:
        float: The similarity score.

    """
    from scipy.stats import entropy, pearsonr
    # remove zeros slightly increase divergence
    x = x[x != 0]
    y = y[y != 0]
    # outer join two distributions
    eps = min(x.min(), y.min()) / 10
    xy = pd.concat([x, y], axis=1).add(eps, fill_value=0)
    x = xy.iloc[:, 0]
    y = xy.iloc[:, 1]
    if metric == 'pearson':
        score, _ = pearsonr(x, y)
    else:
        score = entropy(x, y)
    return score


# NOT USED
def compute_dcorr(data, label='user', index='item', values='rating', frac=0.2):
    idx = data[label].unique()
    sample_size = int(len(idx) * frac)
    sample_idx = np.random.choice(idx, size=sample_size, replace=False)
    dsub = data[data['user'].isin(sample_idx)]
    dcorr = dsub.pivot_table(index=index, columns=label,
                             values=values, fill_value=0).corr().values
    out = dcorr[np.tril_indices(sample_size, -1)]
    return out


# TODO: sample vector with at least 5 interactions
def _sample_pairs(a, frac, size_cap=int(1e6)):
    # sample_size: a fraction of the number of (n choose 2)
    if isinstance(a, int):
        sample_size = int(a * (a - 1) / 2 * frac)
    else:
        sample_size = int(len(a) * (len(a) - 1) / 2 * frac)
    sample_size = min(sample_size, size_cap)
    pairs = np.random.choice(a, size=(sample_size, 2))
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pairs


@njit(parallel=True)
def _sample_pairs_nbp(data, frac, size_cap=np.int(1e6)):
    """Not return unique pairs and slow"""
    sample_size = int(len(data) * (len(data) - 1) / 2 * frac)
    sample_size = min(sample_size, size_cap)
    pairs = np.empty((sample_size, 2))
    for i in numba.prange(sample_size):
        pair = np.random.choice(data, size=2, replace=False)
        pair.sort()
        pairs[i] = pair
    return pairs


def compute_cosine_similarity_pd(data, pairs, label='user', index='item',
                                 values='rating', frac=0.2):
    idx = data[label].unique()
    if callable(pairs):
        pairs = pairs(idx, frac)
    unique_labels = np.unique(pairs)
    data = data[data[label].isin(unique_labels)]
    data = csr_matrix(
        (data[values].values, (data[label].values, data[index].values))
    )
    subpairs = np.array_split(pairs, 10)
    scores = []
    for p in subpairs:
        rrows = p[:, 0]
        lrows = p[:, 1]
        data_norm = norm(data, axis=1)
        vprod = data[rrows].multiply(data[lrows])
        vprod = np.squeeze(vprod.sum(axis=1).getA())
        vnorm = data_norm[rrows] * data_norm[lrows]
        assert np.all(vnorm != 0)
        score = vprod / vnorm
        scores.append(score)
    return np.hstack(scores)


def pairwise_cosine_similarity(data, pairs, axis=0):
    """
    Computes cosine similarity of given index pairs

    Args:
        data (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix): The rating
            matrix.
        pairs (numpy.array): A 2-d array with index pair in each row.
            slice_row (bool): Whether to compute similarity between rows,
            otherwise columns.
        axis (int): Axis along which the vector pair is sliced. axis=0,
            computes row vector pairs, and axis=1 for column vectors.

    Returns:
        numpy.array: cosine similarity scores.

    """
    lp, rp = pairs[:, 0], pairs[:, 1]
    if axis:
        vec_prod = data[:, lp].multiply(data[:, rp])
    else:
        vec_prod = data[lp].multiply(data[rp])
    axis = 1 - axis
    vec_norm = norm(data, axis=axis)
    vprod = np.squeeze(vec_prod.sum(axis=axis).getA())
    vnorm = vec_norm[lp] * vec_norm[rp]
    assert np.all(vnorm != 0)
    return vprod / vnorm


def pairwise_cosine_similarity_split_join(csr, pairs, axis=0, sections=8):
    """
    Computes cosine similarity of given index pairs

    Args:
        csr (scipy.sparse.csr_matrix): The rating matrix.
        pairs (numpy.array): A 2-d array with index pair in each row.
            slice_row (bool): Whether to compute similarity between rows,
            otherwise columns.
        axis (int): Axis along which the vector pair is sliced. axis=0,
            computes row vector pairs, and axis=1 for column vectors.
        sections (int): The number of splits in computation.

    Returns:
        numpy.array: cosine similarity scores.

    """
    subpairs = np.array_split(pairs, sections)
    scores = []
    if axis:
        data = csr.tocsc(copy=False)
    else:
        data = csr
    for pair in subpairs:
        score = pairwise_cosine_similarity(data, pair, axis)
        scores.append(score)
    return np.hstack(scores)


@njit(parallel=True)
def pairwise_cosine_nbp(data, pairs):
    n = len(pairs)
    scores = np.empty(n, dtype=np.float64)
    for i in numba.prange(n):
        ix, iy = pairs[i]
        x, y = data.row(ix), data.row(iy)
        score = cosine_similarity_nbp(x, y)
        scores[i] = score
    return scores


@njit(parallel=True)
def cosine_similarity_nbp(a, b):
    n = len(a)
    prods = 0.0
    anorm = 0.0
    bnorm = 0.0
    for i in numba.prange(n):
        x, y = a[i], b[i]
        prods += x * y
        anorm += x * x
        bnorm += y * y
    return prods / np.sqrt(anorm * bnorm)


class Calibrator(BaseModel):
    """
    The calibrator.

    """
    def __init__(self, pref, obs, ds, n_jobs=1):
        """

        Args:
            pref (simulation_utils.preference.PreferenceModel): The
                preference model
            obs (simulation_utils.observation.ObservationModel): The
                observation model
            ds (simulation_utils.datasets.DataStats): The DataStats object
                which contains statistics of an existing data set.
            n_jobs (int): The number of parallel jobs for computing metrics.

        """
        self.pref = pref
        self.obs = obs
        self.ds = ds
        self.n_jobs = n_jobs

    def _compute_correlation_similarity(self, data, label, index, mrated=5):
        # get bins and values of pair similarity distribution from a data set.
        if label == 'item':
            ds_bin_edges = self.ds.item_corr.index.values
            ds_corr = self.ds.item_corr.iloc[1:]
            item_popularity = (data[['user', 'item']].groupby(
                'item', as_index=False).count())
            items = (item_popularity[item_popularity['user'] >= mrated]['item']
                     .values)
            if len(items) <= 10:
                return 1e6
            subset = data[data['item'].isin(items)]
        else:
            ds_bin_edges = self.ds.user_corr.index.values
            ds_corr = self.ds.user_corr.iloc[1:]
            user_activity = data[['user', 'item']].groupby(
                'user', as_index=False).count()
            users = (user_activity[user_activity['item'] >= mrated]['user']
                     .values)
            if len(users) <= 10:
                return 1e6
            subset = data[data['user'].isin(users)]
        # sample pairs and compute cosine similarity
        sim_cs = compute_cosine_similarity_pd(subset, _sample_pairs, label,
                                              index)
        sim_dist, _ = np.histogram(sim_cs, bins=ds_bin_edges, range=(-1, 1))
        return compute_similarity(ds_corr,
                                  pd.Series(sim_dist, ds_bin_edges[1:]))

    def _compute_popularity_activity_similarity(self, data, label):
        if label == 'item':
            ds_stats = self.ds.item_popularity
        else:
            ds_stats = self.ds.user_activity

        obs_stats = compute_dstats(data[label].values)
        return compute_similarity(ds_stats, obs_stats)

    def generate_simulated_data(self, implicit=True):
        np.random.seed()
        pref = self.pref.generate()
        if implicit:
            pref['rating'] = 1
        return self.obs.sample(pref)

    @delayed
    def _score(self, metric, implicit=True):
        data = self.generate_simulated_data(implicit)
        if metric == 'item-pop':
            return self._compute_popularity_activity_similarity(data, 'item')
        if metric == 'user-act':
            return self._compute_popularity_activity_similarity(data, 'user')
        if metric == 'ucorr':
            return self._compute_correlation_similarity(data, 'user', 'item')
        if metric == 'icorr':
            return self._compute_correlation_similarity(data, 'item', 'user')
        if metric == 'all':
            scores = (
                    self._compute_popularity_activity_similarity(data, 'item'),
                    self._compute_popularity_activity_similarity(data, 'user'),
                    self._compute_correlation_similarity(data, 'user', 'item'),
                    self._compute_correlation_similarity(data, 'item', 'user')
            )
            return np.sum(scores)

    def score(self, metric, implicit=True, ntimes=10, return_std=False):
        scores = Parallel(n_jobs=self.n_jobs)(
            self._score(metric, implicit) for _ in range(ntimes))
        return np.mean(scores)


class CalibratorCSR(BaseModel):
    """
    The calibrator.

    """
    def __init__(self, pref, obs, ds, n_jobs=1):
        """

        Args:
            pref (simulation_utils.preference.PreferenceModel): The
                preference model
            obs (simulation_utils.observation.ObservationModel): The
                observation model
            ds (simulation_utils.datasets.DataStats): The DataStats object
                which contains statistics of an existing data set.
            n_jobs (int): The number of parallel jobs for computing metrics.

        """
        self.pref = pref
        self.obs = obs
        self.ds = ds
        self.n_jobs = n_jobs

    def _compute_correlation_similarity(self, csr, axis=0, mrated=5):
        rated = csr.getnnz((1 - axis))
        n = np.nonzero(rated >= mrated)[0]
        if len(n) <= 10:
            n = np.nonzero(rated >= 1)[0]
        # get bins and values of pair similarity distribution from a data set.
        if axis:
            ds_bin_edges = self.ds.item_corr.index.values
            ds_corr = self.ds.item_corr.iloc[1:]
        else:
            ds_bin_edges = self.ds.user_corr.index.values
            ds_corr = self.ds.user_corr.iloc[1:]
        del rated
        # sample pairs and compute cosine similarity
        pairs = _sample_pairs(n, frac=0.2)
        del n
        sim_cs = pairwise_cosine_similarity_split_join(csr, pairs, axis=axis)
        del pairs
        sim_dist, _ = np.histogram(sim_cs, bins=ds_bin_edges, range=(-1, 1))
        return compute_similarity(ds_corr,
                                  pd.Series(sim_dist, ds_bin_edges[1:]))

    def _compute_popularity_activity_similarity(self, csr, axis=0):
        if axis:
            ds_stats = self.ds.item_popularity
            a = csr.getnnz((1 - axis))
        else:
            ds_stats = self.ds.user_activity
            a = csr.getnnz((1 - axis))
        stats, counts = np.unique(a, return_counts=True)
        obs_stats = pd.Series(data=counts, index=stats)
        return compute_similarity(ds_stats, obs_stats)

    def generate_simulated_data(self, implicit=True):
        np.random.seed()
        pref = self.pref.generate()
        if implicit:
            pref.data[:] = 1
        return self.obs.sample(pref)

    @delayed
    def _score(self, metric, implicit=True, return_sum=True):
        csr = self.generate_simulated_data(implicit)
        if metric == 'item-pop':
            return self._compute_popularity_activity_similarity(csr, axis=1)
        if metric == 'user-act':
            return self._compute_popularity_activity_similarity(csr, axis=0)
        if metric == 'icorr':
            return self._compute_correlation_similarity(csr, axis=1)
        if metric == 'ucorr':
            return self._compute_correlation_similarity(csr, axis=0)
        if metric == 'all':
            scores = (
                    self._compute_popularity_activity_similarity(csr, axis=1),
                    self._compute_popularity_activity_similarity(csr, axis=0),
                    self._compute_correlation_similarity(csr, axis=1),
                    self._compute_correlation_similarity(csr, axis=0)
            )
            if return_sum:
                return np.sum(scores)
            else:
                return scores

    def score(self, metric, implicit=True, ntimes=10,
              return_sum=True, return_std=False):
        scores = Parallel(n_jobs=self.n_jobs)(
            self._score(metric, implicit, return_sum) for _ in range(ntimes))
        if return_std:
            return np.mean(scores, axis=0), np.std(scores, axis=0)
        else:
            return np.mean(scores, axis=0)
