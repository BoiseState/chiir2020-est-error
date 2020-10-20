"""
The Utility module for simulations
"""
import os
import pickle

import numba
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import coo_matrix
from scipy.stats import pareto

from simulation_utils.base import BaseModel
from simulation_utils.observation import _trunc_pareto_p


def summarize_data_sets(data_funcs):
    """
    A utility function for summarizing data sets.

    Args:
        data_funcs (:obj:`list` of :obj:`function`): The list of data functions.
            Each function will be called to generate user-item pairs as a pandas.DataFrame.
            Then the data statistics will be summarized and user activities, popularity curve,
            and popularity count will be plotted.

    Returns:
        None

    """
    dataset_stats = []
    ds_user_profile_stats = []
    ds_item_popularity_stats = []
    ds_user_profile = []
    ds_item_popularity = []
    for name, data_fun in data_funcs:
        # data stats
        dataset = data_fun()
        ds_col_name = dataset.columns
        dataset.rename(columns={ds_col_name[0]: 'user',
                                ds_col_name[1]: 'item'},
                       inplace=True)
        nusers = len(dataset.iloc[:, 0].unique())
        nitems = len(dataset.iloc[:, 1].unique())
        npairs = len(dataset)
        density = npairs / nitems / nusers
        dataset_stats.append([name, nusers, nitems, npairs, density])

        # user profile
        user_profile = dataset.iloc[:, :2].groupby('user', as_index=False).count()

        # get user profile size stats and rename the column to dataset name
        user_profile_stats = user_profile['item'].rename(name).describe()
        ds_user_profile_stats.append(user_profile_stats)

        # save user profile data for plotting
        user_profile['dataset'] = name
        user_profile.drop('user', axis=1)
        ds_user_profile.append(user_profile)

        # item popularity stats
        item_popularity = dataset.iloc[:, :2].groupby('item', as_index=False).count()
        item_popularity_stats = item_popularity['user'].rename(name).describe()
        ds_item_popularity_stats.append(item_popularity_stats)

        item_popularity['dataset'] = name
        item_popularity.drop('item', axis=1)
        ds_item_popularity.append(item_popularity)

    df = pd.DataFrame.from_records(dataset_stats,
                                   columns=['dataset', 'nusers', 'nitems', 'npairs', 'density'])
    print('dataset stats')
    print(df)

    df = pd.concat(ds_user_profile_stats, axis=1)
    print('user profile stats')
    print(df)

    df = pd.concat(ds_item_popularity_stats, axis=1)
    print('item popularity stats')
    print(df)

    user_profile_df = pd.concat(ds_user_profile, axis=0, ignore_index=True)

    item_popularity_df = pd.concat(ds_item_popularity, axis=0, ignore_index=True)
    count_popularity = (item_popularity_df
                        .groupby(['dataset', 'user'])
                        .size().reset_index().rename(columns={0: 'count'}))
    item_popularity_df['rank'] = item_popularity_df.groupby('dataset')['user'].rank(method='min',
                                                                                    ascending=False)

    import seaborn as sns
    import matplotlib.pyplot as plt
    g = sns.FacetGrid(user_profile_df, col='dataset', sharey=False, sharex=False, col_wrap=2,
                      height=5)
    g.map(plt.hist, 'item') \
        .set_titles('{col_name}', size=15) \
        .set_axis_labels('user profile size', 'number of users')

    g = sns.FacetGrid(item_popularity_df, col='dataset', sharey=False, sharex=False, col_wrap=2,
                      height=5)
    g.map(plt.scatter, 'rank', 'user') \
        .set(xscale='log', yscale='log') \
        .set_titles('{col_name}', size=15) \
        .set_axis_labels('rank', 'popularity')

    g = sns.FacetGrid(count_popularity, col='dataset', sharey=False, sharex=False, col_wrap=2,
                      height=5)
    g.map(plt.scatter, 'user', 'count') \
        .set(xscale='log', yscale='log') \
        .set_titles('{col_name}', size=15) \
        .set_axis_labels('popularity', 'count')
    plt.show()


def eval_algorithm(aname, algo, train, pref, users, n):
    """
    A function to evaluate an algorithm and generate recommendations for the given users.

    Args:
        aname (str): The algorithm name
        algo (lenskit.Algorithm): The algorithm to train
        train (pandas.DataFrame): The train data with user, item, rating columns.
        pref (pandas.DataFrame): The true preference data used to train the Oracle recommender
        users (array-like): The user ids to recommend for
        n (int): The number of recommendations per user.

    Returns:
        pandas.DataFrame: The recommendations dataframe with columns: user, item, rank, algorithm
    """
    from lenskit import batch
    from lenskit import topn
    from simulation_utils.recommenders import Oracle
    # train models on training data or preference data if it's Oracle
    algo.fit(pref) if isinstance(algo, Oracle) else algo.fit(train)
    recs = batch.recommend(algo, users, n,
                           topn.UnratedCandidates(train))
    # bind with algorithm names
    recs['algorithm'] = aname
    recs.drop('score', axis=1, errors='ignore', inplace=True)
    return recs


def compute_metrics(recommendations, metrics, test):
    """
    A function to compute metrics based on the given test data.

    Args:
        recommendations (pandas.DataFrame): The recommendations generated from different algorithms.
            This dataframe contain columns: user, item, rank, algorithm.
        metrics (list of str): The list of metric function names. It can be precision, recall,
            recip_rank, ndcg
        test (pandas.DataFrame): The test data, which provides relevance information for the
        recommended items in the `recommendations`.

    Returns:
        pandas.DataFrame: The metrics for each user per algorithm.
    """
    from lenskit.metrics.topn import (
        precision, recall, recip_rank, ndcg
    )
    metric_funcs = {'precision': precision, 'recall': recall,
                    'recip_rank': recip_rank, 'ndcg': ndcg}
    # merge relevance data.
    # items with not-nan rank are recommendations. items with not-nan rating are relevant items.
    recs_rels = (recommendations.groupby('algorithm')[['user', 'item', 'rank']]
                 .apply(lambda x: x.merge(test, how='outer'))
                 .reset_index(0))
    res_list = []
    for metric in metrics:
        metric_func = metric_funcs[metric]
        if metric == 'ndcg':
            # locate recommendation rows
            recs = recs_rels.dropna(subset=['rank']).copy()
            # set missing relevance to 0
            recs.loc[recs['rating'].isna(), 'rating'] = 0
            result = recs.groupby(['algorithm', 'user'])['rating'].apply(metric_func)
            result.rename(metric, inplace=True)
        else:
            result = (recs_rels.groupby(['algorithm', 'user'])[
                          ['item', 'rank', 'rating']
                      ].apply(
                            lambda x: metric_func(
                                        x.loc[x['rank'].notna(), :],
                                        x.loc[x['rating'].notna(), :]
                                        .set_index('item'))
                              )
                      )
            result.rename(metric, inplace=True)
        res_list.append(result)
    return pd.concat(res_list, axis=1).reset_index()


# NOT USED
def compute_user_activity(data, nmax=None, normed=True):
    """
    A function to compute the user activity distribution of the given data

    Args:
        data (pandas.DataFrame): The user consumed data. The first and second columns are treated as
            user ids and item ids respectively.
        nmax (int, optional): The max number of consumed items for a user. The returned
            distribution is truncated from 1 to nmax.
        normed (bool, optional): Whether to return count or normed density

    Returns:
        ndarray: An 1d numpy array with the same length as the total number of items (nmax if
            not None). The value at index k represents the relative frequency (counts) of users
            that consumed k items.

    """
    ds_col_name = data.columns
    user_item = data
    if not {'user', 'item'}.issubset(ds_col_name):
        user_item = data.rename(columns={ds_col_name[0]: 'user',
                                         ds_col_name[1]: 'item'})

    nitems = nmax

    # compute user profile size
    user_profile = user_item.loc[:, ['user', 'item']].groupby('user', as_index=False).count()
    # compute user profile size stats
    user_activity = user_profile.groupby('item', as_index=False).count()
    # compute relative frequency
    if normed:
        total_num = user_activity['user'].sum()
        user_activity['user'] = user_activity['user'] / total_num
    return user_activity['user'].values


def beta_binomial(a, b, n, size=None):
    p = np.random.beta(a, b, size)
    samples = np.random.binomial(n, p)
    return samples


def truncated_beta_binomial(a, b, n, size=None, lower=20):
    out = beta_binomial(a, b, n, size)
    outliers = out < lower
    while any(outliers):
        ndraws = np.count_nonzero(outliers)
        out[outliers] = beta_binomial(a, b, n, ndraws)
        outliers = out < lower
    return out


def truncated_pareto(m, alpha, size=None, upper=737.0):
    p = pareto.cdf(upper, alpha, scale=m)
    u = np.random.uniform(0, p, size)
    y = m / ((1 - u) ** (1 / alpha))
    return y.astype(np.int64)


@njit(parallel=True)
def _trunc_pareto_p_n(m, alpha, p, size):
    out = np.empty(size, dtype=np.int32)
    for i in numba.prange(size):
        out[i] = _trunc_pareto_p(m, alpha, p)
    return out


def truncated_pareto_nbp(m, alpha, size, upper):
    p = pareto.cdf(upper, alpha, scale=m)
    return _trunc_pareto_p_n(m, alpha, p, size)


class TruncatedBetaBinomial(BaseModel):
    """
    Truncated beta binomial distribution.

    """
    def __init__(self, a, b, n, lower=20):
        """

        Args:
            a (float): The alpha parameter of beta distribution. a > 0.
            b (float): The beta parameter of beta distribution. b > 0.
            n (int): The number of trials of binomial distribution. n > lower.
            lower (int): The lower bound of the beta binomial distribution.
        """
        self.a = a
        self.b = b
        self.n = n
        self.lower = lower

    def __call__(self, **kwargs):
        size = kwargs.pop('size', None)
        self.set_params(**kwargs)
        return truncated_beta_binomial(self.a, self.b, self.n,
                                       size, self.lower)


class TruncatedPareto(BaseModel):
    """
    Truncated Pareto distribution.

    """
    def __init__(self, m, alpha, upper=737):
        """

        Args:
            m (float): The scale parameter. It's also the lower bound. `m` > 0.
            alpha (float): The shape parameter. alpha > 0
            upper (float): The upper bound of the distribution support.
                `upper` must be greater than m. The default value is 737 which
                is the upper bound of the user profile size of ml_100k data set.
        """
        self.m = m
        self.alpha = alpha
        self.upper = upper

    def __call__(self, **kwargs):
        size = kwargs.pop('size', None)
        self.set_params(**kwargs)
        return truncated_pareto(self.m, self.alpha, size, self.upper)


def compile_stan_model(model_code, model_name, model_dir='build', **kwargs):
    fn = '.'.join([model_name, 'pkl'])
    fdir = os.path.join(model_dir, fn)
    try:
        with open(fdir, 'rb') as f:
            sm = pickle.load(f)
    except FileNotFoundError as e:
        import pystan
        sm = pystan.StanModel(model_code=model_code,
                              model_name=model_name,
                              **kwargs)
        with open(fdir, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print('Load compiled model')
    return sm


def frame_to_coo(data, row='user', col='item', value='rating',
                 reset_index=True, copy=True):
    """
    Converts a pandas data frame to a scipy sparse coo matrix.

    """
    if reset_index:
        row_idx = pd.Index(data[row].unique())
        col_idx = pd.Index(data[col].unique())
        users = row_idx.get_indexer(data[row]).astype(np.int32, copy=False)
        items = col_idx.get_indexer(data[col]).astype(np.int32, copy=False)
    else:
        users = data[row].to_numpy(dtype=np.int32, copy=copy)
        items = data[col].to_numpy(dtype=np.int32, copy=copy)
    ratings = data[value].to_numpy(dtype=np.float_, copy=copy)
    return coo_matrix((ratings, (users, items)))
