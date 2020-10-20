import gc
import sys
import logging, logging.config
import types
from itertools import product

import numpy as np
import pandas as pd
import pytest
from pytest import approx
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from simulation_utils.calibration import _param_grid, compute_dstats,  \
    _dict_product, compute_dcorr, compute_cosine_similarity_pd, \
    _sample_pairs, Calibrator, _sample_pairs_nbp, \
    cosine_similarity_nbp, \
    pairwise_cosine_similarity, CalibratorCSR, \
    pairwise_cosine_similarity_split_join
from simulation_utils.datasets import ml_100k, DataStats
from simulation_utils.preference import generate_ibp_itemwise, \
    generate_uniform, LatentDirichletAllocation, LatentDirichletAllocationCSR
from simulation_utils.observation import PopularityObservation, \
    TruncParetoProfile, PopularityObservationCSR, UniformObservationCSR, \
    UniformObservation
from simulation_utils.preference import records_to_frame
from simulation_utils.utils import compute_user_activity, TruncatedPareto

log_config = {'version': 1,
              'formatters': {'simple': {'format': '{levelname} {name} {message}',
                                        'style': '{'}},
              'handlers': {'console': {'class': 'logging.StreamHandler',
                                       'level': 'INFO',
                                       'formatter': 'simple',
                                       'stream': 'ext://sys.stdout'},
                           'file': {'class': 'logging.FileHandler',
                                    'level': 'DEBUG',
                                    'formatter': 'simple',
                                    'filename': '../../build/sim.log',
                                    'encoding': 'utf8', 'mode': 'w'}},
              'loggers': {'eval-error': {'level': 'DEBUG',
                                         'handlers': ['console', 'file']},
                          'simulation_utils': {'level': 'DEBUG',
                                               'handlers': ['console', 'file']},
                          'root': {'level': 'INFO', 'handler': ['console']}}}

logging.config.dictConfig(log_config)
_log = logging.getLogger()


def generate_ibp_df(nusers, alpha, c=1, sigma=0):
    items = generate_ibp_itemwise(nusers, alpha, c, sigma)
    df = records_to_frame(items, 'item', 'user').reindex(columns=['user', 'item'])
    return df


def truncated_poisson(lam, size=None):
    k = np.ones(size)
    t = np.exp(-lam) / (1 - np.exp(-lam)) * lam
    s = t
    u = np.random.random(size)
    condition = s < u
    while any(condition):
        k[condition] = k[condition] + 1
        t = t * lam / k
        s = s + t
        condition = s < u
    return k


def test_dict_product():
    gen_params = list(_dict_product(user=[10], item=[10], frac=[0.2, 0.1, 0.3]))
    gen_params = sorted(gen_params, key=lambda x: x.get('frac'))
    assert gen_params == [{'user': 10, 'item': 10, 'frac': 0.1},
                          {'user': 10, 'item': 10, 'frac': 0.2},
                          {'user': 10, 'item': 10, 'frac': 0.3}]


def test_param_grid():
    list_args = [{'preference_model': [generate_ibp_df],
                  'nusers':[10, 20], 'alpha': [20, 50]},
                 {'preference_model': [generate_uniform],
                  'nusers':[10, 20], 'frac': [0.1, 0.2]}]
    param_grid = _param_grid(*list_args)
    assert isinstance(param_grid, types.GeneratorType)
    expected = [
        {'preference_model': generate_ibp_df, 'nusers': 10, 'alpha': 20},
        {'preference_model': generate_ibp_df, 'nusers': 10, 'alpha': 50},
        {'preference_model': generate_ibp_df, 'nusers': 20, 'alpha': 20},
        {'preference_model': generate_ibp_df, 'nusers': 20, 'alpha': 50},
        {'preference_model': generate_uniform, 'nusers': 10, 'frac': 0.1},
        {'preference_model': generate_uniform, 'nusers': 10, 'frac': 0.2},
        {'preference_model': generate_uniform, 'nusers': 20, 'frac': 0.1},
        {'preference_model': generate_uniform, 'nusers': 20, 'frac': 0.2},
    ]
    assert list(param_grid) == expected

    dict_args = {'preference_model': [generate_ibp_df],
                 'nusers': [10, 20],
                 'alpha': [20, 50]}
    param_grid = _param_grid(dict_args)
    assert isinstance(param_grid, types.GeneratorType)
    expected = [
        {'preference_model': generate_ibp_df, 'nusers': 10, 'alpha': 20},
        {'preference_model': generate_ibp_df, 'nusers': 10, 'alpha': 50},
        {'preference_model': generate_ibp_df, 'nusers': 20, 'alpha': 20},
        {'preference_model': generate_ibp_df, 'nusers': 20, 'alpha': 50},
    ]
    assert list(param_grid) == expected


def test_compute_dstats():
    data = ml_100k()
    dstats1 = compute_user_activity(data)
    dstats2 = compute_dstats(data)
    assert np.allclose(dstats1, dstats2)


@pytest.mark.parametrize('scale, nusers, nitems', ((1, 10, 20), (5, 10, 20)))
def test_compute_dcorr(scale, nusers, nitems):
    rating = np.random.randint(scale + 1, size=nusers * nitems)
    user = np.repeat(range(nusers), nitems)
    data = pd.DataFrame({'user': user, 'rating': rating})
    data['item'] = data.groupby('user', as_index=False).cumcount()
    actual = compute_dcorr(data, frac=1)
    indices = np.tril_indices(nusers, -1)
    expected = np.corrcoef(rating.reshape(nusers, nitems))[indices]
    assert actual == approx(expected)


@pytest.mark.parametrize('x, y', zip(np.random.randn(10, 100),
                                     np.random.randn(10, 100)))
def test_cosine_similarity(x, y):
    """Test single pair"""
    expected = 1 - cosine(x, y)
    actual = cosine_similarity_nbp(x, y)
    assert actual == approx(expected)
    data = np.hstack((x, y))
    n = len(x)
    rows = np.repeat([0, 1], n)
    cols = np.hstack((np.arange(n), np.arange(n)))
    csr = csr_matrix((data, (rows, cols)))
    score = pairwise_cosine_similarity(csr, np.array([[0, 1]]))
    assert score == approx(expected)


@pytest.mark.parametrize('scale, nusers, nitems', ((1, 10, 20), (5, 10, 20)))
def test_compute_cosine_similarity(scale, nusers, nitems):
    rating = np.random.randint(scale + 1, size=nusers * nitems)
    user = np.repeat(range(nusers), nitems)
    data = pd.DataFrame({'user': user, 'rating': rating})
    data['item'] = data.groupby('user').cumcount()
    data = data[data['rating'] > 0]
    csr = csr_matrix((data['rating'], (data['user'], data['item'])))
    pairs = np.array([[1, 2], [2, 3], [5, 9]])
    expected = []
    for p in pairs:
        lp = data[data['user'] == p[0]].drop('user', axis=1)
        rp = data[data['user'] == p[1]].drop('user', axis=1)
        merged = lp.merge(rp, how='outer', on='item',
                          suffixes=['_l', '_r'])
        merged.fillna(0, inplace=True)
        score = (merged['rating_l'] @ merged['rating_r'] /
                 np.sqrt((merged['rating_l'] @ merged['rating_l']) *
                         (merged['rating_r'] @ merged['rating_r'])))
        expected.append(score)
    actual = compute_cosine_similarity_pd(data, pairs)
    actual_sparse = pairwise_cosine_similarity(csr, pairs)
    actual_sparse_split_join = pairwise_cosine_similarity_split_join(
        csr, pairs, sections=2
    )
    assert actual == approx(expected)
    assert actual_sparse == approx(expected)
    assert actual_sparse_split_join == approx(expected)

    pairs = np.array([[1, 2], [5, 9], [10, 5], [19, 2]])
    expected = []
    for p in pairs:
        lp = data[data['item'] == p[0]].drop('item', axis=1)
        rp = data[data['item'] == p[1]].drop('item', axis=1)
        merged = lp.merge(rp, how='outer', on='user',
                          suffixes=['_l', '_r'])
        merged.fillna(0, inplace=True)
        score = (merged['rating_l'] @ merged['rating_r'] /
                 np.sqrt((merged['rating_l'] @ merged['rating_l']) *
                         (merged['rating_r'] @ merged['rating_r'])))
        expected.append(score)
    actual = compute_cosine_similarity_pd(data, pairs, 'item', 'user')
    actual_sparse = pairwise_cosine_similarity(csr, pairs, axis=1)
    actual_sparse_split_join = pairwise_cosine_similarity_split_join(
        csr, pairs, axis=1, sections=2
    )
    assert actual == approx(expected)
    assert actual_sparse == approx(expected)
    assert actual_sparse_split_join == approx(expected)


@pytest.mark.parametrize('frac, sample_func',
                         product([0.2, 0.3, 0.4],
                                 [_sample_pairs, _sample_pairs_nbp]))
def test_sample_pairs_size(frac, sample_func):
    result = []
    n = 2000
    for i in range(10):
        expected_size = int(n * (n - 1) / 2 * frac)
        actual_size = len(sample_func(np.arange(n), frac))
        p = actual_size / expected_size
        result.append(p)
    m = np.mean(result)
    assert m == approx(0.9, abs=0.1)


@pytest.mark.parametrize('metric', ('item-pop', 'user-act', 'ucorr', 'icorr'))
def test_calibrator_score(metric):
    """
    Test consistency between parallel runs and sequential runs for each metric.
    """
    ntimes = 20
    pref = LatentDirichletAllocation(50, 50, 20, 20, 0.5, 0.5)
    truncated_pareto = TruncatedPareto(10, 0.5)
    obs = PopularityObservation(truncated_pareto)
    data = ml_100k().sample(1000)
    ds = DataStats(data)
    calibrator = Calibrator(pref, obs, ds, 10)
    score = calibrator.score(metric, ntimes=ntimes)
    func_dict = {'item-pop': calibrator._compute_popularity_activity_similarity,
                 'user-act': calibrator._compute_popularity_activity_similarity,
                 'ucorr': calibrator._compute_correlation_similarity,
                 'icorr': calibrator._compute_correlation_similarity}
    args_dict = {'item-pop': ('item',), 'user-act': ('user',),
                 'ucorr': ('user', 'item'), 'icorr': ('item', 'user')}

    expected_scores = []
    for i in range(ntimes):
        simulated_data = calibrator.generate_simulated_data()
        expected = func_dict[metric](simulated_data, *args_dict[metric])
        expected_scores.append(expected)
    expected_score = np.mean(expected_scores)
    assert score == approx(expected_score, rel=3e-2)


def test_calibrator_score_all():
    """
    Test consistency between parallel runs and sequential runs for metric `all`.
    """
    ntimes = 10
    pref = LatentDirichletAllocationCSR(50, 50, 20, 20, 0.5, 0.5)
    truncated_pareto = TruncParetoProfile(10, 0.5, upper=737)
    obs = PopularityObservationCSR(truncated_pareto)
    data = ml_100k().sample(1000)
    ds = DataStats(data)
    # parallel runs
    calibrator = CalibratorCSR(pref, obs, ds, 10)
    score = calibrator.score('all', ntimes=ntimes)
    func_dict = {'item-pop': calibrator._compute_popularity_activity_similarity,
                 'user-act': calibrator._compute_popularity_activity_similarity,
                 'ucorr': calibrator._compute_correlation_similarity,
                 'icorr': calibrator._compute_correlation_similarity}
    args_dict = {'item-pop': (1,), 'user-act': (0,),
                 'ucorr': (0,), 'icorr': (1,)}
    # sequential runs
    expected_scores = []
    for i in range(ntimes):
        simulated_data = calibrator.generate_simulated_data()
        expected = 0
        for metric, func in func_dict.items():
            expected += func(simulated_data, *args_dict[metric])
        expected_scores.append(expected)
    expected_score = np.mean(expected_scores)
    assert score == approx(expected_score, rel=2e-1)


def test_consistency_pandas_csr_imp():
    data = ml_100k().sample(1000)
    ds = DataStats(data)
    ntimes = 50
    # lda pop models
    # CSR implementation
    pref_csr = LatentDirichletAllocationCSR(50, 50, 20, 20, 0.5, 0.5)
    trunc_pareto_csr = TruncParetoProfile(10, 0.5, upper=737)
    obs_csr = PopularityObservationCSR(trunc_pareto_csr)
    calibrator_csr = CalibratorCSR(pref_csr, obs_csr, ds, 10)

    # pandas implementation
    pref = LatentDirichletAllocation(50, 50, 20, 20, 0.5, 0.5)
    truncated_pareto = TruncatedPareto(10, 0.5, upper=737)
    obs = PopularityObservation(truncated_pareto)
    calibrator = Calibrator(pref, obs, ds, 10)
    for metric in ['item-pop', 'user-act', 'ucorr', 'icorr', 'all']:
        score_csr = calibrator_csr.score(metric, ntimes=ntimes)
        score = calibrator.score(metric, ntimes=ntimes)
        assert score_csr == approx(score, rel=2e-1)

    # lda unif models
    # CSR implementation
    pref_csr = LatentDirichletAllocationCSR(50, 50, 20, 20, 0.5, 0.5)
    trunc_pareto_csr = TruncParetoProfile(10, 0.5, upper=737)
    obs_csr = UniformObservationCSR(trunc_pareto_csr)
    calibrator_csr = CalibratorCSR(pref_csr, obs_csr, ds, 10)

    # pandas implementation
    pref = LatentDirichletAllocation(50, 50, 20, 20, 0.5, 0.5)
    truncated_pareto = TruncatedPareto(10, 0.5, upper=737)
    obs = UniformObservation(truncated_pareto)
    calibrator = Calibrator(pref, obs, ds, 10)
    for metric in ['item-pop', 'user-act', 'ucorr', 'icorr', 'all']:
        score_csr = calibrator_csr.score(metric, ntimes=ntimes)
        score = calibrator.score(metric, ntimes=ntimes)
        assert score_csr == approx(score, rel=2e-1)
