import numpy as np
import pandas as pd
from pytest import approx
from lenskit import crossfold as xf
from lenskit.algorithms.basic import Popular
from lenskit.algorithms.basic import Random
from lenskit.metrics import topn as tnmetrics

from simulation_utils.preference import generate_ibp
from simulation_utils.observation import sample_uniform
from simulation_utils.recommenders import Oracle
from simulation_utils import utils


def data_setup():
    pref = generate_ibp(100, 100, 0.5)
    pref['rating'] = 1
    obs = sample_uniform(pref, 0.2)
    splits = xf.partition_users(obs, 1, xf.SampleFrac(0.5))
    train, test = next(splits)
    return pref, obs, train, test


def test_eval_algorithm():
    pref, obs, train, test = data_setup()
    users = test['user'].unique()
    random_algo = Random()
    oracle_algo = Oracle(None)
    random_recs = utils.eval_algorithm('random', random_algo, train, pref, users, 20)
    assert set(random_recs.columns) == {'algorithm', 'user', 'item', 'rank'}
    oracle_recs = utils.eval_algorithm('oracle', oracle_algo, train, pref, users, 20)
    assert set(oracle_recs.columns) == {'algorithm', 'user', 'item', 'rank'}

    nrec_items = random_recs.groupby(['algorithm', 'user']).count()['item'].unique()
    assert nrec_items == 20


def test_compute_ndcg():
    pref, obs, train, test = data_setup()
    users = test['user'].unique()
    popular_algo = Popular()
    popular_recs = utils.eval_algorithm('popular', popular_algo, train, pref, users, 20)
    rec_rel = popular_recs.merge(test, how='left', on=['user', 'item'])
    rec_rel.loc[rec_rel.rating.isna(), 'rating'] = 0
    expected_score = rec_rel.groupby(['algorithm', 'user']).rating.apply(tnmetrics.ndcg)
    expected_score.rename('ndcg', inplace=True)
    expected_score = expected_score.reset_index()
    actual_score = utils.compute_metrics(popular_recs, ['ndcg'], test)
    assert expected_score.equals(actual_score)


def test_compute_recall():
    pref, obs, train, test = data_setup()
    users = test['user'].unique()
    popular_algo = Popular()
    popular_recs = utils.eval_algorithm('popular', popular_algo, train, pref, users, 20)
    # compute recall
    nrel_test = (test.groupby('user')['item']
                 .count()
                 .rename('ntest_rel')
                 .reset_index())
    nrel_rec = (popular_recs.merge(test, how='left')
                .groupby(['algorithm', 'user'])['rating']
                .count()
                .rename('nrec_rel')
                .reset_index())
    result = nrel_rec.merge(nrel_test, how='left')
    result['recall'] = result['nrec_rel'] / result['ntest_rel']
    expected_res = result.drop(['nrec_rel', 'ntest_rel'], axis=1)

    # compute recall using the lkpy built-in method
    rec_items = (popular_recs.drop('rank', axis=1)
                 .groupby(['algorithm', 'user'])['item']
                 .apply(np.array)
                 .rename('rec_item')
                 .reset_index())
    rel_items = (test.drop('rating', axis=1)
                 .groupby('user')['item']
                 .apply(set)
                 .rename('rel_item')
                 .reset_index())
    rec_rel = rec_items.merge(rel_items, how='left')
    expected_score = (rec_rel.groupby(['algorithm', 'user'])[['rec_item', 'rel_item']]
                      .apply(lambda x: tnmetrics.recall(x.iloc[0, 0], x.iloc[0, 1]))
                      .rename('recall')
                      .reset_index())
    # compute recall using the project method
    actual_score = utils.compute_metrics(popular_recs, ['recall'], test)
    assert expected_res.equals(actual_score)
    assert expected_score.equals(actual_score)


def test_compute_user_activity():
    data = pd.DataFrame({'user': [1, 1, 1, 2, 2, 2, 3, 3],
                         'item': [1, 2, 3, 2, 3, 5, 2, 4]})
    # test the density sums to 1
    ua_density = utils.compute_user_activity(data)
    assert sum(ua_density) == approx(1, abs=1e-6)

    # test profile size counts
    ua_actual = utils.compute_user_activity(data, normed=False)
    ua_expected = np.array([1, 2, 3, 1, 1])
    assert all(ua_expected == ua_actual)

    # test percentile
    ua_count = ua_actual - 1
    ua_count = ua_count.astype(int)
    profile_size = np.array(range(1, 6))
    ua_actual = np.repeat(profile_size, ua_count)
    percentile_actual = np.percentile(ua_actual, [25, 50, 75])
    ua_expected = data.groupby('user', as_index=False).count()
    ua_expected_stats = ua_expected['item'].describe()
    percentile_expected = ua_expected_stats.loc[['25%', '50%', '75%']].values
    assert np.allclose(percentile_expected, percentile_actual)
