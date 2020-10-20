"""
Simulations

Initial runs and analysis of offline evaluation simulations.
"""
import sys
import numpy as np
import pandas as pd
import pickle

from lenskit import crossfold as xf
from lenskit import batch
from lenskit import topn
from lenskit.algorithms import basic
from lenskit.metrics import topn as tnmetrics

from simulation_utils.utils import eval_algorithm, compute_metrics, \
    truncated_pareto
from simulation_utils.recommenders import Oracle
from simulation_utils.preference import generate_uniform, generate_ibp_itemwise
from simulation_utils.observation import sample_uniform_n, sample_popular_n


def main():
    # data generations
    gen_unif = generate_uniform
    gen_ibp = generate_ibp_itemwise
    sample_unif = sample_uniform_n
    sample_pop = sample_popular_n
    # algorithms
    random_algo = basic.Random()
    popular_algo = basic.Popular()
    oracle_algo = Oracle(None)
    # run simulations
    ibp_unif = run_sims(1, 1, gen_ibp, sample_unif,
                        [random_algo, popular_algo, oracle_algo],
                        ['precision', 'recall', 'recip_rank', 'ndcg'])
    ibp_pop = run_sims(10, 2, gen_ibp, sample_pop,
                       [random_algo, popular_algo, oracle_algo],
                       ['precision', 'recall', 'recip_rank', 'ndcg'])
    unif_unif = run_sims(10, 2, gen_unif, sample_unif,
                         [random_algo, popular_algo, oracle_algo],
                         ['precision', 'recall', 'recip_rank', 'ndcg'])
    unif_pop = run_sims(10, 2, gen_unif, sample_pop,
                        [random_algo, popular_algo, oracle_algo],
                        ['precision', 'recall', 'recip_rank', 'ndcg'])
    # save data
    with open('build/ibp-unif-eval-results.pickle', 'wb') as f:
        pickle.dump(ibp_unif, f)
    with open('build/ibp_pop-eval-results.pickle', 'wb') as f:
        pickle.dump(ibp_pop, f)
    with open('build/unif_unif-eval-results.pickle', 'wb') as f:
        pickle.dump(unif_unif, f)
    with open('build/unif_pop-eval-results.pickle', 'wb') as f:
        pickle.dump(unif_pop, f)


def run_experiment(generator, sampler, algorithms, metrics):
    """
    A function to run a single experiment, including:
        generate user true preferences,
        sample observations from the generated preferences
        train algorithms on observations or preferences (if Oracle)
        recommend items to test users
        compute evaluation metrics using true preferences and observations

    Args:
        generator (function):
        sampler:
        algorithms:
        metrics:

    Returns:

    """
    # TODO: make generator and sampler parameters configurable
    pref = generator(100, 200)
    pref['rating'] = 1
    pareto_params = {'dist_func': truncated_pareto,
                     'use_cap': True,
                     'm': 20.00000045,
                     'alpha': 0.510528}
    obs = sampler(pref, **pareto_params)
    # let's get some irrelevant items
    obs_dens = (obs.pivot(index='user', columns='item', values='rating')
                .reset_index()
                .melt(id_vars='user', var_name='item', value_name='rating'))
    obs_dens['item'] = obs_dens['item'].astype(np.int64)
    # split data
    splits = xf.partition_users(obs_dens, 1, xf.SampleFrac(0.2))
    train, test = next(splits)
    obs_train = train.dropna()
    obs_test = test.dropna()
    pref_test = (test.merge(pref, how='left', 
                            on=['user', 'item'], suffixes=['_obs', ''])
                 .drop('rating_obs', axis=1)
                 .dropna())
    # get algorithm names
    # TODO: fix Random algorithm's `__str__` to avoid this.
    algo_names = map(lambda x: str(x).split('.')[-1].split()[0], algorithms)
    nalgo = len(algorithms)
    # map the `eval_algorithm` function to each algorithm
    users = obs_test['user'].unique()
    recs = map(eval_algorithm, algo_names, algorithms, [obs_train]*nalgo,
               [pref]*nalgo, [users]*nalgo, [50]*nalgo)
    recs = pd.concat(recs, axis=0, ignore_index=True)
    # compute metrics based on the test data (observations and true preferences)
    eval_obs = compute_metrics(recs, metrics, obs_test)
    eval_obs = (eval_obs.drop('user', axis=1)
                .groupby('algorithm', as_index=False)
                .mean())
    eval_pref = compute_metrics(recs, metrics, pref_test)
    eval_pref = (eval_pref.drop('user', axis=1)
                 .groupby('algorithm', as_index=False)
                 .mean())
    eval_res = eval_obs.merge(eval_pref, how='left',
                              on='algorithm', suffixes=['_obs', '_pref'])
    return eval_res


def _parallel_wrapper(args):
    from datetime import datetime
    # make children processes use different seeds
    seed = datetime.now().microsecond
    np.random.seed(seed)
    eval_res = run_experiment(*args)
    return eval_res


def run_sims(ntimes=100, nthread=None, *args):
    if ntimes <= 0:
        raise ValueError("ntimes must be greater than 0")
    if len(args) == 0:
        raise ValueError("args must not be empty")
    from multiprocessing import cpu_count
    from multiprocessing import Pool
    if nthread and nthread >= cpu_count():
        raise ValueError('nthread must be less than the system cpu count')
    # create args iterable for each run 
    args_list = [args] * ntimes
    results = None
    with Pool(nthread) as p:
        result = p.map(_parallel_wrapper, args_list)
        results = pd.concat(result, axis=0, ignore_index=True)
    return results


if __name__ == '__main__':
    main()
