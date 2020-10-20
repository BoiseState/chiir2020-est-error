import inspect
import os
from collections import namedtuple
from invoke import task
from itertools import repeat
import logging
import logging.config
import datetime
from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle

from joblib import Parallel, delayed
from lenskit import crossfold as xf
from lenskit.algorithms.basic import Popular
from lenskit.metrics.topn import precision, recall, recip_rank, ndcg
from lenskit.topn import RecListAnalysis
import skopt
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from simulation_utils import logutils
from simulation_utils.calibration import calibrate_data, Calibrator, \
    CalibratorCSR
from simulation_utils.datasets import *
from simulation_utils.preference import generate_uniform, \
    generate_ibp_itemwise, generate_pf, generate_lda, \
    LatentDirichletAllocation, UniformPreference, \
    LatentDirichletAllocationCSR,  IndianBuffetProcessCSR, generate_lda_df, \
    UniformPreferenceCSR
from simulation_utils.observation import sample_popular_n, sample_uniform_n, \
    PopularityObservation, UniformObservation, TruncParetoProfile, \
    PopularityObservationCSR, UniformObservationCSR
from simulation_utils.recommenders import Oracle, Random
from simulation_utils.utils import truncated_beta_binomial, truncated_pareto, \
    eval_algorithm, TruncatedPareto

logutils.setup()
_log = logging.getLogger('simulation_utils.runner')

# sys.stdout = memory_profiler.LogFile('simulation_utils.runner', True)

RLA = RecListAnalysis()
RLA.add_metric(precision)
RLA.add_metric(recall)
RLA.add_metric(recip_rank)
RLA.add_metric(ndcg)

DataSets = namedtuple('DataSets', ('ml_100k', 'ml_1m', 'ml_10m', 'ml_20m',
                                   'bx_implicit', 'bx_explicit',
                                   'az_book', 'az_music',
                                   'az_music_instruments',
                                   'az_music_5core', 'steam_video_game',
                                   'steam_video_game_5core'))
DATA_SETS = DataSets(ml_100k, ml_1m, ml_10m, ml_20m, bx_implicit, bx_explicit,
                     az_book, az_music, az_music_instruments, az_music_5core,
                     steam_video_game, steam_video_game_5core)
_build_path = Path('build')


def make_filename(fdir='build', extension='pkl', include_time=False,
                  exclude_args=('c',)):
    """
    Combines the caller name, arguments and current time to make a file name.

    """
    frame = inspect.currentframe().f_back
    func_name = inspect.getframeinfo(frame).function
    args, _, _, locals_ = inspect.getargvalues(frame)
    arg_values = '-'.join(
        [str(locals_[arg]) for arg in args if arg not in exclude_args])
    if include_time:
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        fn = '-'.join([func_name, arg_values, start_time])
    else:
        fn = '-'.join([func_name, arg_values])
    fn = '.'.join([fn, extension])
    fpath = os.path.join(fdir, fn)
    return fpath


@task
def grid_search_unif_unif(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())

    generator = generate_uniform
    frac = np.random.uniform(0, 1, size=20)
    frac = [x for x in np.unique(frac) if x != 0]

    generator_params = {'nusers': [nusers],
                        'nitems': [nitems],
                        'frac': frac}
    sampler = sample_uniform_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=None)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_unif_unif_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def grid_search_unif_pop(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())

    generator = generate_uniform
    frac = np.random.uniform(0, 1, size=20)
    frac = np.unique(frac)

    generator_params = {'nusers': [nusers],
                        'nitems': [nitems],
                        'frac': frac}
    sampler = sample_popular_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=None)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_unif_pop_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def grid_search_ibp_unif(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())

    generator = generate_ibp_itemwise
    frac = np.random.uniform(0, 1, size=20)
    frac = np.unique(frac)
    alpha = [nitems * x for x in frac if x != 0]
    # c = np.random.exponential(size=10)
    c = np.random.uniform(0, 1, size=20)
    c = np.unique(np.append(c, 1.0)).tolist()
    sigma = np.random.uniform(0, 1, size=10)
    sigma = np.unique(np.append(sigma, 0)).tolist()

    generator_params = {'nusers': [nusers],
                        'alpha': alpha,
                        'c': c,
                        'sigma': sigma}
    sampler = sample_uniform_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=None)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_ibp_unif_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def grid_search_ibp_pop(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())

    generator = generate_ibp_itemwise
    # frac = 1 / (np.log(nusers) + 0.5772)
    # frac = np.random.exponential(size=20)
    frac = np.random.uniform(0, 1, size=20)
    frac = np.unique(frac)
    alpha = [nitems * x for x in frac if x != 0]
    # c = np.random.exponential(size=10)
    c = np.random.uniform(0, 1, size=20)
    c = np.unique(np.append(c, 1.0)).tolist()
    sigma = np.random.uniform(0, 1, size=10)
    sigma = np.unique(np.append(sigma, 0)).tolist()

    generator_params = {'nusers': [nusers],
                        'alpha': alpha,
                        'c': c,
                        'sigma': sigma}
    sampler = sample_popular_n
    # sampler = sample_uniform_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=None)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_ibp_pop_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def grid_search_pf_pop(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())

    generator = generate_pf
    generator_params = {'nusers': [nusers],
                        'nitems': [nitems]}
    sampler = sample_popular_n
    # sampler = sample_uniform_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=1)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_pf_pop_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def grid_search_lda_pop(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())
    frac = np.random.uniform(0, 1, size=10)
    frac = np.unique(frac)
    lam = [nitems * x for x in frac if x != 0]
    k = np.random.randint(200, size=10)
    k = np.unique(k)
    a = np.random.exponential(1, size=10)
    a = np.unique(a)

    generator = generate_lda
    generator_params = {'nusers': [nusers],
                        'nitems': [nitems],
                        'k': k,
                        'lam': lam,
                        'a': a,
                        'b': a,
                        'c': a,
                        'd': a}
    sampler = sample_popular_n
    # sampler = sample_uniform_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=None)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_lda_pop_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def grid_search_lda_unif(c):
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())
    frac = np.random.uniform(0, 1, size=10)
    frac = np.unique(frac)
    lam = [nitems * x for x in frac if x != 0]
    k = np.random.randint(200, size=10)
    k = np.unique(k)
    a = np.random.exponential(1, size=10)
    a = np.unique(a)

    generator = generate_lda
    generator_params = {'nusers': [nusers],
                        'nitems': [nitems],
                        'k': k,
                        'lam': lam,
                        'a': a,
                        'b': a,
                        'c': a,
                        'd': a}
    # sampler = sample_popular_n
    sampler = sample_uniform_n

    beta_binomial_params = {'dist_func': [truncated_beta_binomial],
                            'use_cap': [True],
                            'a': [1.6638872003071793e-05],
                            'b': [2.8157887296158077],
                            'n': [737]}

    pareto_params = {'dist_func': [truncated_pareto],
                     'use_cap': [True],
                     'm': [20.00000045],
                     'alpha': [0.510528]}

    sampler_params = [pareto_params, beta_binomial_params]

    # sampler_params = [beta_binomial_params]
    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")
    results = calibrate_data(data, generator, sampler,
                             generator_params, sampler_params, nthread=None)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")
    fn = f'calibrations_lda_unif_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)

    with open(fdir, 'wb') as f:
        pickle.dump(results, f)


@task
def gp_minimize_unif_unif(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = UniformPreference(nusers, nitems, 100)
        trunc_pareto = TruncatedPareto(20.00000045, 0.510528, 737)
        obs = UniformObservation(trunc_pareto)
        calibrator = Calibrator(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.9, 1.0)
        m_high = max(minua * 1.1, m_low * 1.1)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(50, m_high * 1.1)
        upper_high = max(maxua * 1.1, upper_low * 1.1)
        # search space dimensions are inclusive in skopt numbers!
        space = [Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score(metric, ntimes=n_jobs)
            return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreference(nusers, nitems, lam).generate()
        trunc_pareto = TruncatedPareto(m, alpha, upper)
        df = UniformObservation(trunc_pareto).sample(pref)
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_unif_pop(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = UniformPreference(nusers, nitems, 100)
        trunc_pareto = TruncatedPareto(20.00000045, 0.510528, 737)
        obs = PopularityObservation(trunc_pareto)
        calibrator = Calibrator(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.9, 1.0)
        m_high = max(minua * 1.1, m_low * 1.1)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(50, m_high * 1.1)
        upper_high = max(maxua * 1.1, upper_low * 1.1)
        # search space dimensions are inclusive in skopt numbers!
        space = [Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score(metric, ntimes=n_jobs)
            return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreference(nusers, nitems, lam).generate()
        trunc_pareto = TruncatedPareto(m, alpha, upper)
        df = PopularityObservation(trunc_pareto).sample(pref)
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_lda_pop(c, dataname='ml_100k', metric='ucorr', n_jobs=10):
    ds = DataStats(getattr(DATA_SETS, dataname))
    nusers = ds.nusers
    nitems = ds.nitems
    pref = LatentDirichletAllocation(nusers, nitems, 100, 85.0, 1, 1)
    trunc_pareto = TruncatedPareto(20.00000045, 0.510528)
    obs = PopularityObservation(trunc_pareto)
    calibrator = Calibrator(pref, obs, ds, n_jobs)
    maxua = ds.user_activity.index.max()
    minua = ds.user_activity.index.min()
    # m is tuned for the lower bound of the user activity
    m_low = max(minua * 0.9, 1.0)
    m_high = max(minua * 1.1, m_low * 1.1)
    # upper is tuned for the upper bound of the user activity
    upper_low = max(maxua * 0.9, m_high * 1.1)  # at least higher that m_high
    upper_high = max(maxua * 1.1, upper_low * 1.1)
    # search space dimensions are inclusive in skopt numbers!
    space = [Integer(5, 200, name='pref__k'),
             Real(100, 5000, prior='log-uniform', name='pref__lam'),
             Real(1e-2, 1, name='pref__a'),
             Real(1e-2, 1, name='pref__b'),
             Real(m_low, m_high, name='obs__dist_func__m'),
             Real(1e-1, 20, name='obs__dist_func__alpha'),
             Real(upper_low, upper_high, prior='log-uniform',
                  name='obs__dist_func__upper')]

    @use_named_args(space)
    def objective(**params):
        calibrator.set_params(**params)
        return calibrator.score(metric)

    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")

    res_gp = gp_minimize(objective, space, n_calls=150)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")

    fpath = make_filename()
    skopt.dump(res_gp, fpath, store_objective=False)


@task
def gp_minimize_lda_unif(c, dataname='ml_100k', metric='ucorr', n_jobs=10):
    ds = DataStats(getattr(DATA_SETS, dataname))
    nusers = ds.nusers
    nitems = ds.nitems
    pref = LatentDirichletAllocation(nusers, nitems, 100, 85.0, 1, 1)
    trunc_pareto = TruncatedPareto(20.00000045, 0.510528)
    obs = UniformObservation(trunc_pareto)
    calibrator = Calibrator(pref, obs, ds, n_jobs)
    maxua = ds.user_activity.index.max()
    minua = ds.user_activity.index.min()
    # m is tuned for the lower bound of the user activity
    m_low = max(minua * 0.8, 1.0)
    m_high = max(minua * 1.2, m_low * 1.2)
    # upper is tuned for the upper bound of the user activity
    upper_low = max(maxua * 0.8, m_high * 1.2)  # at least higher that m_high
    upper_high = max(maxua * 1.2, upper_low * 1.2)
    # search space dimensions are inclusive!
    space = [Integer(10, 500, name='pref__k'),
             Real(20, 5000, prior='log-uniform', name='pref__lam'),
             Real(1e-2, 1000, prior='log-uniform', name='pref__a'),
             Real(1e-2, 1000, prior='log-uniform', name='pref__b'),
             Real(10, 30, name='obs__dist_func__m'),
             Real(1e-1, 1000, prior='log-uniform',
                  name='obs__dist_func__alpha'),
             Real(50.0, 1000, prior='log-uniform',
                  name='obs__dist_func__upper')]

    @use_named_args(space)
    def objective(**params):
        calibrator.set_params(**params)
        score = calibrator.score(metric)
        _log.debug(f'one round finished at {datetime.datetime.now()}')
        return score

    start_time = datetime.datetime.now()
    _log.info(f"calibration starts at {start_time}")

    res_gp = gp_minimize(objective, space, n_calls=200)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"calibration ends at {end_time} with total time {duration}")

    fpath = make_filename()
    skopt.dump(res_gp, fpath, store_objective=False)


@task
def gp_minimize_unif_unif_csr(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = UniformPreferenceCSR(nusers, nitems, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score(metric, ntimes=n_jobs)
            return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_random_starts=25, n_calls=150)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam).generate()
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_unif_pop_csr(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = UniformPreferenceCSR(nusers, nitems, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score(metric, ntimes=n_jobs)
            return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_random_starts=25, n_calls=150)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam).generate()
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_lda_unif_csr(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = LatentDirichletAllocationCSR(nusers, nitems, 100, 85.0, 1, 1)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive in skopt numbers!
        space = [Integer(5, 200, name='pref__k'),
                 Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(1e-2, 1, name='pref__a'),
                 Real(1e-2, 1, name='pref__b'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score(metric, ntimes=n_jobs)
            return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_random_starts=25, n_calls=150)
        _log.info(f"calibration ends")

        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = (LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
                .generate())
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_lda_pop_csr(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = LatentDirichletAllocationCSR(nusers, nitems, 100, 85.0, 1, 1)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 737)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive in skopt numbers!
        space = [Integer(5, 200, name='pref__k'),
                 Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(1e-2, 1, name='pref__a'),
                 Real(1e-2, 1, name='pref__b'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score(metric, ntimes=n_jobs)
            return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_random_starts=25, n_calls=150)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = (LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
                .generate())
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_ibp_unif_csr(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = IndianBuffetProcessCSR(nusers, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(10, 1000, prior='log-uniform', name='pref__alpha'),
                 Real(1e-2, 100, prior='log-uniform', name='pref__c'),
                 Real(0, 0.99, name='pref__sigma'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            try:
                calibrator.set_params(**params)
            except ValueError:
                return 1e6  # invalid params return a large score
            else:
                _log.debug(f'calibrate with params {params}')
                score = calibrator.score(metric, ntimes=n_jobs)
                return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_random_starts=25, n_calls=150)
        _log.info(f"calibration ends")

        # save the best params
        best_params = [nusers, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma).generate()
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = UniformObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def gp_minimize_ibp_pop_csr(c, dataname='ml_100k', metric='ucorr', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('gp_minimize', 'params')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = IndianBuffetProcessCSR(nusers, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 737)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(10, 1000, prior='log-uniform', name='pref__alpha'),
                 Real(1e-2, 100, prior='log-uniform', name='pref__c'),
                 Real(0, 0.99, name='pref__sigma'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            try:
                calibrator.set_params(**params)
            except ValueError:
                return 1e6
            else:
                _log.debug(f'calibrate with params {params}')
                score = calibrator.score(metric, ntimes=n_jobs)
                return score

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_random_starts=25, n_calls=150)
        _log.info(f"calibration ends")

        # save the best params
        best_params = [nusers, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma).generate()
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('gp_minimize_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


def run_experiment(pref_gen, obs_gen, pref_kwargs, obs_kwargs, algorithms,
                   rec_list_analysis):
    """
    Runs an single evaluation experiment.

    Args:
        pref_gen:
        obs_gen:
        pref_kwargs:
        obs_kwargs:
        algorithms (:obj:`list` of :obj:`lenskit.algorithms.Algorithm`):
        rec_list_analysis (lenskit.topn.RecListAnalysis):

    Returns:

    """
    np.random.seed()
    pref = pref_gen(**pref_kwargs)
    pref = pref.sort_values(['user', 'item'])
    if 'rating' not in pref.columns:
        pref['rating'] = 1
    obs = obs_gen(pref, **obs_kwargs)
    # let's get some irrelevant items
    # obs_dens = (obs.pivot(index='user', columns='item', values='rating')
    #             .reset_index()
    #             .melt(id_vars='user', var_name='item', value_name='rating'))
    # obs_dens['item'] = obs_dens['item'].astype(np.int64)
    # split data
    splits = xf.partition_users(obs, 1, xf.SampleFrac(0.2))

    obs_train, obs_test = next(splits)
    users = obs_test['user'].unique()
    # preference test set: relevant preferences not in the train set.
    pref_test = pref.merge(obs_train[obs_train['user'].isin(users)],
                           how='left',
                           on=['user', 'item'],
                           suffixes=['', '_obs'])
    pref_test = pref_test.loc[pref_test['rating_obs'].isna(),
                              ['user', 'item', 'rating']]

    algo_names = map(str, algorithms)
    recs = map(eval_algorithm, algo_names, algorithms, repeat(obs_train),
               repeat(pref), repeat(users), repeat(50))
    recs = pd.concat(recs, axis=0, ignore_index=True)
    # compute topn metrics
    obs_eval = rec_list_analysis.compute(recs, obs_test)
    obs_eval = obs_eval.groupby(level='algorithm').mean().reset_index()
    pref_eval = rec_list_analysis.compute(recs, pref_test)
    pref_eval = pref_eval.groupby(level='algorithm').mean().reset_index()
    eval_res = obs_eval.merge(pref_eval, how='left', on='algorithm',
                              suffixes=['_obs', '_pref'])
    return eval_res


def run_simulation(args, times=100, nprocs=None, fdir='build/eval-res.pkl'):
    """
    Runs simulations `times` times using `nprocs` processes
    Args:
        args (:obj:`list` of arguments): The arguments passed in run_experiment.
        times (int): The number of runs.
        nprocs (int): The number of processes
        fdir (str): The file path used to save results.

    Returns:
        pandas.DataFrame: A data frame results. Rows: result for each
            recommender in each run. columns: evaluation metric scores.
    """
    if isinstance(nprocs, str):
        nprocs = int(nprocs)
    ncpus = cpu_count()
    if nprocs and nprocs >= ncpus:
        nprocs = ncpus
        logging.warning(f'nprocs is greater than cpu counts {ncpus}, '
                        f'{nprocs} processes are used')
    with Pool(nprocs) as p:
        result = p.starmap(run_experiment, repeat(args, times))
    results = pd.concat(result, axis=0, ignore_index=True)
    with open(fdir, 'wb') as f:
        pickle.dump(results, f)
    return results


def single_evaluation(preference, observation, algorithms, rec_list_analysis):
    """
    Runs an single evaluation experiment.

    Args:
        preference (simulation_utils.preference.PreferenceModel): The
            preference model
        observation (simulation_utils.observation.ObservationModel): The
            observation model
        algorithms (:obj:`list` of :obj:`lenskit.algorithms.Algorithm`):
        rec_list_analysis (lenskit.topn.RecListAnalysis):

    Returns:

    """
    np.random.seed()
    pref = preference.generate()
    obs = observation.sample(pref)
    # convert data to dataframe
    coo = pref.tocoo(copy=False)
    pref = pd.DataFrame({'user': coo.row, 'item': coo.col, 'rating': coo.data})
    coo = obs.tocoo(copy=False)
    obs = pd.DataFrame({'user': coo.row, 'item': coo.col, 'rating': coo.data})
    obs['rating'] = 1
    del coo
    # let's get some irrelevant items
    # obs_dens = (obs.pivot(index='user', columns='item', values='rating')
    #             .reset_index()
    #             .melt(id_vars='user', var_name='item', value_name='rating'))
    # obs_dens['item'] = obs_dens['item'].astype(np.int64)
    # split data
    splits = xf.partition_users(obs, 1, xf.SampleFrac(0.2))

    obs_train, obs_test = next(splits)
    users = obs_test['user'].unique()
    # preference test set: relevant preferences not in the train set.
    pref_test = pref.merge(obs_train[obs_train['user'].isin(users)],
                           how='left',
                           on=['user', 'item'],
                           suffixes=['', '_obs'])
    pref_test = pref_test.loc[pref_test['rating_obs'].isna(),
                              ['user', 'item', 'rating']]

    algo_names = map(str, algorithms)
    recs = map(eval_algorithm, algo_names, algorithms, repeat(obs_train),
               repeat(pref), repeat(users), repeat(50))
    recs = pd.concat(recs, axis=0, ignore_index=True)
    # compute topn metrics
    obs_eval = rec_list_analysis.compute(recs, obs_test)
    obs_eval = obs_eval.groupby(level='algorithm').mean().reset_index()
    pref_eval = rec_list_analysis.compute(recs, pref_test)
    pref_eval = pref_eval.groupby(level='algorithm').mean().reset_index()
    eval_res = obs_eval.merge(pref_eval, how='left', on='algorithm',
                              suffixes=['_obs', '_pref'])
    return eval_res


def run_simulation_csr(args, times=100, n_jobs=-1, fdir='build/eval-res.pkl'):
    """
    Runs simulations `times` times using `nprocs` processes
    Args:
        args (:obj:`list` of arguments): The arguments passed in run_experiment.
        times (int): The number of runs.
        n_jobs (int): The number of processes
        fdir (str): The file path used to save results.

    Returns:
        pandas.DataFrame: A data frame results. Rows: result for each
            recommender in each run. columns: evaluation metric scores.
    """

    result = Parallel(n_jobs=n_jobs)(
        delayed(single_evaluation)(*arg) for arg in repeat(args, times)
    )
    results = pd.concat(result, axis=0, ignore_index=True)
    with open(fdir, 'wb') as f:
        pickle.dump(results, f)
    return results


@task
def simulate_ibp_unif(c, times=100, nprocs=None):
    pref = generate_ibp_itemwise
    res = skopt.load('build/res_gp_icorr_ibp_unif.pkl')
    assert len(res.x) == 6
    alpha, c, sigma, m, obs_alpha, upper = res.x
    pref_kwargs = {'nusers': 943, 'alpha': alpha,
                   'c': c, 'sigma': sigma}
    obs = sample_uniform_n
    obs_kwargs = {'dist_func': truncated_pareto,
                  'use_cap': True,
                  'm': m,
                  'alpha': obs_alpha,
                  'upper': upper}
    algorithms = [Random(), Popular(), Oracle(None)]
    args = [pref, obs, pref_kwargs, obs_kwargs, algorithms, RLA]
    start_time = datetime.datetime.now()
    fn = f'simulations_ibp_unif_{start_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)
    _log.info(f"simulation starts at {start_time}")
    run_simulation(args, times=times, nprocs=nprocs, fdir=fdir)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"simulation ends at {end_time} with total time {duration}")


@task
def simulate_ibp_pop(c, times=100, nprocs=None):
    pref = generate_ibp_itemwise
    res = skopt.load('build/res_gp_icorr_ibp_pop.pkl')
    assert len(res.x) == 6
    alpha, c, sigma, m, obs_alpha, upper = res.x
    pref_kwargs = {'nusers': 943, 'alpha': alpha,
                   'c': c, 'sigma': sigma}
    obs = sample_popular_n
    obs_kwargs = {'dist_func': truncated_pareto,
                  'use_cap': True,
                  'm': m,
                  'alpha': obs_alpha,
                  'upper': upper}
    algorithms = [Random(), Popular(), Oracle(None)]
    args = [pref, obs, pref_kwargs, obs_kwargs, algorithms, RLA]
    start_time = datetime.datetime.now()
    fn = f'simulations_ibp_pop_{start_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
    fdir = os.path.join('build', fn)
    _log.info(f"simulation starts at {start_time}")
    run_simulation(args, times=times, nprocs=nprocs, fdir=fdir)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    _log.info(f"simulation ends at {end_time} with total time {duration}")


@task
def simulate_lda_unif(c, times=100, nprocs=None):
    logpath = make_filename(extension='.log', include_time=False)
    with logutils.LogFile(logpath):
        pref = generate_lda_df
        res = skopt.load('build/res_gp_icorr_lda_unif.pkl')
        assert len(res.x) == 7
        k, lam, a, b, m, alpha, upper = res.x
        pref_kwargs = {'nusers': 943, 'nitems': 1682,
                       'k': k, 'lam': lam, 'a': a, 'b': b}
        obs = sample_uniform_n
        obs_kwargs = {'dist_func': truncated_pareto,
                      'use_cap': True,
                      'm': m,
                      'alpha': alpha,
                      'upper': upper}
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, pref_kwargs, obs_kwargs, algorithms, RLA]
        start_time = datetime.datetime.now()
        fn = f'simulations_lda_unif_{start_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
        fdir = os.path.join('build', fn)
        _log.info(f"simulation starts at {start_time}")
        run_simulation(args, times=times, nprocs=nprocs, fdir=fdir)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        _log.info(f"simulation ends at {end_time} with total time {duration}")


@task
def simulate_lda_pop(c, times=100, nprocs=None):
    logpath = make_filename(extension='.log', include_time=False)
    with logutils.LogFile(logpath):
        pref = generate_lda_df
        res = skopt.load('build/res_gp_icorr_lda_pop.pkl')
        assert len(res.x) == 7
        k, lam, a, b, m, alpha, upper = res.x
        pref_kwargs = {'nusers': 943, 'nitems': 1682,
                       'k': k, 'lam': lam, 'a': a, 'b': b}
        obs = sample_popular_n
        obs_kwargs = {'dist_func': truncated_pareto,
                      'use_cap': True,
                      'm': m,
                      'alpha': alpha,
                      'upper': upper}
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, pref_kwargs, obs_kwargs, algorithms, RLA]
        start_time = datetime.datetime.now()
        fn = f'simulations_lda_pop_{start_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl'
        fdir = os.path.join('build', fn)
        _log.info(f"simulation starts at {start_time}")
        run_simulation(args, times=times, nprocs=nprocs, fdir=fdir)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        _log.info(f"simulation ends at {end_time} with total time {duration}")


@task
def simulate_unif_unif(c, times=100, nprocs=None, dataname='ml_1m',
                       metric='icorr', params_path=None):
    logpath = make_filename(extension='log', include_time=False,
                            exclude_args=['c', 'times',
                                          'params_path', 'nprocs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = fpath.replace('simulate', 'params')
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = generate_uniform
        pref_kwargs = {'nusers': nusers, 'nitems': nitems, 'lam': lam}
        obs = sample_uniform_n
        obs_kwargs = {'dist_func': truncated_pareto,
                      'use_cap': True,
                      'm': m,
                      'alpha': alpha,
                      'upper': upper}
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, pref_kwargs, obs_kwargs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation(args, times=times, nprocs=nprocs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_unif_pop(c, times=100, nprocs=None, dataname='ml_1m',
                      metric='icorr', params_path=None):
    logpath = make_filename(extension='log', include_time=False,
                            exclude_args=['c', 'times',
                                          'params_path', 'nprocs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = fpath.replace('simulate', 'params')
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = generate_uniform
        pref_kwargs = {'nusers': nusers, 'nitems': nitems, 'lam': lam}
        obs = sample_popular_n
        obs_kwargs = {'dist_func': truncated_pareto,
                      'use_cap': True,
                      'm': m,
                      'alpha': alpha,
                      'upper': upper}
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, pref_kwargs, obs_kwargs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation(args, times=times, nprocs=nprocs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_unif_unif_csr(c, times=100, n_jobs=-1, dataname='ml_1m',
                           params_path=None):
    logpath = make_filename(extension='log', include_time=False,
                            exclude_args=['c', 'times',
                                          'params_path', 'n_jobs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = _build_path / f'params_unif_unif-{dataname}.pkl'
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto)
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation_csr(args, times=times, n_jobs=n_jobs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_unif_pop_csr(c, times=100, n_jobs=-1, dataname='ml_1m',
                          params_path=None):
    logpath = make_filename(extension='log', include_time=False,
                            exclude_args=['c', 'times',
                                          'params_path', 'n_jobs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = _build_path / f'params_unif_pop-{dataname}.pkl'
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto)
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation_csr(args, times=times, n_jobs=n_jobs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_ibp_unif_csr(ctx, times=100, n_jobs=-1, dataname='ml_1m',
                          params_path=None):
    logpath = make_filename(extension='log', include_time=False,
                            exclude_args=['ctx', 'times',
                                          'params_path', 'n_jobs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = _build_path / f'params_ibp_unif-{dataname}.pkl'
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma)
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = UniformObservationCSR(trunc_pareto)
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation_csr(args, times=times, n_jobs=n_jobs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_ibp_pop_csr(ctx, times=100, n_jobs=-1, dataname='ml_1m',
                         params_path=None):
    logpath = make_filename(extension='log',
                            exclude_args=['ctx', 'times',
                                          'params_path', 'n_jobs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = _build_path / f'params_ibp_pop-{dataname}.pkl'
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma)
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto)
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation_csr(args, times=times, n_jobs=n_jobs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_lda_unif_csr(c, times=100, n_jobs=-1, dataname='ml_1m',
                          params_path=None):
    logpath = make_filename(extension='log',
                            exclude_args=['c', 'times',
                                          'params_path', 'n_jobs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = _build_path / f'params_lda_unif-{dataname}.pkl'
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto)
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation_csr(args, times=times, n_jobs=n_jobs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def simulate_lda_pop_csr(c, times=100, n_jobs=-1, dataname='ml_1m',
                         params_path=None):
    logpath = make_filename(extension='log',
                            exclude_args=['c', 'times',
                                          'params_path', 'n_jobs'])
    fpath = logpath.replace('log', 'pkl')
    with logutils.LogFile(logpath):
        if params_path is None:
            params_path = _build_path / f'params_lda_pop-{dataname}.pkl'
        with open(params_path, 'rb') as f:
            _log.info(f'Read params from {params_path}')
            best_params = pickle.load(f)
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto)
        algorithms = [Random(), Popular(), Oracle(None)]
        args = [pref, obs, algorithms, RLA]
        _log.info(f"simulation starts")
        run_simulation_csr(args, times=times, n_jobs=n_jobs, fdir=fpath)
        _log.info(f"simulation ends")


@task
def calibration_table_unif_unif_csr(c, dataname='ml_100k', metric='ucorr',
                                    n_jobs=-1, ntimes=20, res_file=None):
    logpath = make_filename(extension='log',
                            exclude_args=('c', 'n_jobs', 'res_file', 'ntimes'))
    fpath = logpath.replace('log', 'txt')
    build_path = Path('build')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        # load the best params
        if res_file:
            res_path = build_path / res_file
            res_gp = skopt.load(res_path)
            best_params = [nusers, nitems, *res_gp.x]
        else:
            params_path = (logpath.replace('log', 'pkl')
                           .replace('calibration_table', 'params'))
            if metric == 'avg_loss':
                params_path = build_path / f'params_unif_unif-{dataname}.pkl'
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        _log.debug(f'calibrate with params {best_params}')
        _log.info(f"calibration starts")
        stats_mean, stats_std = calibrator.score('all', ntimes=ntimes,
                                                 return_sum=False,
                                                 return_std=True)
        _log.info(f"calibration ends")
        with open(fpath, 'w') as f:
            f.write('Data,Pref,Obs,Item Pop,User Act,I-I Sim,U-U Sim\n')
            records = (f'{dataname},Unif,Unif,{stats_mean[0]}/{stats_std[0]},'
                       f'{stats_mean[1]}/{stats_std[1]},'
                       f'{stats_mean[2]}/{stats_std[2]},'
                       f'{stats_mean[3]}/{stats_std[3]}')
            f.write(records)


@task
def calibration_table_unif_pop_csr(c, dataname='ml_100k', metric='ucorr',
                                   n_jobs=-1, ntimes=20, res_file=None):
    logpath = make_filename(extension='log',
                            exclude_args=('c', 'n_jobs', 'res_file', 'ntimes'))
    fpath = logpath.replace('log', 'txt')
    build_path = Path('build')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        # load the best params
        if res_file:
            res_path = build_path / res_file
            res_gp = skopt.load(res_path)
            best_params = [nusers, nitems, *res_gp.x]
        else:
            params_path = (logpath.replace('log', 'pkl')
                           .replace('calibration_table', 'params'))
            if metric == 'avg_loss':
                params_path = build_path / f'params_unif_pop-{dataname}.pkl'
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        _log.debug(f'calibrate with params {best_params}')
        _log.info(f"calibration starts")
        stats_mean, stats_std = calibrator.score('all', ntimes=ntimes,
                                                 return_sum=False,
                                                 return_std=True)
        _log.info(f"calibration ends")
        with open(fpath, 'w') as f:
            f.write('Data,Pref,Obs,Item Pop,User Act,I-I Sim,U-U Sim\n')
            records = (f'{dataname},Unif,Pop,{stats_mean[0]}/{stats_std[0]},'
                       f'{stats_mean[1]}/{stats_std[1]},'
                       f'{stats_mean[2]}/{stats_std[2]},'
                       f'{stats_mean[3]}/{stats_std[3]}')
            f.write(records)


@task
def calibration_table_lda_unif_csr(c, dataname='ml_100k', metric='ucorr',
                                   n_jobs=-1, ntimes=20, res_file=None):
    logpath = make_filename(extension='log',
                            exclude_args=('c', 'n_jobs', 'res_file', 'ntimes'))
    fpath = logpath.replace('log', 'txt')
    build_path = Path('build')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        # load the best params
        if res_file:
            res_path = build_path / res_file
            res_gp = skopt.load(res_path)
            best_params = [nusers, nitems, *res_gp.x]
        else:
            params_path = (logpath.replace('log', 'pkl')
                           .replace('calibration_table', 'params'))
            if metric == 'avg_loss':
                params_path = build_path / f'params_lda_unif-{dataname}.pkl'
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
        # generate data with the best parameters
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        _log.debug(f'calibrate with params {best_params}')
        _log.info(f"calibration starts")
        stats_mean, stats_std = calibrator.score('all', ntimes=ntimes,
                                                 return_sum=False,
                                                 return_std=True)
        _log.info(f"calibration ends")
        with open(fpath, 'w') as f:
            f.write('Data,Pref,Obs,Item Pop,User Act,I-I Sim,U-U Sim\n')
            records = (f'{dataname},LDA,Unif,{stats_mean[0]}/{stats_std[0]},'
                       f'{stats_mean[1]}/{stats_std[1]},'
                       f'{stats_mean[2]}/{stats_std[2]},'
                       f'{stats_mean[3]}/{stats_std[3]}')
            f.write(records)


@task
def calibration_table_lda_pop_csr(c, dataname='ml_100k', metric='ucorr',
                                  n_jobs=-1, ntimes=20, res_file=None):
    logpath = make_filename(extension='log',
                            exclude_args=('c', 'n_jobs', 'res_file', 'ntimes'))
    fpath = logpath.replace('log', 'txt')
    build_path = Path('build')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        # load the best params
        if res_file:
            res_path = build_path / res_file
            res_gp = skopt.load(res_path)
            best_params = [nusers, nitems, *res_gp.x]
        else:
            params_path = (logpath.replace('log', 'pkl')
                           .replace('calibration_table', 'params'))
            if metric == 'avg_loss':
                params_path = build_path / f'params_lda_pop-{dataname}.pkl'
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
        # generate data with the best parameters
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        _log.debug(f'calibrate with params {best_params}')
        _log.info(f"calibration starts")
        stats_mean, stats_std = calibrator.score('all', ntimes=ntimes,
                                                 return_sum=False,
                                                 return_std=True)
        _log.info(f"calibration ends")
        with open(fpath, 'w') as f:
            f.write('Data,Pref,Obs,Item Pop,User Act,I-I Sim,U-U Sim\n')
            records = (f'{dataname},LDA,Pop,{stats_mean[0]}/{stats_std[0]},'
                       f'{stats_mean[1]}/{stats_std[1]},'
                       f'{stats_mean[2]}/{stats_std[2]},'
                       f'{stats_mean[3]}/{stats_std[3]}')
            f.write(records)


@task
def calibration_table_ibp_unif_csr(c, dataname='ml_100k', metric='ucorr',
                                   n_jobs=-1, ntimes=20, res_file=None):
    logpath = make_filename(extension='log',
                            exclude_args=('c', 'n_jobs', 'res_file', 'ntimes'))
    fpath = logpath.replace('log', 'txt')
    build_path = Path('build')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        # load the best params
        if res_file:
            res_path = build_path / res_file
            res_gp = skopt.load(res_path)
            best_params = [nusers, *res_gp.x]
        else:
            params_path = (logpath.replace('log', 'pkl')
                           .replace('calibration_table', 'params'))
            if metric == 'avg_loss':
                params_path = build_path / f'params_ibp_unif-{dataname}.pkl'
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
        # generate data with the best parameters
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma)
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        _log.debug(f'calibrate with params {best_params}')
        _log.info(f"calibration starts")
        stats_mean, stats_std = calibrator.score('all', ntimes=ntimes,
                                                 return_sum=False,
                                                 return_std=True)
        _log.info(f"calibration ends")
        with open(fpath, 'w') as f:
            f.write('Data,Pref,Obs,Item Pop,User Act,I-I Sim,U-U Sim\n')
            records = (f'{dataname},IBP,Unif,{stats_mean[0]}/{stats_std[0]},'
                       f'{stats_mean[1]}/{stats_std[1]},'
                       f'{stats_mean[2]}/{stats_std[2]},'
                       f'{stats_mean[3]}/{stats_std[3]}')
            f.write(records)


@task
def calibration_table_ibp_pop_csr(c, dataname='ml_100k', metric='ucorr',
                                  n_jobs=-1, ntimes=20, res_file=None):
    logpath = make_filename(extension='log',
                            exclude_args=('c', 'n_jobs', 'res_file', 'ntimes'))
    fpath = logpath.replace('log', 'txt')
    build_path = Path('build')
    with logutils.LogFile(logpath):
        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        # load the best params
        if res_file:
            res_path = build_path / res_file
            res_gp = skopt.load(res_path)
            best_params = [nusers, *res_gp.x]
        else:
            params_path = (logpath.replace('log', 'pkl')
                           .replace('calibration_table', 'params'))
            if metric == 'avg_loss':
                params_path = build_path / f'params_ibp_pop-{dataname}.pkl'
            with open(params_path, 'rb') as f:
                best_params = pickle.load(f)
        # generate data with the best parameters
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma)
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        _log.debug(f'calibrate with params {best_params}')
        _log.info(f"calibration starts")
        stats_mean, stats_std = calibrator.score('all', ntimes=ntimes,
                                                 return_sum=False,
                                                 return_std=True)
        _log.info(f"calibration ends")
        with open(fpath, 'w') as f:
            f.write('Data,Pref,Obs,Item Pop,User Act,I-I Sim,U-U Sim\n')
            records = (f'{dataname},IBP,Pop,{stats_mean[0]}/{stats_std[0]},'
                       f'{stats_mean[1]}/{stats_std[1]},'
                       f'{stats_mean[2]}/{stats_std[2]},'
                       f'{stats_mean[3]}/{stats_std[3]}')
            f.write(records)


@task
def skopt_unif_unif(c, dataname='ml_1m', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('skopt', 'params')
    metrics = ('item-pop', 'user-act', 'icorr', 'ucorr')
    models = ('unif_unif_csr', 'unif_pop_csr',
              'ibp_unif_csr', 'ibp_pop_csr',
              'lda_unif_csr', 'lda_pop_csr')
    with logutils.LogFile(logpath):
        # retrieve the best score across models for a single data set and
        # a single metric optimized for
        _log.info('Loading best scores by any models')
        optimal_scores = []
        for metric in metrics:
            scores = []
            for model in models:
                res_path = f'gp_minimize_{model}-{dataname}-{metric}.pkl'
                res_path = _build_path / res_path
                score = skopt.load(res_path).fun
                scores.append(score)
            min_kl = min(scores)
            optimal_scores.append(min_kl)

        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = UniformPreferenceCSR(nusers, nitems, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score('all', ntimes=n_jobs, return_sum=False)
            score /= optimal_scores  # relative loss for optimized stats
            return np.mean(score)

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=25)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam).generate()
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('skopt_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def skopt_unif_pop(c, dataname='ml_1m', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('skopt', 'params')
    metrics = ('item-pop', 'user-act', 'icorr', 'ucorr')
    models = ('unif_unif_csr', 'unif_pop_csr',
              'ibp_unif_csr', 'ibp_pop_csr',
              'lda_unif_csr', 'lda_pop_csr')
    with logutils.LogFile(logpath):
        # retrieve the best score across models for a single data set and
        # a single metric optimized for
        _log.info('Loading best scores by any models')
        optimal_scores = []
        for metric in metrics:
            scores = []
            for model in models:
                res_path = f'gp_minimize_{model}-{dataname}-{metric}.pkl'
                res_path = _build_path / res_path
                score = skopt.load(res_path).fun
                scores.append(score)
            min_kl = min(scores)
            optimal_scores.append(min_kl)

        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = UniformPreferenceCSR(nusers, nitems, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score('all', ntimes=n_jobs, return_sum=False)
            score /= optimal_scores  # relative loss for optimized stats
            return np.mean(score)

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=25)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, lam, m, alpha, upper = best_params
        pref = UniformPreferenceCSR(nusers, nitems, lam).generate()
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('skopt_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def skopt_ibp_unif(c, dataname='ml_1m', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('skopt', 'params')
    metrics = ('item-pop', 'user-act', 'icorr', 'ucorr')
    models = ('unif_unif_csr', 'unif_pop_csr',
              'ibp_unif_csr', 'ibp_pop_csr',
              'lda_unif_csr', 'lda_pop_csr')
    with logutils.LogFile(logpath):
        # retrieve the best score across models for a single data set and
        # a single metric optimized for
        _log.info('Loading best scores by any models')
        optimal_scores = []
        for metric in metrics:
            scores = []
            for model in models:
                res_path = f'gp_minimize_{model}-{dataname}-{metric}.pkl'
                res_path = _build_path / res_path
                score = skopt.load(res_path).fun
                scores.append(score)
            min_kl = min(scores)
            optimal_scores.append(min_kl)

        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = IndianBuffetProcessCSR(nusers, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(10, 1000, prior='log-uniform', name='pref__alpha'),
                 Real(1e-2, 100, prior='log-uniform', name='pref__c'),
                 Real(0, 0.99, name='pref__sigma'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            try:
                calibrator.set_params(**params)
            except ValueError:
                return 1e6  # invalid params return a large score
            else:
                _log.debug(f'calibrate with params {params}')
                score = calibrator.score('all', ntimes=n_jobs, return_sum=False)
                score /= optimal_scores  # relative loss for optimized stats
                return np.mean(score)

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=25)
        _log.info(f"calibration ends")
        skopt.dump(res_gp, fpath, store_objective=False)
        # save the best params
        best_params = [nusers, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma).generate()
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = UniformObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('skopt_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def skopt_ibp_pop(c, dataname='ml_1m', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('skopt', 'params')
    metrics = ('item-pop', 'user-act', 'icorr', 'ucorr')
    models = ('unif_unif_csr', 'unif_pop_csr',
              'ibp_unif_csr', 'ibp_pop_csr',
              'lda_unif_csr', 'lda_pop_csr')
    with logutils.LogFile(logpath):
        # retrieve the best score across models for a single data set and
        # a single metric optimized for
        _log.info('Loading best scores by any models')
        optimal_scores = []
        for metric in metrics:
            scores = []
            for model in models:
                res_path = f'gp_minimize_{model}-{dataname}-{metric}.pkl'
                res_path = _build_path / res_path
                score = skopt.load(res_path).fun
                scores.append(score)
            min_kl = min(scores)
            optimal_scores.append(min_kl)

        ds = DataStats(getattr(DATA_SETS, dataname))
        nusers = ds.nusers
        nitems = ds.nitems
        pref = IndianBuffetProcessCSR(nusers, 100)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 700)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive!
        space = [Real(10, 1000, prior='log-uniform', name='pref__alpha'),
                 Real(1e-2, 100, prior='log-uniform', name='pref__c'),
                 Real(0, 0.99, name='pref__sigma'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            try:
                calibrator.set_params(**params)
            except ValueError:
                return 1e6  # invalid params return a large score
            else:
                _log.debug(f'calibrate with params {params}')
                score = calibrator.score('all', ntimes=n_jobs, return_sum=False)
                score /= optimal_scores  # relative loss for optimized stats
                return np.mean(score)

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=25)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, alpha, c, sigma, m, obs_alpha, upper = best_params
        pref = IndianBuffetProcessCSR(nusers, alpha, c, sigma).generate()
        trunc_pareto = TruncParetoProfile(m, obs_alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('skopt_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def skopt_lda_unif(c, dataname='ml_1m', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('skopt', 'params')
    metrics = ('item-pop', 'user-act', 'icorr', 'ucorr')
    models = ('unif_unif_csr', 'unif_pop_csr',
              'ibp_unif_csr', 'ibp_pop_csr',
              'lda_unif_csr', 'lda_pop_csr')
    with logutils.LogFile(logpath):
        # retrieve the best score across models for a single data set and
        # a single metric optimized for
        _log.info('Loading best scores by any models')
        optimal_scores = []
        for metric in metrics:
            scores = []
            for model in models:
                res_path = f'gp_minimize_{model}-{dataname}-{metric}.pkl'
                res_path = _build_path / res_path
                score = skopt.load(res_path).fun
                scores.append(score)
            min_kl = min(scores)
            optimal_scores.append(min_kl)
        _log.info(f'Load data set {getattr(DATA_SETS, dataname).__name__} and '
                  f'compute statistics')

        ds = DataStats(getattr(DATA_SETS, dataname))
        _log.info('initialize models')
        nusers = ds.nusers
        nitems = ds.nitems
        pref = LatentDirichletAllocationCSR(nusers, nitems, 100, 85.0, 1, 1)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 737)
        obs = UniformObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive in skopt numbers!
        space = [Integer(5, 200, name='pref__k'),
                 Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(1e-2, 1, name='pref__a'),
                 Real(1e-2, 1, name='pref__b'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score('all', ntimes=n_jobs, return_sum=False)
            score /= optimal_scores  # relative loss for optimized stats
            return np.mean(score)

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=25)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = (LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
                .generate())
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = UniformObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('skopt_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


@task
def skopt_lda_pop(c, dataname='ml_1m', n_jobs=1):
    logpath = make_filename(extension='log', exclude_args=('c', 'n_jobs'))
    fpath = logpath.replace('log', 'pkl')
    param_path = fpath.replace('skopt', 'params')
    metrics = ('item-pop', 'user-act', 'icorr', 'ucorr')
    models = ('unif_unif_csr', 'unif_pop_csr',
              'ibp_unif_csr', 'ibp_pop_csr',
              'lda_unif_csr', 'lda_pop_csr')
    with logutils.LogFile(logpath):
        # retrieve the best score across models for a single data set and
        # a single metric optimized for
        _log.info('Loading best scores by any models')
        optimal_scores = []
        for metric in metrics:
            scores = []
            for model in models:
                res_path = f'gp_minimize_{model}-{dataname}-{metric}.pkl'
                res_path = _build_path / res_path
                score = skopt.load(res_path).fun
                scores.append(score)
            min_kl = min(scores)
            optimal_scores.append(min_kl)
        _log.info(f'Load data set {getattr(DATA_SETS, dataname).__name__} and '
                  f'compute statistics')

        ds = DataStats(getattr(DATA_SETS, dataname))
        _log.info('initialize models')
        nusers = ds.nusers
        nitems = ds.nitems
        pref = LatentDirichletAllocationCSR(nusers, nitems, 100, 85.0, 1, 1)
        trunc_pareto = TruncParetoProfile(20.00000045, 0.510528, 737)
        obs = PopularityObservationCSR(trunc_pareto)
        calibrator = CalibratorCSR(pref, obs, ds, n_jobs)
        maxua = ds.user_activity.index.max()
        minua = ds.user_activity.index.min()
        # m is tuned for the lower bound of the user activity
        m_low = max(minua * 0.8, 1.0)
        m_high = max(minua * 1.2, m_low * 1.2)
        # upper is tuned for the upper bound of the user activity
        # at least higher that m_high
        # upper_low = max(maxua * 0.9, m_high * 1.1)
        upper_low = max(maxua * 0.8, m_high * 1.2)
        upper_high = max(maxua * 1.2, upper_low * 1.2)
        # search space dimensions are inclusive in skopt numbers!
        space = [Integer(5, 200, name='pref__k'),
                 Real(5, 2000, prior='log-uniform', name='pref__lam'),
                 Real(1e-2, 1, name='pref__a'),
                 Real(1e-2, 1, name='pref__b'),
                 Real(m_low, m_high, name='obs__dist_func__m'),
                 Real(1e-1, 20, name='obs__dist_func__alpha'),
                 Real(upper_low, upper_high, prior='log-uniform',
                      name='obs__dist_func__upper')]

        @use_named_args(space)
        def objective(**params):
            calibrator.set_params(**params)
            _log.debug(f'calibrate with params {params}')
            score = calibrator.score('all', ntimes=n_jobs, return_sum=False)
            score /= optimal_scores  # relative loss for optimized stats
            return np.mean(score)

        _log.info(f"calibration starts")
        res_gp = gp_minimize(objective, space, n_calls=150, n_random_starts=25)
        _log.info(f"calibration ends")
        # save the best params
        best_params = [nusers, nitems, *res_gp.x]
        with open(param_path, 'wb') as f:
            pickle.dump(best_params, f)
        skopt.dump(res_gp, fpath, store_objective=False)
        # generate data with the best parameters
        nusers, nitems, k, lam, a, b, m, alpha, upper = best_params
        pref = (LatentDirichletAllocationCSR(nusers, nitems, k, lam, a, b)
                .generate())
        trunc_pareto = TruncParetoProfile(m, alpha, upper)
        obs = PopularityObservationCSR(trunc_pareto).sample(pref)
        coo = obs.tocoo(copy=False)
        df = pd.DataFrame({'user': coo.row, 'item': coo.col,
                           'rating': coo.data})
        dpath = Path(fpath).name.replace('skopt_', '')
        dpath = Path('data/simulated') / dpath
        with open(dpath, 'wb') as f:
            pickle.dump(df, f)
        return df


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()
