import logging, logging.config
import datetime
import numpy as np
import pandas as pd
import pickle
import yaml

from simulation_utils.calibration import calibrate_data
from simulation_utils.datasets import ml_100k
from simulation_utils.preference import generate_ibp_itemwise, records_to_frame
from simulation_utils.observation import sample_popular_n, sample_uniform_n
from simulation_utils.utils import truncated_beta_binomial, truncated_pareto


with open('logging.yaml') as lf:
    log_config = yaml.load(lf)
    
logging.config.dictConfig(log_config)
_log = logging.getLogger()


def generate_ibp_df(nusers, alpha, c=1, sigma=0):
    items = generate_ibp_itemwise(nusers, alpha, c, sigma)
    df = records_to_frame(items, 'item', 'user').reindex(columns=['user', 'item'])
    return df


def main():
    data = ml_100k()
    nusers = len(data['user'].unique())
    nitems = len(data['item'].unique())

    generator = generate_ibp_df
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

    with open(f'build/eval_results_{end_time.strftime("%Y-%m-%d-%H-%M-%S")}.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
