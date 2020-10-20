import logging
import numpy as np
import pytest
from pytest import approx
from scipy.sparse import csr_matrix

from simulation_utils.preference import IndianBuffetProcess
from simulation_utils.observation import UniformObservation, \
    PopularityObservation, TruncParetoProfile, random_choice, \
    UniformObservationCSR, PopularityObservationCSR
from simulation_utils.utils import TruncatedPareto

_log = logging.getLogger(__name__)


@pytest.mark.parametrize('obs_model', [UniformObservation, PopularityObservation])
def test_set_params(obs_model):
    truncated_pareto = TruncatedPareto(20, 0.51)
    observation = obs_model(truncated_pareto)
    expected_params = {'dist_func__m': 30,
                       'dist_func__alpha': 0.6,
                       'cap': False}
    observation.set_params(**expected_params)
    actual_params = observation.get_params()
    expected_params.update(dist_func=truncated_pareto, dist_func__upper=737)
    assert expected_params == actual_params


@pytest.mark.parametrize('obs_model', [UniformObservation,
                                       PopularityObservation])
def test_sample_params(obs_model):
    ibp = IndianBuffetProcess(20, 10)
    pref = ibp.generate()
    truncated_pareto = TruncatedPareto(20, 0.51)
    observation = obs_model(truncated_pareto)
    expected_params = {'dist_func__m': 20,
                       'dist_func__alpha': 0.51,
                       'cap': True}
    actual_params = observation.get_params()
    expected_params.update(dist_func=truncated_pareto, dist_func__upper=737)
    assert expected_params == actual_params
    # calling sample method sets new parameters
    obs = observation.sample(pref, m=5, alpha=0.6, upper=700, cap=False)
    actual_params = observation.get_params()
    expected_params.update(cap=False, dist_func=truncated_pareto,
                           dist_func__m=5, dist_func__alpha=0.6,
                           dist_func__upper=700)
    assert expected_params == actual_params


@pytest.mark.parametrize('size', [1, 2, 3, 4, 5])
def test_random_choice(size):
    p = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.25])
    p = p / p.sum()
    counts = np.zeros(len(p))
    for i in range(100000):
        samples = random_choice(p, size)
        counts[samples] += 1
    counts_np = np.zeros(len(p))
    for i in range(100000):
        samples_np = np.random.choice(len(p), size, replace=False, p=p)
        counts_np[samples_np] += 1
    actual = counts / counts.sum()
    expected = counts_np / counts_np.sum()
    assert actual == approx(expected, abs=2e-2)


def test_sample_csr_empty():
    # empty row #2 and column #2
    row = np.array([0, 0, 1, 1, 3, 3, 5])
    col = np.array([0, 1, 3, 4, 3, 5, 7])
    value = np.random.randint(1, 5, size=len(row))
    csr = csr_matrix((value, (row, col)))
    tpp = TruncParetoProfile(1, 0.5, 20)
    # uniform sampler
    unif_sampler = UniformObservationCSR(tpp)
    obs = unif_sampler.sample(csr)
    # test sampling values correctly
    assert np.array_equal(obs[obs.nonzero()], csr[obs.nonzero()])
    obs = obs.tocoo()
    obs_row = obs.row
    obs_col = obs.col
    # test sampling row and col correctly without empty cols and rows.
    assert np.all(np.isin(obs_row, row))
    assert np.all(np.isin(obs_col, col))
    # popular sampler
    pop_sampler = PopularityObservationCSR(tpp)
    obs = pop_sampler.sample(csr)
    assert np.array_equal(obs[obs.nonzero()], csr[obs.nonzero()])
    obs = obs.tocoo()
    obs_row = obs.row
    obs_col = obs.col
    assert np.all(np.isin(obs_row, row))
    assert np.all(np.isin(obs_col, col))
