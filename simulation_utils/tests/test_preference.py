import logging
import numpy as np
import pandas as pd
from scipy.special import gammaln, gamma
import pytest
from pytest import approx, raises

from simulation_utils.preference import generate_uniform, generate_ibp, \
    generate_lda, generate_lda_nbp, _draw_multinomial_nbp, _non_zero_nbp, \
    multinomial, records_to_coo, generate_uniform_csr
from simulation_utils.preference import generate_ibp_nb, generate_ibp_nbp
from simulation_utils.preference import UniformPreference, \
    IndianBuffetProcess, LatentDirichletAllocation
from simulation_utils.preference import _draw_bernoulli, _draw_bernoulli_nbp
from simulation_utils.preference import _draw_nnew_ibp, _draw_nnew_ibp_nbp
from simulation_utils.preference import generate_ibp_itemwise


_log = logging.getLogger(__name__)


def random_ibp_params(n):
    out = []
    for i in range(n):
        nusers = np.random.randint(100, 1000)
        nitems = np.random.randint(100, 1000)
        # H_nusers = np.log(nusers) + 0.5772  # approximate Harmonic numbers
        # alpha = nitems / H_nusers
        frac = np.random.random()
        alpha = nitems * frac
        c = 1 + np.random.random()
        sigma = np.random.random()
        out.append((nusers, alpha, c, sigma))
    return out


def random_lda_params(n):
    out = []
    for i in range(n):
        nusers = np.random.randint(100, 1000)
        nitems = np.random.randint(100, 1000)
        k = np.random.randint(100, 1000)
        frac = np.random.random()
        lam = nitems * frac
        a = np.random.uniform(0.1, 1)
        b = np.random.uniform(0.1, 1)
        out.append((nusers, nitems, k, lam, a, b))
    return out


valid_params = random_ibp_params(5)


def test_uniform_generator():
    nusers = 1000
    nitems = 100
    frac = 0.5
    data = generate_uniform(nusers, nitems, frac)
    npairs = len(data)
    density = npairs / nusers / nitems
    assert density == approx(frac, abs=1e-2)


def test_ibp_updates():
    alpha = 0.5
    sigma = np.random.random()
    c = np.random.uniform(-sigma, 2)
    for u in range(1, 150):
        lam1 = alpha * gamma(1 + c) / gamma(c + sigma) * np.exp(
            gammaln(u - 1 + c + sigma) - gammaln(u + c))
        lam2 = (alpha * gamma(1 + c) * gamma(u - 1 + c + sigma) /
                (gamma(u + c) * gamma(c + sigma)))
        lam3 = alpha * np.exp(gammaln(1 + c) + gammaln(u - 1 + c + sigma)
                              - gammaln(u + c) - gammaln(c + sigma))
        assert lam1 == approx(lam2, abs=1e-6)
        assert lam2 == approx(lam3, abs=1e-6)
        assert not np.isnan(lam3)


def test_uniform_preference():
    nusers = 200
    nitems = 100
    frac = 0.5
    up = UniformPreference(nusers, nitems, frac)
    pref = up.generate()
    assert all(pref.columns == np.array(['user', 'item']))
    actual_nusers = len(pref['user'].unique())
    actual_nitems = len(pref['item'].unique())
    assert nusers == actual_nusers
    assert nitems == actual_nitems
    npairs = len(pref)
    density = npairs / actual_nitems / actual_nusers
    assert density == approx(frac, abs=1e-2)


def test_indian_buffet_process():
    nusers = 200
    avg_nitems = 100
    H_nusers = np.log(nusers) + 0.5772  # Harmonic numbers
    alpha = avg_nitems / H_nusers
    avg_density = 1 / H_nusers

    # test the average number of nitems and density
    nitems = []
    densities = []
    for i in range(5):
        ibp = IndianBuffetProcess(nusers, alpha)
        pref = ibp.generate()
        assert all(pref.columns == np.array(['user', 'item']))
        actual_nusers = len(pref['user'].unique())
        actual_nitems = len(pref['item'].unique())
        npairs = len(pref)
        assert nusers == actual_nusers
        density = npairs / actual_nusers / actual_nitems
        nitems.append(actual_nitems)
        densities.append(density)
    actual_avg_nitems = np.mean(nitems)
    actual_avg_density = np.mean(densities)
    assert actual_avg_nitems == approx(avg_nitems, abs=10)
    assert actual_avg_density == approx(avg_density, abs=2e-2)


def test_ibp_check_values():
    nusers = 200
    alpha = 20
    ibp = IndianBuffetProcess(nusers, alpha)
    # test alpha
    with raises(ValueError):
        IndianBuffetProcess(nusers, -5)
    with raises(ValueError):
        IndianBuffetProcess(nusers, None)
    with raises(ValueError):
        ibp.set_params(alpha=-2.5)
    with raises(ValueError):
        ibp.set_params(alpha=None)

    with raises(ValueError):
        ibp.set_params(alpha=10, c=-2, sigma=1)

    with raises(ValueError):
        ibp.set_params(alpha=10, c=2, sigma=1)


@pytest.mark.parametrize('nusers, alpha, c, sigma', valid_params)
def test_serial_generate_ibp_consistency(nusers, alpha, c, sigma):
    # test numba and numpy serial implementations have the same results for
    # n times of random parameters
    # set seed for ibp generation
    seed = np.random.randint(100)
    np.random.seed(seed)
    # numpy implementation
    pref_np = generate_ibp(nusers, alpha, c, sigma).values
    # numba implementation
    items = generate_ibp_nb(nusers, alpha, c, sigma, debug=True, seed=seed)
    user_item = map(lambda x: pd.DataFrame({'user': x[0], 'item': x[1]}),
                    zip(range(nusers), items))
    pref_nb = pd.concat(user_item, axis=0, ignore_index=True).values
    assert np.array_equal(pref_np, pref_nb)


@pytest.mark.parametrize('nusers, alpha, c, sigma', valid_params)
def test_draw_nnew_ibp_stats(nusers, alpha, c, sigma):
    nsamples = 2000  # the number of run
    samples = np.empty((nsamples, nusers), dtype=np.int64)
    samples_nbp = np.empty((nsamples, nusers), dtype=np.int64)

    log_gamma_a = gammaln(1 + c)
    log_gamma_b = gammaln(c + sigma)
    users = np.arange(nusers)
    expected_lambdas = alpha * np.exp(log_gamma_a + gammaln(users + c + sigma) -
                                      gammaln(users + 1 + c) - log_gamma_b)
    # test mean
    for i in range(nsamples):
        nnew, _, _ = _draw_nnew_ibp(nusers, alpha, c, sigma)
        samples[i] = nnew
        nnew, _, _ = _draw_nnew_ibp_nbp(nusers, alpha, c, sigma)
        samples_nbp[i] = nnew
    actual_lambdas = samples.mean(axis=0)
    actual_lambdas_nbp = samples_nbp.mean(axis=0)
    assert expected_lambdas == approx(actual_lambdas, rel=1e-1)
    assert expected_lambdas == approx(actual_lambdas_nbp, rel=1e-1)

    # test cumsum
    nnew, npicked_start, npicked_end = _draw_nnew_ibp(nusers, alpha, c, sigma)
    assert np.all(np.cumsum(nnew) == npicked_end)
    expected_start = np.cumsum(nnew) - nnew
    assert np.all(expected_start == npicked_start)
    nnew, npicked_start, npicked_end = _draw_nnew_ibp_nbp(nusers, alpha, c, sigma)
    assert np.all(np.cumsum(nnew) == npicked_end)
    expected_start = np.cumsum(nnew) - nnew
    assert np.all(expected_start == npicked_start)


@pytest.mark.parametrize('prob', np.round(np.random.rand(5, 3), 2))
def test_draw_bernoulli(prob):
    nsamples = 20000
    nprob = len(prob)
    samples = np.empty((nsamples, nprob), dtype=np.int64)
    samples_nbp = np.empty((nsamples, nprob), dtype=np.int64)
    for i in range(nsamples):
        pick = _draw_bernoulli(prob)
        samples[i] = pick
        pick = _draw_bernoulli_nbp(prob)
        samples_nbp[i] = pick
    actual = samples.mean(axis=0)
    actual_nbp = samples_nbp.mean(axis=0)
    assert np.allclose(prob, actual, rtol=1e-2, atol=1e-2)
    assert np.allclose(prob, actual_nbp, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('func', [generate_ibp_nb, generate_ibp_nbp])
def test_generate_ibp_stats(func):
    nusers = 100
    avg_nitems = 200
    H_nusers = np.log(nusers) + 0.5772  # approximate Harmonic numbers
    # H_nusers = np.reciprocal(hmean(np.arange(1, nusers + 1))) * nusers
    alpha = avg_nitems / H_nusers
    avg_density = 1 / H_nusers
    avg_npairs = nusers * alpha

    # test the average number of nitems and density
    nitems = []
    densities = []
    npairs = []

    for i in range(1000):
        pref = func(nusers, alpha)
        actual_nusers = len(pref)
        assert nusers == actual_nusers
        items = np.hstack(pref)
        actual_npairs = len(items)
        actual_nitems = len(np.unique(items))
        density = actual_npairs / actual_nusers / actual_nitems
        nitems.append(actual_nitems)
        densities.append(density)
        npairs.append(actual_npairs)
    actual_avg_nitems = np.mean(nitems)
    actual_avg_density = np.mean(densities)
    actual_avg_npairs = np.mean(npairs)
    assert actual_avg_nitems == approx(avg_nitems, rel=1e-2)
    assert actual_avg_density == approx(avg_density, abs=2e-2)
    assert actual_avg_npairs == approx(avg_npairs, rel=1e-2)


@pytest.mark.parametrize('func', [generate_ibp_itemwise])
def test_generate_ibp_itemwise_stats(func):
    nusers = 2000
    avg_nitems = 200
    H_nusers = np.log(nusers) + 0.5772  # approximate Harmonic numbers
    # H_nusers = np.reciprocal(hmean(np.arange(1, nusers + 1))) * nusers
    alpha = avg_nitems / H_nusers
    avg_density = 1 / H_nusers
    avg_npairs = nusers * alpha

    # test the average number of nitems and density
    nitems = []
    densities = []
    npairs = []

    for i in range(1000):
        pref = func(nusers, alpha)
        actual_nitems = len(pref)
        users = np.hstack(pref)
        actual_nusers = len(np.unique(users))
        assert nusers == actual_nusers
        actual_npairs = len(users)
        density = actual_npairs / actual_nusers / actual_nitems
        nitems.append(actual_nitems)
        densities.append(density)
        npairs.append(actual_npairs)
    actual_avg_nitems = np.mean(nitems)
    actual_avg_density = np.mean(densities)
    actual_avg_npairs = np.mean(npairs)

    assert actual_avg_nitems == approx(avg_nitems, rel=1e-2)
    assert actual_avg_density == approx(avg_density, abs=2e-2)
    assert actual_avg_npairs == approx(avg_npairs, rel=1e-2)


@pytest.mark.parametrize('nusers', [100, 200])
def test_generate_lda(nusers):
    pref = generate_lda(nusers, 20, 10, 10, 1, 1)
    assert len(pref['user'].unique()) == nusers
    assert np.array_equal(pref.columns, ['user', 'item', 'rating'])


@pytest.mark.parametrize('nusers', [100, 200])
def test_generate_lda_nbp(nusers):
    pref = generate_lda_nbp(nusers, 20, 10, 10, 1, 1)
    assert len(pref['user'].unique()) == nusers
    assert np.array_equal(pref.columns, ['user', 'item', 'rating'])


@pytest.mark.parametrize('nusers, nitems, k, lam, a, b', random_lda_params(5))
def test_generate_lda_consistency(nusers, nitems, k, lam, a, b):
    pref_np = generate_lda(nusers, nitems, k, lam, a, b)
    pref_nbp = generate_lda_nbp(nusers, nitems, k, lam, a, b)
    # test mean and std for each columns
    d1 = pref_nbp.describe().values[1:3]
    d2 = pref_np.describe().values[1:3]
    assert d1 == approx(d2, rel=1e-1)


@pytest.mark.parametrize('n, k, a, times', [[100, 10, 0.1, 100000]])
def test_draw_multinomial_nbp(n, k, a, times):
    alpha = np.full(k, a)
    pvals = np.random.dirichlet(alpha, n)
    nt = np.random.randint(1, 100, size=n)
    samples = np.dstack(_draw_multinomial_nbp(nt, pvals) for _ in range(
        times))
    actual_mean = samples.mean(axis=2)
    expected_mean = nt.reshape((-1, 1)) * pvals
    assert actual_mean == approx(expected_mean, rel=1e-1, abs=1e-1)


@pytest.mark.parametrize('n, pvals, times',
                         [[100, np.random.dirichlet(np.full(10, 0.1)), 100000],
                          [100, np.array([0.2, 0.8, 0, 0, 0]), 100000]])
def test_multinomial(n, pvals, times):
    expected_mean = n * pvals
    samples = np.vstack(multinomial(n, pvals) for _ in range(times))
    np_samples = np.vstack(
        np.random.multinomial(n, pvals) for _ in range(times))
    actual_mean = samples.mean(axis=0)
    np_mean = np_samples.mean(axis=0)
    assert actual_mean == approx(expected_mean, rel=1e-1, abs=1e-2)
    assert actual_mean == approx(np_mean, rel=1e-1, abs=1e-2)


def test_non_zero_nbp():
    arr = np.array([[1, 2, 3, 0, 2], [2, 0, 3.3, 0, 0]])
    expected_x, expected_y = np.nonzero(arr)
    expected_values = arr[expected_x, expected_y]
    x, y, values = _non_zero_nbp(arr)
    assert np.array_equal(expected_x, x)
    assert np.array_equal(expected_y, y)
    assert np.array_equal(expected_values, values)


def test_records_to_coo():
    record = [np.array([0, 1, 2]), np.array([1, 3, 4]), np.array([1, 2, 3])]
    index = np.array([0, 0, 1])
    value = [np.array([3, 1, 2]), np.array([2, 3, 4]), np.array([5, 2, 3])]
    actual = records_to_coo(record, index, value)
    actual_values = actual.data
    actual_rows = actual.row
    actual_cols = actual.col
    expected_rows = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    expected_cols = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    expected_values = np.array([3, 1, 2, 3, 4, 5, 2, 3])
    actual = np.column_stack((actual_rows, actual_cols, actual_values))
    expected = np.column_stack((expected_rows, expected_cols, expected_values))
    assert np.array_equal(expected, actual)


def test_generate_uniform_csr():
    nusers = 1000
    nitems = 1000
    lam = 100
    ratings = generate_uniform_csr(nusers, nitems, lam)
    assert ratings.shape == (nusers, nitems)
    assert np.mean(ratings.getnnz(axis=1)) == approx(lam, abs=1)
