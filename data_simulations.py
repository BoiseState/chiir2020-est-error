from os import path
import numpy as np
from multiprocessing import Pool
from simulation_utils.preference import IndianBuffetProcess


def main():
    nusers = 77805
    avg_nitems = 166690
    H_nusers = np.log(nusers) + 0.5772  # Harmonic numbers
    alpha = avg_nitems / H_nusers
    c = 1
    sigma = [0, 0.2, 0.5, 0.8]
    args_iter = zip([nusers]*4, [alpha]*4, [c]*4, sigma)
    with Pool(4) as p:
        p.starmap(ibp_generation, args_iter)


def ibp_generation(nusers, alpha, c, sigma):
    ibp = IndianBuffetProcess(nusers, alpha, c, sigma)
    preference = ibp.generate()
    filename = f'nusers{nusers}-alpha{round(alpha, 2)}-c{c}-sigma{sigma}'
    preference.to_csv(path.join('data', 'simulated', filename), index=False)


if __name__ == '__main__':
    main()
