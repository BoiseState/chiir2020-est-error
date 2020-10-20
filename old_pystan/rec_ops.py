"""
Recommender operations for the project.
"""

import logging

from lenskit import batch, topn

_log = logging.getLogger(__name__)

def train_save(algo, ratings, file):
    """
    Train an alogrithm and save its model to a file.
    """
    _log.info('training %s on %s', algo, len(ratings))
    model = algo.train(ratings)
    _log.info('saving model to %s', file)
    algo.save_model(model, file)

def gen_recs(algo, ratings, users, mod_file, out_file):
    """
    Generate recommendations from an algorithm.
    """
    _log.info('loading model from %s', mod_file)
    model = algo.load_model(mod_file)
    _log.info('producing recs for %d users', len(users))
    cand_fun = topn.UnratedCandidates(ratings)
    recs = batch.recommend(algo, model, users, 100, cand_fun)
    recs.to_csv(out_file, index=False)
