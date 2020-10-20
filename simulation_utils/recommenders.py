import pandas as pd
from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import Memorized


class Oracle(Memorized):
    """
    The oracle algorithm memorizes users' true preferences.

    The Oracle algorithm should be initialized by a pandas.DataFrame with
    `user`, `item`, and `rating` columns. If the user has a binary
    preference, set the rating to 1 or 0.
    """

    def fit(self, ratings, *args, **kwargs):
        self.scores = ratings.copy(True)
        if 'rating' not in self.scores.columns:
            self.scores['rating'] = 1
        return self

    def __str__(self):
        return 'Oracle'


class Random(Recommender):
    """
    The Random algorithm that recommends random items from all the items or
    candidate items(if provided).
    """

    def __init__(self, random_state=None):
        """
        Args:
             random_state: int or Series, optional
                Seed/Seeds for the random sampling of items.
                If int, then recommending random items for each user with
                the same seed. If Series, then it contains the seeds for the
                users, indexed by user id.
        """
        self.random_state = random_state
        self.items = None

    def fit(self, ratings, *args, **kwargs):
        items = pd.DataFrame(ratings['item'].unique(), columns=['item'])
        self.items = items
        return self

    def recommend(self, user, n=None, candidates=None, ratings=None):
        model = self.items
        seed = None
        if isinstance(self.random_state, int):
            seed = self.random_state
        if isinstance(self.random_state, pd.Series):
            seed = self.random_state.get(user)

        frac = None
        if n is None:
            frac = 1

        if candidates is not None:
            return (pd.DataFrame(candidates, columns=['item'])
                    .sample(n, frac, random_state=seed)
                    .reset_index(drop=True))
        else:
            return (model.sample(n, frac, random_state=seed)
                    .reset_index(drop=True))

    def __str__(self):
        return 'Random'
