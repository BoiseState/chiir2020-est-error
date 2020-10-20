"""
Data sets.

"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from simulation_utils.calibration import compute_dstats, \
    compute_cosine_similarity_pd, _sample_pairs

data_path = Path(__file__).parent.parent / 'data'
if not data_path.exists():
    data_path.mkdir()

_log = logging.getLogger(__name__)


def reindex_columns(data, columns):
    """Reindices columns inplace"""
    for col in columns:
        index = pd.Index(data[col].unique())
        new_ids = index.get_indexer(data[col])
        data[col] = new_ids
    return data


def download_file(url, saveto, context=None):
    from urllib.request import urlopen
    _log.info(f'downloading file from {url}')
    response = urlopen(url, context)
    file = response.read()
    with open(saveto, 'wb') as f:
        f.write(file)
    _log.info(f'file downloaded to {saveto}')


def ml_100k():
    """
    MovieLens dataset.

    Unique users: 943, unique items: 1682, ratings: 100000
    User activity:
        25%   50%    75%  count    max        mean   min         std
        33.0  65.0  148.0  943.0  737.0  106.044539  20.0  100.931743
    Item Popularity:
        25%   50%    75%  count    max        mean   min         std
        6.0  27.0  80.0  1682.0  583.0  59.453032  1.0  80.383846
    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'ml-100k' / 'u.data'
    return (pd.read_csv(file_path, sep='\t',
                        names=['user', 'item', 'rating', 'timestamp'])
            .drop('timestamp', axis=1))


def ml_1m():
    """
    MovieLens one million dataset.

    Unique users: 6040, unique items: 3706, ratings: 1000209
    User activity:
        25%   50%    75%   count     max        mean   min         std
        44.0  96.0  208.0  6040.0  2314.0  165.597517  20.0  192.747029
    Item Popularity:
        25%   50%    75%   count     max        mean   min         std
        33.0  123.5  350.0  3706.0  3428.0  269.889099  1.0  384.047838
    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    parsed_fpath = data_path / 'ml-1m' / 'ratings.dat'
    downloaded_fpath = (data_path / 'ml-1m' /
                        'ml-1m.zip')
    if not parsed_fpath.exists() and not downloaded_fpath.exists():
        downloaded_fpath.parent.mkdir(parents=True, exist_ok=True)
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        download_file(url, downloaded_fpath)
    if not parsed_fpath.exists():
        _log.info(f'Unzip file downloaded at {downloaded_fpath}')
        from zipfile import ZipFile
        with ZipFile(downloaded_fpath, 'r') as f:
            f.extractall(parsed_fpath.parent.parent)
            _log.info(f'File extracted to {parsed_fpath.parent}')
    return (pd.read_csv(parsed_fpath, sep='::',
                        names=['user', 'item', 'rating', 'timestamp'],
                        engine='python')
            .drop('timestamp', axis=1)
            .rename(columns={'userId': 'user', 'movieId': 'item'}))


def ml_10m():
    """
    MovieLens ten million dataset.

    Unique users: 69878, unique items: 10677, ratings: 10000054
    User activity:
        25%   50%    75%    count     max       mean   min        std
        35.0  69.0  156.0  69878.0  7359.0  143.10733  20.0  216.71258
    Item Popularity:
        25%    50%    75%    count      max        mean  min          std
        34.0  135.0  626.0  10677.0  34864.0  936.597733  1.0  2487.328304
    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'ml-10m' / 'ratings.dat'
    return (pd.read_csv(file_path, sep='::',
                        names=['user', 'item', 'rating', 'timestamp'])
            .drop('timestamp', axis=1)
            .rename(columns={'userId': 'user', 'movieId': 'item'}))


def ml_20m():
    """
    MovieLens twenty million dataset.

    Unique users: 138493, unique items: 26744, ratings: 20000263
    User activity:
        25%   50%    75%     count     max       mean   min         std
        35.0  68.0  155.0  138493.0  9254.0  144.41353  20.0  230.267257
    Item Popularity:
        25%   50%    75%    count      max        mean  min          std
        3.0  18.0  205.0  26744.0  67310.0  747.841123  1.0  3085.818268
    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'ml-20m' / 'ratings.csv'
    return (pd.read_csv(file_path)
            .drop('timestamp', axis=1)
            .rename(columns={'userId': 'user', 'movieId': 'item'}))


def bx_implicit():
    """
    Book Crossing implicit dataset.

    Unique users: 77803, unique items: 166687, ratings: 431534
    User activity:
        25%  50%  75%    count     max      mean  min        std
        1.0  1.0  3.0  77803.0  8140.0  5.546496  1.0  42.838022
    Item Popularity:
        25%  50%  75%     count    max      mean  min       std
        1.0  1.0  2.0  166687.0  769.0  2.588888  1.0  8.765113
    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'bx' / 'bx-implicit.csv'
    df = pd.read_csv(file_path)
    return df[df['rating'] > 0].rename(columns={'userID': 'user',
                                                'bookID': 'item'})


def bx_explicit():
    """
    Book Crossing rating dataset.

    Unique users: 77805, unique items: 166690, ratings: 431627
    User activity:
        25%  50%  75%    count     max      mean  min        std
        1.0  1.0  3.0  77805.0  8167.0  5.547548  1.0  42.906237
    Item Popularity:
        25%  50%  75%     count     max      mean  min        std
        1.0  1.0  2.0  166690.0  769.0  2.589399  1.0  8.772745

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'bx' / 'bx-ratings.csv'
    df = pd.read_csv(file_path)
    return df[df['rating'] > 0].rename(columns={'userID': 'user',
                                                'bookID': 'item'})


def az_book():
    """
    Amazon book rating dataset.

    Unique users: 8026324, unique items: 2330066, ratings: 22507155
    User activity:
        25%  50%  75%      count      max      mean  min        std
        1.0  1.0  2.0  8026324.0  43201.0  2.804167  1.0  23.022114
    Item Popularity:
        25%  50%  75%      count      max     mean  min        std
        1.0  2.0  6.0  2330066.0  21398.0  9.65945  1.0  64.374683

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'az-small' / 'ratings_Books.csv'
    df = pd.read_csv(file_path, names=['user', 'item', 'rating', 'timestamp'])
    df.drop('timestamp', axis=1, inplace=True)
    df = df[df['rating'] > 0]
    user_id = pd.Index(df['user'].unique())
    item_id = pd.Index(df['item'].unique())
    users = user_id.get_indexer(df['user'])
    items = item_id.get_indexer(df['item'])
    df['user'] = users
    df['item'] = items
    return df.astype(np.int32)


def az_music():
    """
    Amazon music rating dataset.

    Unique users: 478235, unique items: 266414, ratings: 836006.
    User activity:
        25%  50%  75%     count     max      mean  min       std
        1.0  1.0  2.0  478235.0  1126.0  1.748107  1.0  4.227086
    Item Popularity:
        25%  50%  75%     count     max      mean  min        std
        1.0  1.0  2.0  266414.0  1953.0  3.137996  1.0  17.532391

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'az-music' / 'ratings_Digital_Music.csv'
    df = pd.read_csv(file_path, names=['user', 'item', 'rating', 'timestamp'])
    df.drop('timestamp', axis=1, inplace=True)
    df = df[df['rating'] > 0]
    user_id = pd.Index(df['user'].unique())
    item_id = pd.Index(df['item'].unique())
    users = user_id.get_indexer(df['user'])
    items = item_id.get_indexer(df['item'])
    df['user'] = users
    df['item'] = items
    return df.astype(np.int32)


def az_music_instruments():
    """
    Amazon music instruments dataset.

    Unique users: 339231, unique items: 83046, ratings: 500176.
    User activity:
        25%  50%  75%     count     max      mean  min       std
        1.0  1.0  1.0  339231.0  483.0  1.474441  1.0  2.342608
    Item Popularity:
        25%  50%  75%     count     max      mean  min        std
        1.0  2.0  4.0  83046.0  3523.0  6.022879  1.0  28.294073

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = (data_path / 'az-music-instruments' /
                 'ratings_Musical_Instruments.csv')
    df = pd.read_csv(file_path, names=['user', 'item', 'rating', 'timestamp'])
    df.drop('timestamp', axis=1, inplace=True)
    df = df[df['rating'] > 0]
    user_id = pd.Index(df['user'].unique())
    item_id = pd.Index(df['item'].unique())
    users = user_id.get_indexer(df['user'])
    items = item_id.get_indexer(df['item'])
    df['user'] = users
    df['item'] = items
    return df.astype(np.int32)


def az_music_5core():
    """
    Amazon music five-core dataset.

    Unique users: 5541, unique items: 3568, ratings: 64706.
    User activity:
        25%  50%   75%   count    max       mean  min      std
        5.0  7.0  11.0  5541.0  578.0  11.677676  5.0  18.2228
    Item Popularity:
        25%   50%   75%   count    max      mean  min        std
        6.0  10.0  20.0  3568.0  272.0  18.13509  5.0  21.639818

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    file_path = data_path / 'az-music-5core' / 'azm_5core.json.gz'
    if not file_path.exists():
        _log.info('file not found, downloading from '
                  'http://jmcauley.ucsd.edu/data/amazon/')
        file_path.parent.mkdir(parents=True, exist_ok=True)
        url = ('http://snap.stanford.edu/data/amazon/'
               'productGraph/categoryFiles/reviews_Digital_Music_5.json.gz')
        data = pd.read_json(url, lines=True)
        data = data[['reviewerID', 'asin', 'overall']].rename(
            columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating'},
            copy=True)
        data.drop_duplicates(['user', 'item'], inplace=True)
        data = data[data['rating'] > 0]
        data = reindex_columns(data, ['user', 'item'])
        data.to_json(file_path, orient='records', lines=True)
        _log.info(f'downloaded dataset is saved to {file_path}')
    else:
        data = pd.read_json(file_path, orient='records', lines=True)
    return data.astype(np.int32).reindex(columns=['user', 'item', 'rating'],
                                         copy=False)


def az_book_5core():
    """Takes a huge memory"""
    file_path = data_path / 'az-book-5core' / 'azb_5core.json.gz'
    if not file_path.exists():
        _log.info('file not found, downloading from '
                  'http://jmcauley.ucsd.edu/data/amazon/')
        file_path.parent.mkdir(parents=True, exist_ok=True)
        url = ('http://snap.stanford.edu/data/amazon/'
               'productGraph/categoryFiles/reviews_Books_5.json.gz')
        data = pd.read_json(url, lines=True)
        data = data[['reviewerID', 'asin', 'overall']].rename(
            columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating'},
            copy=True)
        data.drop_duplicates(['user', 'item'], inplace=True)
        data = data[data['rating'] > 0]
        data = reindex_columns(data, ['user', 'item'])
        data.to_json(file_path, orient='records', lines=True)
        _log.info(f'downloaded dataset is saved to {file_path}')
    else:
        data = pd.read_json(file_path, orient='records', lines=True)
    return data.astype(np.int32).reindex(columns=['user', 'item', 'rating'],
                                         copy=False)


def _parsing_steam_game_data(file, saveto):
    import ast
    import gzip
    # parse invalid json
    with gzip.open(file) as rf:
        user_records = pd.DataFrame.from_records(
            ast.literal_eval(line.decode('utf8')) for line in rf)
    # clean up
    user_records['user_id'] = user_records['user_id'].astype('category')
    user_records['user_steam_id'] = user_records['steam_id'].astype('i8')
    user_records = user_records[['user_id', 'user_steam_id', 'items']]
    user_records = user_records.drop_duplicates(subset=['user_id'])

    def unpack_user_items(row):
        """Unpack items from the items in a user's row"""
        # items is a series of lists - chain will make one long iterable
        # convert this to a data frame
        idf = pd.DataFrame.from_records(row.items)
        # now fix up data types
        idf['item_id'] = idf['item_id'].astype('i8')
        idf['user_id'] = row.user_id
        return idf[['user_id', 'item_id', 'item_name']]

    user_games = pd.concat(
        (unpack_user_items(row) for row in user_records.itertuples() if
         len(row.items) > 0),
        ignore_index=True)
    del user_games['item_name']
    user_games.to_csv(saveto, index=False)
    _log.info(f'parsed file is saved to {saveto}')


def steam_video_game():
    """
    Steam video game dataset.

    Unique users: 70912, unique items: 10978, ratings: 5094082.
    User activity:
        25%   50%   75%    count     max       mean  min      std
        14.0  40.0  87.0  70912.0  7762.0  71.836671  1.0  132.366763
    Item Popularity:
        25%   50%   75%    count     max       mean  min      std
        9.0  43.0  220.0  10978.0  49136.0  464.026416  1.0  1793.593696

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    parsed_fpath = data_path / 'steam-video-game' / 'steam-video-game.csv'
    downloaded_fpath = (data_path / 'steam-video-game' /
                        'australian_users_items.json.gz')
    if not parsed_fpath.exists() and not downloaded_fpath.exists():
        downloaded_fpath.parent.mkdir(parents=True, exist_ok=True)
        url = ('http://jmcauley.ucsd.edu/data/steam/'
               'australian_users_items.json.gz')
        download_file(url, downloaded_fpath)

    if not parsed_fpath.exists():
        _log.info('parsing downloaded file')
        _parsing_steam_game_data(downloaded_fpath, parsed_fpath)

    data = pd.read_csv(parsed_fpath).rename(columns={'user_id': 'user',
                                                     'item_id': 'item'})
    data = reindex_columns(data, ['user', 'item'])
    data['rating'] = 1
    return data.astype(np.int32)


def steam_video_game_5core():
    """
    Steam video game dataset.

    Unique users: 62936, unique items: 9192, ratings: 5073447.
    User activity:
        25%   50%   75%    count     max       mean  min      std
        14.0  40.0  87.0  70912.0  7762.0  71.836671  1.0  132.366763
    Item Popularity:
        25%   50%   75%    count     max       mean  min      std
        9.0  43.0  220.0  10978.0  49136.0  464.026416  1.0  1793.593696

    Returns:
        pandas.DataFrame: The rating dataframe.

    """
    parsed_fpath = data_path / 'steam-video-game' / 'steam-pruned.csv'
    data = pd.read_csv(parsed_fpath).rename(columns={'user_id': 'user',
                                                     'item_id': 'item'})
    data = reindex_columns(data, ['user', 'item'])
    data['rating'] = 1
    return data.astype(np.int32)


def _compute_cosine_similarity_dist(data, label, index):
    cs = compute_cosine_similarity_pd(data, _sample_pairs, label, index)
    hist, bins = np.histogram(cs, bins='auto', range=(-1, 1))
    # line up hist with bins
    return pd.Series(np.insert(hist, 0, 0), index=bins)


class DataStats(object):
    """
    The DataStats class with members of data key statistics.

    """
    def __init__(self, data_func, implicit=True):
        self.data_func = data_func
        if callable(data_func):
            data = data_func()
            self.is_func = True
        else:
            data = data_func
            self.is_func = False

        if implicit:
            data = data[data['rating'] > 0]
            data['rating'] = 1

        self.nusers = len(data['user'].unique())
        self.nitems = len(data['item'].unique())
        self.item_popularity = compute_dstats(data['item'].values)
        self.user_activity = compute_dstats(data['user'].values)
        # items with at least 5 users used to compute item item similarity
        item_popularity = (data[['user', 'item']].groupby(
            'item', as_index=False).count())
        items = item_popularity[item_popularity['user'] >= 5]['item'].values
        subset = data[data['item'].isin(items)]
        self.item_corr = _compute_cosine_similarity_dist(subset, 'item', 'user')
        del item_popularity, items
        # users with at least 5 items used to compute user user similarity
        user_activity = data[['user', 'item']].groupby(
            'user', as_index=False).count()
        users = user_activity[user_activity['item'] >= 5]['user'].values
        subset = data[data['user'].isin(users)]
        self.user_corr = _compute_cosine_similarity_dist(subset, 'user', 'item')
        del user_activity, users, data

    def __str__(self):
        if self.is_func:
            return self.data_func.__name__

    # def data(self):
    #     if self.is_func:
    #         return self.data_func()
    #     else:
    #         return self.data_func
