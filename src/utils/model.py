import numpy as np
import pandas as pd
import time
from functools import wraps

def normalize_ratings(Y, R):
    Ymean = np.zeros(Y.shape[0])
    for i in range(Y.shape[0]):
        if np.sum(R[i, :]) > 0:
            Ymean[i] = np.sum(Y[i, :] * R[i, :]) / np.sum(R[i, :])

    Ynorm = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        Ynorm[i, :] = (Y[i, :] - Ymean[i]) * R[i, :]

    return Ynorm, Ymean

def compute_popularity(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.Series:
    def melt_sparse(df):
        df_sparse = df.astype(pd.SparseDtype("float", fill_value=0))
        df_long = df_sparse.reset_index().melt(
            id_vars='user_id', 
            var_name='video_id', 
            value_name='watch_ratio'
        )
        return df_long[df_long['watch_ratio'] > 1e-3]

    train_long = melt_sparse(df_train)
    test_long = melt_sparse(df_test)

    combined_long = pd.concat([train_long, test_long], axis=0, ignore_index=True)\
                    .drop_duplicates(subset=['user_id', 'video_id'], keep='first')

    combined_long_sparse = combined_long.pivot(index='user_id', columns='video_id', values='watch_ratio')\
                                        .astype(pd.SparseDtype("float", fill_value=0))

    item_popularity = (combined_long_sparse > 1e-3).sum(axis=0) / len(combined_long_sparse)
    return item_popularity

def timeit(method):
    @wraps(method)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(f"{method.__name__} took {(end_time - start_time):.4f} seconds")
        return result
    return timed
