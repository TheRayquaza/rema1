import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from src.models.base import AbstractModel
from src.utils.model import timeit

class CollaborativeFiltering(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_similarity = None
        self.user_item_matrix = None
        self._prediction_cache = {}  # Internal cache for user predictions

    @timeit
    def fit(self, data: pd.DataFrame):
        self.user_item_matrix = data.fillna(0)
        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    @timeit
    def predict(self, user_id: int, top_n=5):
        if user_id not in self.user_item_matrix.index:
            print(f"user {user_id} not found, cold start not inferred")
            return []

        similar_users = self.user_similarity[user_id].drop(user_id).sort_values(ascending=False)

        weighted_scores = pd.Series(dtype=np.float64)
        for sim_user, sim_score in similar_users.items():
            user_videos = self.user_item_matrix.loc[sim_user]
            weighted_scores = weighted_scores.add(user_videos * sim_score, fill_value=0)

        pred = weighted_scores.sort_values(ascending=False).head(top_n).index.tolist()
        self._prediction_cache[user_id] = pred  # Cache the prediction
        return pred

    def load(self, path: str):
        with open(path, 'rb') as f:
            model = pickle.load(f)
            self.user_similarity = model.user_similarity
            self.user_item_matrix = model.user_item_matrix
            self._prediction_cache = model._prediction_cache
        return self

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
