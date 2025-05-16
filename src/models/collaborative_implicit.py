from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import pandas as pd
import numpy as np
from src.models.base import AbstractModel
from src.utils.model import timeit

class CollaborativeFilteringImplicitALS(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @timeit
    def fit(self, data: pd.DataFrame, num_features=100, lambda_=1, iterations=25, alpha=50, random_state=None):
        self.user_item_matrix = data.fillna(0)
        self.user_indices = data.index.values
        self.item_indices = data.columns.values

        item_user_matrix = csr_matrix(self.user_item_matrix.values.T)

        self.model = AlternatingLeastSquares(
            factors=num_features,
            regularization=lambda_,
            iterations=iterations,
            alpha=alpha,
            calculate_training_loss=True,
            random_state=random_state
        )
        self.model.fit(item_user_matrix)
        return self

    def get_marks(self, user_ids=None):
        if user_ids is None:
            user_ids = self.user_indices

        predictions = {}
        for user_id in user_ids:
            if user_id not in self.user_indices:
                print(f"User {user_id} not found.")
                predictions[user_id] = []
                continue

            user_idx = np.where(self.user_indices == user_id)[0][0]
            user_items = csr_matrix(self.user_item_matrix.loc[user_id].values.reshape(1, -1))
            ids, scores = self.model.recommend(user_idx, user_items, N=len(self.item_indices), filter_already_liked_items=False)
            user_predictions = {int(self.item_indices[i]): float(scores[j]) for j, i in enumerate(ids)}
            predictions[user_id] = user_predictions

        return predictions
    
    @timeit
    def predict(self, user_ids=None, top_n=5):
        marks = self.get_marks(user_ids)
        predictions = {}
        for user_id in user_ids:
            sorted_items = sorted(marks[user_id].items(), key=lambda x: x[1], reverse=True)
            predictions[user_id] = [int(item) for item, score in sorted_items[:top_n]]
        return predictions

    def load(self, path: str):
        self.model = AlternatingLeastSquares().load(path+".npz")
        saved = np.load(path+"metadata.npz", allow_pickle=True)
        matrix = saved["matrix"]
        self.user_indices = saved["user_indices"]
        self.item_indices = saved["item_indices"]
        self.user_item_matrix = pd.DataFrame(matrix, index=self.user_indices, columns=self.item_indices)
        self.logger.info(f"Model loaded from {path}")
        return self

    def save(self, path: str):
        self.model.save(path)
        np.savez(path+"metadata.npz", 
            matrix=self.user_item_matrix,
            user_indices=self.user_indices,
            item_indices=self.item_indices,
        )
        self.logger.info(f"Model saved to {path}")
        return self
