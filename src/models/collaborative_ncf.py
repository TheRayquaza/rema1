import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from src.models.base import AbstractModel
from src.utils.model import normalize_ratings, timeit

class CollaborativeFilteringNCF(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @timeit
    def fit(self, data: pd.DataFrame, num_features=30, lambda_=1.0, iterations=100, learning_rate=1e-2,
            reg=False, min_mse=1e-3, lr_decay=0.5, min_lr=1e-5):
        data = data.fillna(0)
        self.user_item_matrix = data.copy()
        R = (data != 0).astype(float).values
        Y = data.values

        self.user_indices, self.item_indices = data.index.values, data.columns.values

        num_users, num_items = Y.shape
        Ynorm, self.Ymean = normalize_ratings(Y, R)
        Y_tensor = tf.constant(Ynorm, dtype=tf.float32)
        R_tensor = tf.constant(R, dtype=tf.float32)

        self.U = tf.Variable(tf.random.normal((num_users, num_features), stddev=0.01, dtype=tf.float32), name="X")
        self.V = tf.Variable(tf.random.normal((num_items, num_features), stddev=0.01, dtype=tf.float32), name="W")
        self.b = tf.Variable(tf.zeros((1, num_items),  dtype=tf.float32), name="b")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        bar = tqdm(range(iterations), desc="Training NCF", colour="green", dynamic_ncols=False)
        for i in bar:
            loss = 0
            with tf.GradientTape() as tape:
                Y_pred = tf.matmul(self.U, tf.transpose(self.V)) + self.b
                loss = tf.reduce_sum(R_tensor * tf.square(Y_tensor - Y_pred)) / tf.reduce_sum(R_tensor)
                if reg:
                    loss += lambda_ * (tf.reduce_sum(tf.square(self.U)) + tf.reduce_sum(tf.square(self.V)))

            grads = tape.gradient(loss, [self.U, self.V, self.b])
            self.optimizer.apply_gradients(zip(grads, [self.U, self.V, self.b]))

            current_lr = self.optimizer.learning_rate.numpy()
            if loss < min_mse and current_lr > min_lr:
                new_lr = max(current_lr * lr_decay, min_lr)
                self.optimizer.learning_rate.assign(new_lr)

            bar.set_description_str(
                f"Epoch {i+1}/{iterations} | MSE: {(tf.reduce_sum(R_tensor * tf.square(Y_tensor - Y_pred)) / tf.reduce_sum(R_tensor)).numpy():.4f} | " + \
                f"MAE: {(tf.reduce_sum(tf.abs(R_tensor * (Y_tensor - Y_pred))) / tf.reduce_sum(R_tensor)).numpy():.4f} | " + \
                f"LR: {self.optimizer.learning_rate.numpy():.5f}"
            )

        self.U_np = self.U.numpy()
        self.V_np = self.V.numpy()
        self.b_np = self.b.numpy()
        return self

    def get_marks(self, user_ids=None):
        if user_ids is None:
            user_ids = self.user_indices

        predictions = {}
        full_pred_matrix = np.matmul(self.U_np, self.V_np.T) + self.b_np
        full_pred_matrix += self.Ymean[:, np.newaxis]  # un-normalize

        for user_id in user_ids:
            if user_id not in self.user_indices:
                print(f"User {user_id} not found in training data (cold start not handled).")
                predictions[user_id] = []
                continue

            user_idx = np.where(self.user_indices == user_id)[0][0]
            predictions[user_id] = full_pred_matrix[user_idx].copy()
        return predictions

    @timeit
    def predict(self, user_ids=None, top_n=5):
        if user_ids is None:
            user_ids = self.user_indices

        predictions = {}
        full_pred_matrix = np.matmul(self.U_np, self.V_np.T) + self.b_np
        full_pred_matrix += self.Ymean[:, np.newaxis]  # un-normalize

        for user_id in user_ids:
            if user_id not in self.user_indices:
                self.logger(f"User {user_id} not found in training data (cold start not handled).")
                predictions[user_id] = []
                continue

            user_idx = np.where(self.user_indices == user_id)[0][0]
            user_pred_ratings = full_pred_matrix[user_idx].copy()

            user_pred_ratings[self.user_item_matrix.iloc[user_idx].to_numpy() > 0] = -np.inf

            top_indices = np.argsort(user_pred_ratings)[-top_n:][::-1]
            recommendations = [int(self.item_indices[i]) for i in top_indices]
            predictions[user_id] = recommendations

        return predictions

    def load(self, path: str):
        saved = np.load(path, allow_pickle=True)
        self.U_np = saved["U"]
        self.V_np = saved["V"]
        self.b_np = saved["b"]
        self.Ymean = saved["Ymean"]
        self.user_indices = saved["user_indices"]
        self.item_indices = saved["item_indices"]
        self.logger.info(f"Model loaded from {path}")
        return self

    def save(self, path: str):
        np.savez(path,
                 U=self.U_np,
                 V=self.V_np,
                 b=self.b_np,
                 Ymean=self.Ymean,
                 user_indices=self.user_indices,
                 item_indices=self.item_indices)
        self.logger.info(f"Model saved to {path}")
        return self
