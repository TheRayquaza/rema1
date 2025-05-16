import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.model import normalize_ratings, timeit
from src.models.base import AbstractModel

class CollaborativeFilteringALS(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.U = None
        self.V = None
        self.Y = None
        self.Ymean = None
        self.user_indices = None
        self.item_indices = None
        self.user_item_matrix = None
        self.losses = []

    @timeit
    def fit(self, data: pd.DataFrame, num_features=30, lambda1_=0.1, lambda2_=0.1,
            iterations=100, early_stopping=1e-3, reg=True, scale=0.2, normalization=False):

        R = (~data.isna()).astype(float).values
 
        self.user_item_matrix = data.copy()
        Y_hat = data.copy().fillna(0).values

        if normalization:
            Y_hat, self.Ymean = normalize_ratings(Y_hat, R)
        else:
            self.Ymean = np.zeros(Y_hat.shape[0])

        self.user_indices = data.index.values
        self.item_indices = data.columns.values

        num_users, num_items = Y_hat.shape
        # Initialize user factors matrix
        self.U = np.random.normal(loc=0.0, scale=scale, size=(num_features, num_users))
        self.V = None

        self.losses = []

        for i in range(iterations):
            # Update item factors
            if not reg:
                self.V = np.linalg.solve(
                    self.U @ self.U.T,
                    self.U @ Y_hat
                )
            else:
                self.V = np.linalg.solve(
                    self.U @ self.U.T + lambda1_ * np.eye(num_features),
                    self.U @ Y_hat
                )

            # Update user factors
            if not reg:
                self.U = np.linalg.solve(
                    self.V @ self.V.T,
                    self.V @ Y_hat.T
                )
            else:
                self.U = np.linalg.solve(
                    self.V @ self.V.T + lambda2_ * np.eye(num_features),
                    self.V @ Y_hat.T
                )

            # Predict ratings
            Y = np.zeros_like(Y_hat)
            for u in range(num_users):
                Y[u, :] = self.U[:, u].T @ self.V + self.Ymean[u]

            if np.isnan(Y).any():
                raise ValueError("NaNs detected in prediction matrix Y")

            # Calculate loss only on observed ratings
            loss = np.sum(np.square((Y_hat - Y) * R)) / np.sum(R)
            self.losses.append(loss)

            if i % 5 == 0:
                print(f"Iteration {i}: MSE loss = {loss:.4f}")
                print(f"Mean absolute error (watch ratio predicted gap) {(np.sum(np.abs((Y_hat - Y) * R)) / np.sum(R)):.4f}")

            if i >= 5 and np.all(np.abs(np.diff(self.losses[-5:])) < early_stopping):
                print(f"Early stopping at iteration {i}, recent losses: {self.losses[-5:]}")
                break

        return self

    def plot_loss(self, save=False, show=True, path=None):
        if not self.losses:
            print("No loss data to plot. Run `fit()` first.")
            return
        if save and not path:
            raise ValueError("Path must be specified if save=True")

        plt.figure(figsize=(8, 5))
        plt.plot(self.losses, label="Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss Over Iterations")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(path)
        if show:
            plt.show()

    def get_marks(self, user_ids=None):
        if user_ids is None:
            pred_matrix = np.zeros((len(self.user_indices), len(self.item_indices)))
            for u in range(len(self.user_indices)):
                pred_matrix[u, :] = self.U[:, u].T @ self.V + self.Ymean[u]
            return pred_matrix

        predictions = {}
        for user_id in user_ids:
            if user_id not in self.user_indices:
                print(f"User {user_id} not found in training data (cold start not handled).")
                predictions[user_id] = []
                continue

            user_idx = np.where(self.user_indices == user_id)[0][0]
            user_pred = self.U[:, user_idx].T @ self.V + self.Ymean[user_idx]
            predictions[user_id] = user_pred

        return predictions

    @timeit
    def predict(self, user_ids=None, top_n=5):
        marks = self.get_marks(user_ids)
        predictions = {}
        for user_id in user_ids:
            top_indices = np.argsort(marks[user_id])[-top_n:][::-1]
            recommended_items = [int(self.item_indices[i]) for i in top_indices]
            predictions[user_id] = recommended_items

        return predictions
    
    def save(self, path: str):
        np.savez(path,
                U=self.U,
                V=self.V,
                Ymean=self.Ymean,
                user_indices=self.user_indices,
                item_indices=self.item_indices,
                losses=self.losses)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        npzfile = np.load(path)
        self.U = npzfile['U']
        self.V = npzfile['V']
        self.Ymean = npzfile['Ymean']
        self.user_indices = npzfile['user_indices']
        self.item_indices = npzfile['item_indices']
        self.losses = npzfile['losses'].tolist()
        self.logger.info(f"Model loaded from {path}")
        return self
