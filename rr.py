import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RidgeRegression(BaseEstimator, RegressorMixin):

    def __init__(self, λ = 1.0):
        self.λ = λ
        self.w_ = None

    def fit(self, X, y):
        
        # insert dummy feature
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        
        I = np.identity(X.shape[1])
        I[0][0] = 0                         # justify this line

        self.w_ = np.linalg.inv(X.T @ X + self.λ * I) @ X.T @ y

    def predict(self, X):

        if self.w_ is None:
            raise RuntimeError('Model is still to fit')
        
        # insert dummy feature
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) 

        return X @ self.w_