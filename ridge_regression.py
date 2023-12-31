import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RR(BaseEstimator, RegressorMixin):

    def __init__(self, λ = 1.0):
        
        if λ < 0: raise ValueError("λ must be >= 0")
        self.λ = λ
        
        self.w_ = None

    def fit(self, X, y):
        
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        
        I = np.identity(X.shape[1])
        I[0][0] = 0

        self.w_ = np.linalg.inv(X.T @ X + self.λ * I) @ X.T @ y

    def predict(self, X):

        if self.w_ is None:
            raise RuntimeError('Model is still to fit')
        
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) 

        return X @ self.w_