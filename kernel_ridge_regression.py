import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator, RegressorMixin

class KRR(BaseEstimator, RegressorMixin):

    def __init__(self, λ = 1.0, γ = 1.0):

        if λ < 0: raise ValueError("λ must be >= 0")
        if γ <= 0: raise ValueError("γ must be > 0")
        self.λ = λ
        self.γ = γ

        self.X_ = None
        self.c_ = None 

    def gaussian_kernel(self, X, Y):
        return np.exp((-0.5 / self.γ) * np.square(distance.cdist(X, Y, 'euclidean')))

    def fit(self, X, y):
        
        self.X_ = X.values
        K = self.gaussian_kernel(self.X_, self.X_)
        self.c_ = y.T @ np.linalg.inv(K + self.λ * np.identity(self.X_.shape[0]))

    def predict(self, X):

        if self.c_ is None or self.X_ is None:
            raise RuntimeError('Model is still to fit')
        
        return self.c_ @ self.gaussian_kernel(self.X_, X.values)