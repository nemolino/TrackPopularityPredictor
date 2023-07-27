import numpy as np

class RidgeRegression:

    def __init__(self, λ = 1.0):
        self.λ = λ
        self.w = None

    def fit(self, X, y):
        
        # insert dummy feature
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        
        I = np.identity(X.shape[1])
        I[0][0] = 0                         # justify this line

        self.w = np.linalg.inv(X.T @ X + self.λ * I) @ X.T @ y

    def predict(self, X):

        if self.w is None:
            raise RuntimeError('Model is still to fit')
        
        # insert dummy feature
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) 

        return X @ self.w
    
    def get_params(self, deep=True):
        return {"λ": self.λ}