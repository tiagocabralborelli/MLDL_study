import numpy as np 

def step_function(x):
    return np.whete(x > 1,1,0)

class Perceptron:
    def __init__(self, lr = 0.01, n_inter = 1000)
    self.lr = lr
    self.n_inter = n_inter
    self.activation = step_function
    self.weigths = None
    self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape

        self.weigths = np.random(2,n_features)
        self.bias = 0

        prediction = np.dot(self.weigths*X) + self.bias
         

