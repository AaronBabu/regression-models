import numpy as np
from numpy import reshape

class LogLikelihood:
    
    def __init__(self):
        #self.w -> Transformation function matrix 
        self.w = None
    
    """
    def train(self, x_data, y_data, learning_rate=0.01, num_iterations=1000)
        x_data -> input x training data for model 
        y_data -> result y training data for model
        learning_rate -> learning rate for regression model
        num_iterations -> number of iterations to cycle through 

    Creates a model given the training data using logistic regression 
        
    Returns new self.w matrix
    """
    def train(self, x_data, y_data, learning_rate=0.01, num_iterations=1000):
        n = len(x_data)
        x = np.hstack((np.ones((n, 1)), x_data.reshape(n, 1)))
        y = y_data.reshape(n, 1)
        self.w = np.zeros((2, 1))
        
        for i in range(num_iterations):
            y_pred = np.dot(x, self.w)
            error = y_pred - y
            gradient = np.dot(x.T, error)
            self.w -= learning_rate * gradient
        
        return self.w
    
    def predict(self, x_test):
        n = len(x_test)
        x = np.hstack((np.ones((n, 1)), x_test.reshape(n, 1)))
        y_pred = np.dot(x, self.w)
        return y_pred.flatten()