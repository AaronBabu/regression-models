import numpy as np
from numpy import reshape

class LogLikelihood:
    
    def __init__(self):
        # Initialize self.w as None, it will hold the transformation function matrix 
        self.w = None
    
    """
    Train the logistic regression model using the given training data.
    
    Parameters:
        x_data: input x training data for the model
        y_data: result y training data for the model
        learning_rate: learning rate for regression model
        num_iterations: number of iterations to cycle through
    
    Returns:
        The new self.w matrix.
    """
    def train(self, x_data, y_data, learning_rate=0.01, num_iterations=1000):
        # Get the length of the input data
        n = len(x_data)
        # Add a column of ones to x_data and reshape y_data
        x = np.hstack((np.ones((n, 1)), x_data.reshape(n, 1)))
        y = y_data.reshape(n, 1)
        # Initialize the transformation function matrix as zeros
        self.w = np.zeros((2, 1))
        
        # Iterate through the number of iterations specified
        for i in range(num_iterations):
            # Calculate the predicted output using sigmoid function
            y_pred = self._sigmoid(np.dot(x, self.w))
            # Calculate the difference between predicted and actual output
            error = y - y_pred
            # Calculate the gradient using the dot product of transposed input data and error
            gradient = np.dot(x.T, error)
            # Update the transformation function matrix using the learning rate and gradient
            self.w += learning_rate * gradient
            
        return self.w
    
    """
    Use the trained model to predict the result y for the given input x data.
    
    Parameters:
        x_test: input x test data to predict y results
    
    Returns:
        The predicted y values.
    """
    def predict(self, x_test):
        n = len(x_test)
        x = np.hstack((np.ones((n, 1)), x_test.reshape(n, 1)))
        y_pred = self._sigmoid(np.dot(x, self.w))
        return y_pred.flatten()
    
    """
    Calculate the log-likelihood of the logistic regression model using the given training data.
    
    Parameters:
        x_data: input x training data for the model
        y_data: result y training data for the model
    
    Returns:
        The log-likelihood of the model.
    """
    def log_likelihood(self, x_data, y_data):
        n = len(x_data)
        # Add a column of ones to x_data and reshape y_data
        x = np.hstack((np.ones((n, 1)), x_data.reshape(n, 1)))
        y = y_data.reshape(n, 1)
        # Calculate the predicted output using sigmoid function
        y_pred = self._sigmoid(np.dot(x, self.w))
        # Calculate the log-likelihood using the formula
        ll = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return ll
    
    """
    Sigmoid function to be used in calculating the predicted output.
    
    Parameters:
        z: the input value for the sigmoid function
    
    Returns:
        The sigmoid value of the input.
    """
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    
#For testing purposes
import matplotlib.pyplot as plt

# Define the input x data and result y data
x_data = np.random.normal(size=100)
y_data = np.random.binomial(1, 1 / (1 + np.exp(-(1.5 * x_data - 0.5))), size=100)

# Train the logistic regression model
ll = LogLikelihood()
ll.train(x_data, y_data)

# Define the test x data to predict y values
x_test = np.linspace(-4, 4, 200)

# Use the trained model to predict the result y values
y_pred = ll.predict(x_test)
print(y_pred)

# Plot the input data points and the predicted line
plt.scatter(x_data, y_data, c=y_data)
plt.plot(x_test, y_pred)
plt.show()










