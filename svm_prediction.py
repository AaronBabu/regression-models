from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np


class SVMPrediction:

    """
    SVM Prediction model
    x_train -> input training data
    y_train -> output training data
    x_test -> input test data
    self.w - transformation matrix of weights
    uses random_search to optimize hyperparameters for transformation matrix
    """

    def __init__(self, kernel="rbf", C=1.0):
        # initializes hyperparameters for SVM classifer
        self.kernel = kernel
        self.C = C
        self.regression = SVR(kernel=self.kernel, C=self.C)

    def train(self, x_train, y_train):
        # determines the ideal transformation matrix for prediction
        # performs a random_search operation to optimize appropriate hyperparameters for w
        distribution = {
            "C": [1, 10, 100],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
        random_search = RandomizedSearchCV(
            self.regression, param_distributions=distribution, cv=5, n_iter=10
        )
        random_search.fit(np.array(x_train), np.array(y_train))
        self.regression = random_search.best_estimator_

    def predict(self, x_test):
        # predicts output data given test data
        return self.regression.predict(x_test)


# Load the data from the csv file
data = pd.read_csv("training_data.csv")
x_train = data[["Correct", "Total"]].values
y_train = data["Cluster Rating"].values

# Create and train the model
model = SVMPrediction()
model.train(x_train, y_train)

# Predict the y for the given x values
x_test = [[1, 2], [2, 2], [3, 5], [5, 5]]
y_pred = model.predict(x_test)
print(f"Predictions: {y_pred}")
