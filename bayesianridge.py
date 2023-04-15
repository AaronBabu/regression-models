import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RandomizedSearchCV


class BayesianRidgeWrapper:
    def __init__(self):
        self.model = None

    def train(self, x_train, y_train):
        # Define the hyperparameter search space
        distributions = {
            "n_iter": [int(x) for x in np.linspace(start=10, stop=200, num=10)],
            "alpha_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "alpha_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "lambda_1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            "lambda_2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        }

        # Create a RandomizedSearchCV object
        bayes_random = RandomizedSearchCV(
            estimator=BayesianRidge(),
            param_distributions=distributions,
            n_iter=10,
            cv=5,
            n_jobs=-1,
        )
        bayes_random.fit(x_train, y_train)
        self.model = bayes_random.best_estimator_

    def predict(self, x_test):
        return self.model.predict(x_test)


# Load the data from the csv file
data = pd.read_csv("training_data.csv")
x_train = data[["Correct", "Total"]].values
y_train = data["Cluster Rating"].values

# Create and train the model
model = BayesianRidgeWrapper()
model.train(x_train, y_train)

# Predict the y for the given x values
x_test = [[1, 2], [2, 2], [3, 5], [5, 5]]
y_pred = model.predict(x_test)
print(f"Predictions: {y_pred}")
