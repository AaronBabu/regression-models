import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class RidgeRegression:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])

    def train(self):
        self.model.fit(self.x_data, self.y_data)

    def predict(self, x_test):
        return self.model.predict(x_test)

# Read data from the CSV file
data = pd.read_csv("training_data.csv")
x_data = data[["Correct", "Total"]].values
y_data = data["Cluster Rating"].values

# Initialize, train, and predict
ridge_regression = RidgeRegression(x_data, y_data)
ridge_regression.train()
x_test = np.array([[1, 2], [2, 2], [3, 5], [5, 5]])
y_pred = ridge_regression.predict(x_test)

# Print the predictions
print("Predictions:", y_pred)

# Plot the model and the predictions
plt.scatter(x_data[:, 0], y_data, color='blue', label='Training data')
plt.scatter(x_test[:, 0], y_pred, color='red', label='Predicted data')
plt.xlabel("Questions answered correctly")
plt.ylabel("Competency rating")
plt.legend()
plt.show()
