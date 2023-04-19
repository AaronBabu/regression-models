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

    def export_unscaled_weights(self):
        # Retrieve the scaled coefficients and intercept
        scaled_coef = self.model.named_steps['ridge'].coef_
        scaled_intercept = self.model.named_steps['ridge'].intercept_

        # Retrieve the scaler's mean and scale (std deviation)
        mean = self.model.named_steps['scaler'].mean_
        scale = self.model.named_steps['scaler'].scale_

        # Calculate the unscaled coefficients and intercept
        unscaled_coef = scaled_coef / scale
        unscaled_intercept = scaled_intercept - np.dot(mean, unscaled_coef)

        return unscaled_coef, unscaled_intercept


# Read data from the CSV file
data = pd.read_csv("training_data.csv")
x_data = data[["Correct", "Total"]].values
y_data = data["Cluster Rating"].values

# Initialize, train, and predict
ridge_regression = RidgeRegression(x_data, y_data)
ridge_regression.train()
x_test = np.array([[1, 2], [2, 2], [1, 5], [5, 5]])
y_pred = ridge_regression.predict(x_test)
w, intercept = ridge_regression.export_unscaled_weights()

# Print the predictions
print("Predictions:", y_pred)
print("W:", w)
print("Intercept:", intercept)
print((np.dot(x_test, w) + intercept))


# Plot the model and the predictions
plt.scatter(x_data[:, 0], y_data, color='blue', label='Training data')
plt.scatter(x_test[:, 0], y_pred, color='red', label='Predicted data')
plt.xlabel("Questions answered correctly")
plt.ylabel("Competency rating")
plt.legend()
plt.show()
