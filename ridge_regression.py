from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.epsilon = 1e-4  
        
    def train(self, x_data, y_data):
        y_data = y_data + self.epsilon  
        y_data_log = np.log(y_data)  
        self.model.fit(x_data, y_data_log)
        self.model.coef_[0] *= 0.5
        
    def predict(self, x_new):
        y_pred_log = self.model.predict(x_new)
        y_pred = np.exp(y_pred_log)  
        return y_pred

# Load the CSV file
data = pd.read_csv('training_data.csv')

# Separate the input and output data
X = data[['Correct', 'Total']].values
y = data['Cluster Rating'].values

# Train the model
rr = RidgeRegression()
rr.train(X, y)

# Make predictions on new data
X_new = np.array([[1, 2], [2, 2], [4, 5], [5, 5]])
y_pred = rr.predict(X_new)

# Plot the input data and the regression line
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y)
ax.set_xlabel('Correct')
ax.set_ylabel('Total')
ax.set_title('Input Data')

# Create a meshgrid of points to evaluate the regression function
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = rr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the regression line as a contour plot
ax.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(y.min(), y.max(), 10))
print(y_pred)
plt.show()



# Plot the data




