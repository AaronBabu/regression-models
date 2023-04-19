import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:

    """
    Log Regression Prediction model
    x_data -> input training data
    y_data -> output training data
    self.w - transformation matrix w
    
    Creates and trains a Log Regression model to be used for predictions
    """

    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        
    #Maps z onto sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    #Calculates the likelihood minimizing loss
    def likelihood(self, theta):
        z = np.dot(self.x_data, theta)
        y_pred = self.sigmoid(z)
        loss = -self.y_data * np.log(y_pred) - (1 - self.y_data) * np.log(1 - y_pred)
        return np.mean(loss)
    
    #Minimizes loss using gradient 
    def gradient(self, theta):
        z = np.dot(self.x_data, theta)
        y_pred = self.sigmoid(z)
        error = y_pred - self.y_data
        return np.dot(self.x_data.T, error) / self.y_data.size
    
    #Trains the data
    def train(self, lr=0.5, num_iterations=1000):
        self.theta = np.zeros(self.x_data.shape[1])
        self.loss_history = []
        
        for i in range(num_iterations):
            self.theta -= lr * self.gradient(self.theta)
            self.loss_history.append(self.likelihood(self.theta))
            
    #Used to predict y values given x values 
    def predict(self, x):
        x = np.array(x)
        z = np.dot(x, self.theta)
        return self.sigmoid(z)
    
    def export_weights(self):
        return self.theta
    
# Load the data from the csv file
data = pd.read_csv('training_data.csv')
x_data = data[['Correct', 'Total']].values
y_data = data['Cluster Rating'].values
model = LogisticRegression(x_data, y_data)
# Create and train the model
learningRates = [0.001, 0.005, 0.01, 0.07, 0.1, 0.4, 0.6, 1, 2, 0.8, 0.05]
for lr in learningRates:
    model.train(lr=lr)

    # Predict the y for the given x values
    x_pred = [[1, 2], [2, 2], [3, 5], [5, 5]]
    y_pred = model.predict(x_pred)
    print("lr: " + str(lr) + " " + str(y_pred))

w = model.export_weights()
x_test = [[1, 2], [2, 2], [3, 5], [5, 5]]
print(1 / (1 + np.exp(-(np.dot(x_test, w)))))
print(w)



# Plot the model and the predictions
x_range = np.linspace(0, 6, 100)
y_range = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x_range, y_range)
theta = np.array([model.theta[0], model.theta[1], model.theta[1]])
zz = model.sigmoid(np.dot(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()], theta)).reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap=plt.cm.RdBu_r, alpha=.8)
plt.colorbar()
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap=plt.cm.RdBu_r, edgecolors='k')
plt.scatter(np.array(x_pred)[:, 0], np.array(x_pred)[:, 1], c=y_pred, cmap=plt.cm.RdBu_r, edgecolors='k', marker='x')
plt.xlabel('Number of questions answered correctly')
plt.ylabel('Total number of questions')
plt.show()