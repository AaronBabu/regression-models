from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SVMPrediction:

    """
    SVM Prediction model
    x_train -> input training data
    y_train -> output training data
    x_test -> input test data
    self.w - transformation matrix of weights
    uses random_search to optimize hyperparameters for transformation matrix
    """

    def __init__(self, x_train, y_train, kernel='rbf', C=1.0):
        # initializes hyperparameters for SVM classifer
        self.kernel = kernel
        self.C = C
        self.classifier = SVR(kernel=self.kernel, C=self.C)
        self.w = 0
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def train(self):
        # determines the ideal transformation matrix for prediction
        # performs a random_search operation to optimize appropriate hyperparameters for w
        distribution = {'C': [0.1, 1, 10, 100],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
        random_search = RandomizedSearchCV(self.classifier, param_distributions=distribution, cv=5, n_iter=10)
        random_search.fit(self.x_train, self.y_train)
        self.classifier = random_search.best_estimator_
        
        
    def predict(self, x_test):
        # predicts output data given test data
        return self.classifier.predict(x_test)
    
# Load the data from the csv file
data = pd.read_csv('training_data.csv')
x_train = data[['Correct', 'Total']].values
y_train = data['Cluster Rating'].values

# Create and train the model
model = SVMPrediction(x_train, y_train)
model.train()

# Predict the y for the given x values
x_pred = [[1, 2], [2, 2], [3, 5], [5, 5]]
y_pred = model.predict(x_pred)
print(y_pred)
        
