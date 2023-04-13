from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

class SVMPrediction:

    """
    SVM Prediction model
    x_train -> input training data
    y_train -> output training data
    x_test -> input test data
    self.w - transformation matrix of weights
    uses random_search to optimize hyperparameters for transformation matrix
    """

    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        # initializes hyperparameters for SVM classifer
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.classifier = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self.w = 0
        
    def train(self, x_train, y_train):
        # determines the ideal transformation matrix for prediction
        # performs a random_search operation to optimize appropriate hyperparameters for w
        distribution = {'C': [0.1, 1, 10, 100],
                      'gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
        random_search = RandomizedSearchCV(self.classifier, param_distributions=distribution, cv=5, n_iter=10)
        random_search.fit(x_train, y_train)
        self.classifier = random_search.best_estimator_
        self.w = self.classifier.coef_
        return self.w
        
    def predict(self, x_test):
        # predicts output data given test data
        return self.classifier.predict(x_test)
    