import numpy as np 
import values
 
"""
Inputs:
parsedContent = {
     'Cluster 1' : {'concept1' : (1, 2), 'concept2' : (3, 5)}, 
     'Cluster 2' : {'concept3' : (1, 5), 'concept4' : (5, 5)}
}
"""
def parseToConceptPicker(parsed_content, model):
    assert model in ["ridge", "log", "svm", "bayesianridge"]
    if model == "ridge":
        return ridge(parsed_content, values.RIDGE_W, values.RIDGE_INTERCEPT)
    if model == "log": 
        return log(parsed_content, values.LOG_W)
    if model == "bayesianridge":
        return ridge(parsed_content, values.BAYESIANRIDGE_W, values.BAYESIANRIDGE_INTERCEPT)
    if model == "svm": 
         return svm(parsed_content, values.SVM_SUPPORT_VECTORS, values.SVM_DUAL_COEFS, values.SVM_INTERCEPT, values.SVM_GAMMA)

def ridge(parsed_content, w, intercept):
    updated_content = {}
    for cluster, concepts in parsed_content.items():
        updated_concepts = {}
        for concept, x_values in concepts.items():
            x_values = np.array(x_values).reshape(1, -1)
            pred = np.dot(x_values, w) + intercept
            updated_concepts[concept] = pred[0]
        updated_content[cluster] = updated_concepts
    return updated_content

def log(parsed_content, w):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    updated_content = {}
    for cluster, concepts in parsed_content.items():
        updated_concepts = {}
        for concept, x_values in concepts.items():
            x_values = np.array(x_values).reshape(1, -1)
            z = np.dot(x_values, w)
            pred = sigmoid(z)[0]
            updated_concepts[concept] = pred
        updated_content[cluster] = updated_concepts
    return updated_content

def svm(x_test, support_vectors, dual_coef, bias, gamma):
    def RBF(x,z,gamma,axis=None):
        x = np.array(x)
        z = np.array(z)
        x = x.reshape(1, -1)  # reshape x to (1, 2)
        z = z.reshape(1, -1)  # reshape z to (1, 2)
        return np.exp((-gamma*np.linalg.norm(x-z, axis=axis)**2))

    A = []
    # Loop over all suport vectors to calculate K(Xi, X_test), for Xi belongs to the set of support vectors
    for x in support_vectors:
        A.append(RBF(x, x_test, gamma))
    A = np.array(A)

    return (np.sum(dual_coef*A)+bias)



# def rbf_kernel(x1, x2, gamma):
#     return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# def decision_function(x, support_vectors, dual_coefs, intercept, gamma):
#     kernel_values = np.array([rbf_kernel(x, sv, gamma) for sv in support_vectors])
#     decision_value = np.dot(dual_coefs, kernel_values) + intercept
#     return decision_value

# def svm(x_test, support_vectors, dual_coefs, intercept, gamma):
#     decision_values = []
#     for x in x_test:
#         decision_value = decision_function(x, support_vectors, dual_coefs, intercept, gamma)
#         decision_values.append(decision_value)
#     return np.array(decision_values)

parsed_content = {
    'Cluster 1': {'concept1': (1, 2), 'concept2': (3, 5)},
    'Cluster 2': {'concept3': (3, 5), 'concept4': (5, 5)}
}

parsed_content = [1, 2]

print(parseToConceptPicker(parsed_content, "svm"))

"""
outputs -> {
    'Cluster 1': {'concept1': 0.5129150789377697, 'concept2': 0.6130149189377697}, 
    'Cluster 2': {'concept3': 0.6130149189377697, 'concept4': 0.9449037589377697}
    }
"""


