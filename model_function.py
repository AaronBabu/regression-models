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

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

def decision_function(x, support_vectors, dual_coefs, intercept, gamma):
    kernel_values = np.array([rbf_kernel(x, sv, gamma) for sv in support_vectors])
    decision_value = np.dot(dual_coefs, kernel_values) + intercept
    return decision_value

def svm(x_test, support_vectors, dual_coefs, intercept, gamma):
    decision_values = []
    for x in x_test:
        decision_value = decision_function(x, support_vectors, dual_coefs, intercept, gamma)
        decision_values.append(decision_value)
    return np.array(decision_values)

parsed_content = {
    'Cluster 1': {'concept1': (1, 2), 'concept2': (3, 5)},
    'Cluster 2': {'concept3': (1, 5), 'concept4': (5, 5)}
}

print(parseToConceptPicker(parsed_content, "svm"))



