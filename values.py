UPPER_THRESHOLD = 0.8
LOWER_THRESHOLD = 0.2
CONFIDENCE = 0.5

RIDGE_W = [0.16571224, -0.07708103]
RIDGE_INTERCEPT = 0.5012660902619637

LOG_W = [0.78651063, -0.36088408]

SVM_SUPPORT_VECTORS = [[0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                       [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                       [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                       [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                       [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                       [0., 1.], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                       [0., 1.], [0., 1.], [0., 1.], [0., 1.], [1., 1.], 
                       [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], 
                       [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], 
                       [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], 
                       [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], 
                       [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], 
                       [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [0., 2.], [0., 2.], [0., 2.], [0., 2.], [0., 2.], 
                       [1., 2.], [1., 2.], [1., 2.], [1., 2.], [1., 2.], 
                       [1., 2.], [2., 2.], [2., 2.], [2., 2.], [2., 2.], 
                       [2., 2.], [2., 2.], [2., 2.], [2., 2.], [2., 2.], 
                       [2., 2.], [2., 2.], [2., 2.], [2., 2.], [2., 2.], 
                       [2., 2.], [2., 2.], [2., 2.], [2., 2.], [2., 2.], 
                       [2., 2.], [2., 2.], [2., 2.], [2., 2.], [2., 2.], 
                       [2., 2.], [2., 2.], [2., 2.], [2., 2.], [2., 2.], 
                       [2., 2.], [0., 3.], [0., 3.], [0., 3.], [0., 3.], 
                       [0., 3.], [0., 3.], [2., 4.], [2., 4.], [2., 4.], 
                       [2., 4.], [2., 4.], [2., 4.], [2., 4.], [2., 4.], 
                       [2., 4.], [2., 4.], [2., 4.], [2., 4.], [0., 5.], 
                       [0., 5.], [0., 5.], [0., 5.], [0., 5.], [0., 5.], 
                       [0., 5.], [0., 5.], [0., 5.], [0., 5.], [0., 5.], 
                       [1., 5.], [1., 5.], [1., 5.], [1., 5.], [1., 5.], 
                       [1., 5.], [2., 5.], [2., 5.], [2., 5.], [2., 5.], 
                       [2., 5.], [2., 5.], [2., 5.], [2., 5.], [2., 5.], 
                       [2., 5.], [2., 5.], [2., 5.], [2., 5.], [2., 5.], 
                       [2., 5.], [2., 5.], [2., 5.], [2., 5.], [2., 5.], 
                       [2., 5.], [2., 5.], [2., 5.], [2., 5.], [2., 5.], 
                       [2., 5.], [2., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.], [5., 5.], 
                       [5., 5.], [5., 5.], [5., 5.], [5., 5.]]
SVM_DUAL_COEFS = [[-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, -10, -9.95990523, -10, -10, -10, -10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8.57792814, 10, 10, 7.22041046, 10, 0.12860298, 10, 10, -10, -10, -10, -10, -10, -10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, -10, -3.62925654, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -2.3377798, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]]
SVM_INTERCEPT = [0.50088448]
SVM_GAMMA = 0.18285714285714286





