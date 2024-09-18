import numpy as np

X = np.array([
    [1,3,4],
    [2,4,5],
    [3,5,6]
])

y = np.array([0,1,1])

threshold = 2
left_indices = X[:,0] < threshold
print(X[left_indices])