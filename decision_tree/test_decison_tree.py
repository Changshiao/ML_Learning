# test_decision_tree.py

import numpy as np
from decision_tree import DecisionTree

X = np.array([[2, 3],
              [1, 1],
              [4, 5],
              [6, 7]])

y = np.array([0, 0, 1, 1])


tree_fixed = DecisionTree(max_depth=7)

# Train the decision tree
tree_fixed.fit(X, y)

# Predict the training data
fixed_predictions = tree_fixed.predict(X)
print(fixed_predictions)