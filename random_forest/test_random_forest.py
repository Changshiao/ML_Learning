# test_decision_tree.py

import numpy as np
from random_forest import Random_forest
"""
样本集X
第一个是喜欢吃甜食程度,与数字成线性正相关,约定3是正常值
第二个是性别,1男,0女
第三个是是否运动,1是,0否

预测集y
1为瘦,2为正常,3为胖
"""
X = np.array([[1, 0, 0],
              [1, 0, 1],
              [3, 1, 1],
              [6, 0, 0],
              [6, 0, 1],
              [4, 1, 1]])

y = np.array([1, 1, 2, 3, 2, 2])

tree_test = Random_forest(num_tree=10, sample_size=3, max_depth=5)
tree_test.fit(X, y)
tree_test_predict = tree_test.predict(X)
if np.array_equal(tree_test_predict, y):
    print("correctly building!")
test_X = np.array([[5, 1, 0],
                   [2, 0, 1]])
predictions = tree_test.predict(test_X)
print(predictions)