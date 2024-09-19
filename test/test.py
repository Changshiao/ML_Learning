# test_decision_tree.py

import numpy as np
from decision_tree import DecisionTree

# 创建一个简单的数据集
X = np.array([[2, 3],
              [1, 1],
              [4, 5],
              [6, 7]])

y = np.array([0, 0, 1, 1])

# 创建决策树实例
tree = DecisionTree(max_depth=2)

# 训练决策树
tree.fit(X, y)

# 预测
predictions = tree.predict(X)

print("Predictions:", predictions)

# 输出应为 [0, 0, 1, 1]，表示分类正确
