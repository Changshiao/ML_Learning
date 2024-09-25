from decision_tree import DecisionTree
import numpy as np

# 使用Bootstrae处理数据
def bootstrap_sample(X, y, num_samples, sample_size):
    """
    X样本集
    num_samples生成样本个数
    sample_size每个样本的大小
    """
    X_bootstrap_sample = np.empty((num_samples, sample_size, X.shape[1]))
    y_bootstrap_sample = np.empty((num_samples, sample_size))   
    # 创建一个储存Bootstrap样本的数值
    for sample in range(num_samples):
        indices = np.random.choice(X.shape[0], size=sample_size, replace=True)
        # 随机选取每个样本,通过索引来选择更加合适
        X_bootstrap_sample[sample] = X[indices]
        y_bootstrap_sample[sample] = y[indices]
    return X_bootstrap_sample,y_bootstrap_sample


class Random_forest():
    def __init__(self, num_tree=None, sample_size=None, max_depth=None):
        self.num_tree = num_tree  # 设置树木个数
        self.sample_size = sample_size  # bootstrap 样本大小
        self.max_depth = max_depth  # 决策树的最大深度
        self.forest = []  # 用于存储森林中的树  
    
    def _build_forest(self, X, y):
        X_bootstrap, y_bootstrap = bootstrap_sample(X, y, self.num_tree, self.sample_size)
        for i in range(self.num_tree):
            decisiontree = DecisionTree(max_depth=self.max_depth)
            decisiontree.fit(X_bootstrap[i], y_bootstrap[i])
            self.forest.append(decisiontree)
            
    def fit(self, X, y):
        self._build_forest(X, y)
            
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.forest])  # 返回所有的预测值，是一个二维数组
        """
        将二维数组转置对每一行进行最大值选取
        将每一行的最大值存储到预测值并返回
        """
        predictions = predictions.astype(int) 
        prediction = [int(np.bincount(row).argmax()) for row in predictions.T]
        return prediction