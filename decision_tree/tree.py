class Node:
    def __init__(self, feature_index = None, left = None, right = None, threshold = None, info_gain = None, value = None):    
        '''
        初始化节点
        feature_index代表要选取分割的特征值
        threshold代表进行分裂的阈值
        info_gain代表信息增益
        value代表节点值
        '''
        self.feature_index = feature_index
        self.left = left
        self.right = right
        self.threshold = threshold
        self.info_gain = info_gain
        self.value = value
    