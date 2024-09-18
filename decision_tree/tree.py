import numpy as np

#所有的数据集都经过numpy处理过

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

#计算信息增益
def cal_info_gain(y, y_left, y_right):
    y_entropy = entropy(y)
    y_left_entropy = entropy(y_left)
    y_right_entropy = entropy(y_right)
    info_gain = y_entropy - len(y_left)/len(y) * y_left_entropy - len(y_right)/len(y) * y_right_entropy
    return info_gain

#计算熵
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def split(self, X, y):
#这一步是根据node的数据，对于要选取的feature逐一进行split再选取最好的feature_index
    feature_num = len(X[1]) #feature总个数
    left = []
    right = []
    max_info_index = -1 #默认最具代表性的feature为-1
    max_info_gain = 0 #默认最大信息增益为0
    for feature_index in range(feature_num):
        #split要点：两层循环，第一层是选择feature_index,第二层是在feature_index中选择shreshold，最优者的是feature_idnex中shreshold的选取        
        thresholds = np.unique(X[:,feature_index])#获取当前特征的所有唯一值
        for threshold in thresholds :
            #这里使用bool数组来记录左右树的样本index，ai说的，我觉得用下标存储更方便
            left_indicies = X[:,feature_index] < threshold
            right_indicies = X[: ,feature_index] >= threshold
            if len(left_indicies) != 0 and len(right_indicies) != 0:#如果分割出来出现了一端没有一段有那么就不算split成功
                X_left = X[left_indicies]
                X_right = X[right_indicies]
                y_left = y[left_indicies]
                y_right = y[right_indicies]
                #数据分割后计算entropy或者其他方法
                info_gain = cal_info_gain(y,y_left,y_right)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    max_info_index = feature_index#分割完成
                    best_threshold = threshold

    return max_info_index, best_threshold, max_info_gain

#在构建树的过程需要返回节点，构建树应该在class decision_tree中，需要重新架构

class desicion_tree:
    def __init__(self, ):
        pass


    pass
    