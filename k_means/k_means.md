# K-Means 算法

假设有N个数据点，每个数据有4个维度
1. 随机选取k个初始聚类中心

2. 计算每个数据点到聚类中心的距离判断数据点的归类
   - 使用欧几里得距离(Euclidean Distance)
   - $d = \sqrt{(d_1)^2 + (d_2)^2 + (d_3)^2 + (d_4)^2}$
   - $d_n$ 对应特征之间的绝对值
   - 距离最近的作为一类
  
3. 计算SSE(Sum of Squared Errors)
   - 有所数据点到其所属聚类中心距离之和
   - $SSE = \sum_{i=1}^{N} \sum_{k=1}^{K} r_{ik} \| x_i - \mu_k \|^2$
   -  $\mu_{k1}, \mu_{k2}, \mu_{k3}, \mu_{k4}$ 是聚类中心 $\mu_k$ 的四个特征。

   - $x_{i1}, x_{i2}, x_{i3}, x_{i4}$ 是数据点 $x_i$ 的四个特征。$r_{ik}$ 是指示函数，如果数据点 $x_i$ 属于聚类 $k$，则 $r_{ik} = 1$，否则 $r_{ik} = 0$。
   - SSE是评估聚类效果的重要依据，也可以判断聚类过程中的收敛性。

4. 更新聚类中心，在每一个聚类中计算每个维度的平均值作为新聚类中心对应维度的值。

5. 重复2至4步骤

### Hints：
* 在K-Means中，初始聚类中心的随机选择具有较大偶然性，所以会重复很多次初始化。根据SSE可以判断哪个模型更优
* ATTENTION！！！！！！！！！！！
    - 没有代码演示，因为我懒