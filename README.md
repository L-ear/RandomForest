# RandomForest
自己使用python实现的Cart分类决策树和基于该决策树的随机森林(仅供参考)

代码依赖numpy和pandas，运行前请确保有这两个包

关键代码部分都有比较详细的注释(^-^)

决策树使用孩子表示法，因为决策树对每个sample的预测需要的是根据父节点找其孩子节点的操作


使用的数据集是kaggle比赛入门的经典数据集 titanic disaster，要求根据乘客的数据预测他是生存还是死亡，是一个二分类的问题。

原数据集仓库里也有，在original_data文件夹下。还是附个下载链接吧 https://www.kaggle.com/c/titanic/data


使用pandas对原数据集进行清洗和onehot编码，最后划分前600条数据作为训练集，后289条数据作为测试集。

pre_data.py做的就是这部分工作。

划分后的数据集在data文件夹下。


决策树只设置了一个可调参数：

min_sample_leaf(落在叶子上的最小样本数)

当min_sample_leaf = 31时，我跑的分类正确率为0.702422


随机森林有四个可调参数：

ip 随机挑选的样本比例为(ip,1)中的一个随机数

jp 随机挑选的特征比例

n_trees 树的数量

min_sample_leaf 叶节点上的最小样本数

当n_trees=60,min_sample_leaf=5,ip=0.85,jp=0.7时，我跑的分类正确率为0.813149

## 需要补充的一点是，对于输入的train和test数据集，代码默认label是放在第一列的
