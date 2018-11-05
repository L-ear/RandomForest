# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:02:26 2018

@author: L-ear
"""
import time
import pandas as pd
import numpy as np
#定义节点类
class node:
    def __init__(self,data_index,
                 left=None,right=None,
                 feature=None,split=None,
                 out = None
                 ):
        self.data_index = data_index# 集合 落在节点上集合的行索引
        self.left = left# int 左子树下标
        self.right = right# int 右子树下标
        self.feature = feature# string 分裂特征
        self.split = split# int or float 分割值
        self.out = out# 叶节点的输出值

def build_tree(S, min_sample_leaf):
    # S为构建决策树使用的数据集
    # min_sample_leaf为叶节点的最小样本数
    # 使用孩子表示法存储树结构
    root = node(S.index)
    tree = []
    tree.append(root)
    # i指向当前处理的叶节点
    i = 0
    # j指向tree列表的末尾元素，便于向父节点添加新的叶节点的索引
    j = 0
    # 循环
    # 调用divide函数处理第i个节点
    # 根据返回值判断第i个节点是否可划分
    # 若可划分，则向tree list并入两个新的叶节点，同时为第i个节点添加子树索引
    # 若不可划分，比较i，j大小，若i == j,则跳出循环
    # 否则进入下次循环
    while True:
        res = divide(S, tree[i], min_sample_leaf)
        if res:
            tree.extend(res)# 将两个叶节点并入树中
            tree[i].left = j+1
            tree[i].right = j+2
            j += 2
            i += 1
        elif i == j:
            break
        else:
            i += 1
    return tree


def divide(S, leaf, min_sample_leaf):
    # 划分叶节点，判断是否可划分
    data = S.loc[leaf.data_index]# 取出节点数据集 
    res = gini_min(data,min_sample_leaf)
    if not res:
        leaf.out = data.iloc[:,0].mode()[0] # 众数作为预测结果
        return None
    feature, split = res
    # gini_min函数返回值为二元组，(最佳分割特征,分割值)
    leaf.feature = feature
    leaf.split = split
    left = node(data[data[feature] <= split].index)
    right = node(data[data[feature] > split].index)
    return left, right


def gini_min(data, min_sample_leaf):
    # 根据基尼系数得到数据集上的最佳划分
    res = []# 三元组列表(gini,feature,split)
    S = data.shape[0]
    for feature in np.arange(1,data.shape[1]):
        #首先判断该列是否为onehot变量，避免onehot变量排序
        if is_one_hot(data,feature):
            bool_indexby0 = data.iloc[:,feature] == 0
            s1 = data.loc[bool_indexby0,data.columns[0]]
            S1 = s1.shape[0]
            S2 = S-S1
            if S1<min_sample_leaf or S2<min_sample_leaf:
                continue
            s2 = data.loc[not bool_indexby0,data.columns[0]]
            res.append(((S1*gini(s1) + S2*gini(s2))/S,feature,0))
        else:
            Gini_list = []# 二元组列表(gini,split)，存放每个特征的最优gini值和分割点
            s = data.iloc[:,[0,feature]]
            s = s.sort_values(s.columns[1])
            for i in np.arange(min_sample_leaf-1,S-min_sample_leaf):
                if s.iloc[i,1] == s.iloc[i+1,1]:
                    continue
                else:
                    S1 = i+1
                    S2 = S-S1
                    s1 = data.iloc[:(i+1),0]
                    s2 = data.iloc[(i+1):,0]
                    Gini_list.append(((S1*gini(s1) + S2*gini(s2))/S,s.iloc[i,1]))
            # 存在Gini_list为空的情况
            if Gini_list:
                Gini_min,split = min(Gini_list,key=lambda x:x[0])
                res.append((Gini_min,feature,split))
    # res也可能为空
    if res:
        _,feature,split = min(res,key=lambda x:x[0])
        return (data.columns[feature],split)
    else:
        return None
    
       
def gini(s):
    # 1-sum(pi^2)
    p = np.array(s.value_counts(True))
    return 1-np.sum(np.square(p))


def is_one_hot(data,feature):
    for i in range(data.shape[0]):
        v = data.iloc[i,feature]
        if v != 0 or v != 1:
            return False
    return True

def classifier(tree, sample):
    # 对于一个样本，从根节点开始
    # 根据节点的划分属性和分割值，寻找其子节点
    # 判断子节点是否为叶节点
    # 是，则得到输出，否则继续寻找子节点
    i = 0
    while True:
        node = tree[i]
        if node.out != None:
            return node.out
        if sample[node.feature] <= node.split:
            i = node.left
        else:
            i = node.right
    
def hit_rate(tree, test):
    # 逐一获取样本的分类结果
    # 对比label属性的数据，确定分类是否准确
    y = test.pop(test.columns[0])
    length = y.size
    y_p = pd.Series([0]*length,index=y.index)
    for i in range(length):
        x = test.iloc[i]
        y_p.iloc[i] = classifier(tree,x)
#    print(y_p)
    deta = y-y_p
    return deta[deta==0].size/length
    
if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    t1 = time.time()
    min_sample_leaf = 31
    tree = build_tree(train ,min_sample_leaf)
    t2 = time.time()
    score = hit_rate(tree, test)
    t3 = time.time()
    print('决策树的构建时间为：%f'%(t2-t1))
    print('测试集分类时间为：%f'%(t3-t2))
    print('分类正确率为：%f'%score)
    print('参数设置为min_sample_leaf：%d'%min_sample_leaf)
    










