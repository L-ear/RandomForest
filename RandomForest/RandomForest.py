import DecisionTree as DT
import numpy as np
import pandas as pd
import time

def RandomForest(train, n_trees, min_sample_leaf, ip, jp):
    forest = []# 存放训练得到的所有的决策树
    fn = int(jp*(train.shape[1]-1))
    for n in range(n_trees):
        t1 = time.time()
        sf = np.random.choice(
                np.arange(1,train.shape[1]),
                fn,
                replace=False)
        sf = np.append(0,sf) # 保证label在第一列
        train_n = train.iloc[:,sf]
        p = np.random.random_sample()*(1-ip)+ip
        train_n = train_n.loc[
                np.random.choice(train_n.index,
                                 int(p*train_n.index.size),
                                 replace=False)]
        #train_n为随机选出的第n棵树的训练集
        forest.append(DT.build_tree(train_n, min_sample_leaf))
        t2 = time.time()
        print('构建第%d棵树的时间为%f'%(n,t2-t1))
    return forest   



def hit_rate(forest, test):
    # 取出测试集中非标签属性的数据
    # 逐一获取样本的分类结果
    # 对比label属性的数据，确定分类是否准确
    y = test.pop(test.columns[0])
    length = y.size
    y_p = pd.Series([0]*length,index=y.index)
    n_trees = len(forest)
    res = [0]*n_trees # 存放每棵树的预测结果
    for i in range(length):
        x = test.iloc[i]
        for t in range(n_trees):
            res[t] = DT.classifier(forest[t],x)
        y_p.iloc[i] = max(res,key=res.count)
    deta = y-y_p
    return deta[deta==0].size/length
    
if __name__ == "__main__":
    ip = 0.85
    jp = 0.7
    n_trees = 60
    min_sample_leaf = 5
    # 上面是待调参数
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    forest = RandomForest(train,n_trees,min_sample_leaf,ip,jp)
    t1 = time.time()
    score = hit_rate(forest,test)
    t2 = time.time()
    print('测试集分类时间为%f'%(t2-t1))
    print('参数设置为n_trees=%d,min_sample_leaf=%d,ip=%f,jp=%f'%(n_trees,min_sample_leaf,ip,jp))
    print('分类正确率为%f'%score)


