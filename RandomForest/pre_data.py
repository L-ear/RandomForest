import pandas as pd
train = pd.read_csv("original_data/train.csv")
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
train.Age.fillna(train.Age.mode()[0],inplace=True)
train.dropna(inplace=True)
train = pd.get_dummies(train)
train[:600].to_csv("data/train.csv",index=False)# 不保存行索引
train[600:].to_csv("data/test.csv",index=False)
