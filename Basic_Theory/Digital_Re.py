import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinManScaler
from common.functions import sigmoid,softmax

def get_data():
    data = pd.read_csv("../data/train.csv")#pandas库处理csv文件，读取数据
    X = data.drop("label",axis=1) #drop函数删除指定的行或列，axis表示删除那一个维度，0表示行，1表示列
    y = data["label"] #拿每一个数据的标签
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)#按照0.3的比例划分出来预测集合

    preprocessor = MinMaxScaler()#将数据压缩到0~1之间
    x_train = preprocessor.fit_transform(x_train) #fit_transform函数用于拟合数据并进行转换
    x_test = preprocessor.transform(x_test)#transform使用前面已经拟合了的数据进行转换 即X_max X_min 等参数已经确定下来了
    return x_test,x_train

def init_network():
    network = joblib.load("../data/nn_sample)")  #用于保存numpy数组，保存神经网络的权重和编制参数，以及已经训练好的模型

    return network

def predict(network,x):
    w1,w2,w3 = network["W1"],network["W2"],network["W3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]

    a1 = np.dot(x,w1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3)+b3
    y = softmax(a3)
    return y

x,t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)#进行一次前线传播，得到预测结果
    p = np.argmax(y_batch,axis=1)#得到预测的结果，且拿到最高分数的标签，
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))
