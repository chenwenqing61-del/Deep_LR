import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer #缺失值处理器
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from torch.utils.data import TensorDataset,DataLoader

def create_dataset():
    #对于数值型特征使用均值填充缺失值，后标准化；类别型特征使用字符串“NaN”填充缺失值，再编码
    data =pd.read_csv("Data/train.csv")
    data.drop(["Id"],axis=1,inplace=True)
    X = data.drop(["SalePrice"],axis=1)
    Y = data["SalePrice"]

    numerical_features = X.select_dtypes(exclude="object").columns
    categorical_features = X.select_dtypes(include="object").columns
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    numerical_transfomer = Pipeline(
        steps = [
            ("fillna",SimpleImputer(strategy="mean")), #fillna只是这个步骤的名字，使用SimpleImputer对缺失值进行处理，用均值进行填充
            ("std",StandardScaler()),#进行标准化
        ]
    )

    categorical_transfomer = Pipeline(
        steps = [
            ("fillna",SimpleImputer(strategy="constant",fill_value="NaN")),#使用常数NaN对缺失值进行填充
            ("onehot",OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numerical_transfomer,numerical_features),
            ("cat",categorical_transfomer,categorical_features)
        ]
    )

    #进行特征预处理
    x_train = pd.DataFrame(preprocessor.fit_transform(x_train).toarray(),
                           columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(preprocessor.transform(x_test).toarray(),
                          columns=preprocessor.get_feature_names_out())

    train_dataset = TensorDataset(torch.tensor(x_train.values).float(),torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(),torch.tensor(y_test.values).float())

    return train_dataset,test_dataset,x_train.shape[1]
def log_rmse(pred,target):
    mse = nn.MSELoss()
    pred.squeeze_()
    pred = torch.clamp(pred,1,float("inf"))
    return torch.sqrt(mse(torch.log(pred),torch.log(target)))

def train(model,train_dataset,test_dataset,lr,epoch_num,batch_size,device):
    def init_weight(layer):
        if type(layer)==nn.Linear:
            nn.init.xavier_normal_(layer.weight)

    model.apply(init_weight)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    train_loss_list=[]
    test_loss_list=[]

    for epoch in range(epoch_num):
        model.train()
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        train_loss_accumulation=0
        for batch_count,(X,y) in enumerate(train_loader):
            #前向传播
            X,y=X.to(device),y.to(device)
            output = model(X)

            loss_value = log_rmse(output,y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loss_accumulation+=loss_value.item()
            print(f"\repoch:{epoch:0>3}[{'='*(int((batch_count+1)/len(train_loader)*50)):<50}]",end='')
        this_train_loss = train_loss_accumulation/len(train_loader)
        train_loss_list.append(this_train_loss)

        model.eval()
        test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
        test_loss_accumulation=0
        with torch.no_grad():
            for X,y in test_loader:
                X,y = X.to(device),y.to(device)
                pred = model(X)
                loss_value = log_rmse(pred,y)
                test_loss_accumulation+=loss_value.item()
            this_test_loss =test_loss_accumulation/len(test_loader)
            test_loss_list.append(this_test_loss)
            print(f"train_loss:{this_train_loss:.6f}.test_loss:{this_test_loss:.6f}")
    return train_loss_list,test_loss_list

train_dataset,test_dataset,feature_num=create_dataset()

#构建模型
model = nn.Sequential(
    nn.Linear(feature_num,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128,1)
)

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
train_loss_list,test_loss_list = train(model,train_dataset,test_dataset,0.1,200,64,device)
plt.plot(train_loss_list,"r-",label="train_loss",linewidth=3)
plt.plot(test_loss_list,"k--",label="test_loss",linewidth=2)
plt.legend()
plt.show()

