
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

with open('../data/data.txt', 'r') as f:
    data = []
    lines = f.readlines()
    for i in range(len(lines)):
        data.append([])
        ctr = 0
        for entry in lines[i].split():
            if ctr == 36:
                data[i].append(float(entry))
            else:
                data[i].append(int(entry))
            ctr += 1

print(len(lines))

device = "cuda" if torch.cuda.is_available() else "cpu"

x = [line[:36] for line in data]
y = [line[36:] for line in data]

#x= StandardScaler().fit_transform(x)




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test= scaler_x.fit_transform(x_test)


model = nn.Sequential(
    nn.Linear(in_features=36, out_features=64),
    nn.Tanh(),
    nn.Linear(in_features=64, out_features=64),
    nn.Tanh(),
    nn.Linear(in_features=64, out_features=64),
    nn.Tanh(),
    nn.Linear(in_features=64, out_features=64),
    nn.Tanh(),
    nn.Linear(in_features=64, out_features=64),
    nn.Tanh(),
    nn.Linear(in_features=64, out_features=1),
).to(device)

loss_fn=nn.MSELoss() #gonna use L2 norm for this experiment.
opt = optim.SGD(model.parameters(), lr=0.01)

#training


x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

x_train, y_train= x_train.to(device), y_train.to(device)
x_test, y_test  = x_test.to(device), y_test.to(device)
epoch=1000
for epoch in range(epoch):

    model.train()
    y_pred= model(x_train)
    loss= loss_fn(y_pred, y_train)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    opt.step()

    model.eval()
    with torch.inference_mode():
        test_pred= model(x_test)
        test_loss= loss_fn(test_pred,y_test)
    if (epoch+1) % 100  == 0:
        print(f'Epoch:{epoch+1} | Loss: {loss} | Test Loss: {test_loss}')


'''print(model(x_test[:10]))
print(model(x_test[:10])- y_test[:10])
print(loss_fn(model(x_test[:10]),y_test[:10]))
print(y_test[:10])'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linreg = LinearRegression()
linreg.fit(x_train.cpu(), y_train.cpu())
y_pred_lr = linreg.predict(x_test.cpu())
print("Linear Regression MSE:", mean_squared_error(y_test.cpu(), y_pred_lr))

