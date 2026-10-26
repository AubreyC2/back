import sklearn
from sklearn.datasets import make_circles
from sklearn.externals.array_api_compat.torch import squeeze
from torch import nn
import torch



#1000 samples

n_samples= 100

#create circles

X, y = make_circles(n_samples,
                    noise= 0.03,
                    random_state=42)

import pandas as pd
circles = pd.DataFrame({"x1": X[:, 0],
                        "x2": X[:, 1],
                        "label":y})


import matplotlib.pyplot as plt

plt.scatter(x= X[:,0],
            y= X[:, 1],
            c=y,
            cmap = plt.cm.RdYlBu)


## check input output shapes

print(X.shape, y.shape)
X_sample = X[0]
y_sample = y[0]

# turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)




#split data into training and test set.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size= 0.2,
                                                    random_state=42,
                                                    )

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1= nn.Linear(in_features= 2, out_features=10)
        self.layer_2= nn.Linear(in_features= 10, out_features=10)
        self.layer_3= nn.Linear(in_features=10, out_features=1)
        self.relu= nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        z = self.layer_1(x)
        z = self.relu(z)
        z = self.layer_2(z)
        z = self.layer_3(z)
        return z


model_0= CircleModelV0().to(device)
print(model_0)

'''
model_0= nn.Sequential(
    nn.Linear(in_features=2, out_features=15),
    nn.ReLU,
    nn.Linear(in_features=15, out_features=15),
    nn.Linear(in_features=15, out_features=15),
    nn.Linear(in_features=15, out_features=15),
    nn.Linear(in_features=15, out_features=15),
    nn.Linear(in_features=15, out_features=1)
).to(device)
'''
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))

'''
print(f'len of prediction:{len(untrained_preds)}, Shape: {untrained_preds.shape}')
print(f'len of test samples:{len(X_test)}, Shape: {X_test.shape}')
print(f'\n first 10 preds: {untrained_preds[:10]}')
print(f'\n Desired:{y_test[:10]}')
'''

loss_fn= nn.BCEWithLogitsLoss()
opt= torch.optim.SGD(params= model_0.parameters(),
                     lr=0.1
                     )
with torch.inference_mode():
    y_logits= model_0(X_test.to(device))[:5]
print(f'Logists: {y_logits}')
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc= (correct/len(y_pred)) *100
    return acc

y_pred_probs= torch.sigmoid(y_logits)
'''
print(y_pred_probs)
print(torch.round(y_pred_probs))
'''
#just unpacks what y_preds is doing, take logits(outputs of trianing set) run it
# run it sigmoid and finally round it
y_pred_lables= torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
y_preds= torch.round(y_pred_probs)
'''
print(torch.eq(y_preds, y_pred_lables))
'''

torch.manual_seed(42)
epoch= 500
X_train, y_train= X_train.to(device), y_train.to(device)
X_test, y_test  = X_test.to(device), y_test.to(device)

for epoch in range(epoch):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred= torch.round(torch.sigmoid(y_logits))
    loss= loss_fn(y_logits,
                  y_train)
    acc= accuracy_fn(y_true=y_train,
                     y_pred= y_pred)
    opt.zero_grad()

    loss.backward()

    opt.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits= model_0(X_test).squeeze()
        test_pred= torch.round(torch.sigmoid(test_logits))

        test_loss= loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true =y_test,
                               y_pred= test_pred)

    '''if epoch % 100 ==0:
        print(f'Epoch:{epoch} | Loss:{loss:.5f} | Acc:{acc:.2f}')'''

y_predidction= torch.round(torch.sigmoid(model_0(X_test))).squeeze()
print(y_predidction)
print(y_test)
