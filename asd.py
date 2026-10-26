import matplotlib.pyplot as plt
import torch
from torch import nn, is_deterministic_algorithms_warn_only_enabled

#desired
w= 2
b= 3
x= torch.arange(0, 1 ,0.02)
y= w*x + b
print(y[:10])

#linear regression model
train_split = int(.8 *len(x))
x_train= x[:train_split]
y_train= y[:train_split]
x_q = x[train_split:]
y_q = y[train_split:]
class LinearRegressionModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.weights= nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype= torch.float))
        self.bias= nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  self.weights * x +self.bias
torch.manual_seed(42)
model=LinearRegressionModel()
model.forward(torch.tensor([1]))
y_preds= model(x_train)


with torch.inference_mode():
    y_preds = model(x_train)

print(y_preds[:10])

#lost fn
lost_fn= nn.L1Loss()

#opt
opt= torch.optim.SGD(params= model.parameters(),
                     lr=0.01)

#training loop
print(f'pre{model(x_q)}')
epochs= 1000

epoch_count = []
loss_values = []
test_loss_values= []
for epoch in range(epochs):
    model.train() #turns on grad
    y_preds = model(x_train) #forward pass
    lost =lost_fn(y_preds, y_train)# loss
    opt.zero_grad()
    lost.backward() #perform backpropogation
    opt.step() #opt cumulate theorefore need to 0 it before
    model.eval() #turns off grad
    with torch.inference_mode():
        test_pred = model(x_q)
        test_loss = lost_fn(test_pred, y_q)
    if epoch % 100 == 0:
        epoch_count.append(epoch)
        loss_values.append(lost)
        test_loss_values.append(test_loss)
        print(f'Epoch: {epoch} | Test: {lost} | Test loss: {test_loss}')
        print(model.state_dict())





print(w, b)
with torch.inference_mode():
    y_preds_new= model(x_q)


print(f' after training: {y_preds_new}')
print(x_q)

import numpy as np

np.array(torch.tensor(loss_values).cpu().numpy())
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label= "train lost"),
plt.plot(epoch_count, test_loss_values, label= "test lost")
plt.title("training and test lost curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();

from pathlib import Path

#create directory
MODEL_PATH= Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#create save path
MODEL_NAME = '01_pytorch_workflow_model.pth'
MODEL_SAVE_PATH= MODEL_PATH / MODEL_NAME

print(MODEL_SAVE_PATH)

#save the model
print(f'Saving model to :{MODEL_SAVE_PATH}')
torch.save(model.state_dict(),MODEL_SAVE_PATH)

loaded_model= LinearRegressionModel()