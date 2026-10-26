import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
x_train= (torch.rand(1000,37))
y_train= (torch.rand(1000,1))
x_test= (torch.rand(1000,37))
y_test= (torch.rand(1000,1))

x_train, y_train= x_train.to(device), y_train.to(device)
x_test, y_test  = x_test.to(device), y_test.to(device)

model = nn.Sequential(
    nn.Linear(in_features=37, out_features=64),
    nn.Linear(in_features=64, out_features=32),
    nn.Linear(in_features=32, out_features=1),
).to(device)

loss_fn=nn.MSELoss() #gonna use L2 norm for this experiment.
opt= optim.SGD((model.parameters()), lr= 0.1)

#training
model.train()

epoch=1000
for epoch in range(epoch):
    model.train()
    y_pred= model(x_train)
    loss= loss_fn(y_pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.eval()
    with torch.inference_mode():
        test_pred= model(x_test)
        test_loss= loss_fn(test_pred,y_test)
    if (epoch) % 100 == 0:
        print(f'Epoch:{epoch} |Loss: {loss}')

