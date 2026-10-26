import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

num_classes = 4
num_features = 2
random_seed = 42

#makin the blobs

x_blob, y_blob = make_blobs(n_samples= 1000,
                            n_features= num_features,
                            centers= num_classes,
                            cluster_std= 1.5,
                            random_state=random_seed)
# turn data into tensors

x_blob =  torch.from_numpy(x_blob).type((torch.float))
y_blob =  torch.from_numpy(y_blob).type((torch.LongTensor))

#splitting data
x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=random_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#building a multiclass model
class BlobModel(nn.Module):
    def __init__(self, input_feature, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack= nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features)).to(device)
    def forward(self,x):
        return  self.linear_layer_stack(x)
model_blob= BlobModel(input_feature=2,
                      output_features=4,
                      hidden_units=8).to(device)

print(model_blob)

#create loss fn

loss_fn= nn.CrossEntropyLoss()
opt= torch.optim.SGD(params=model_blob.parameters(),
                     lr= 0.1)

y_logits= model_blob(x_blob_train.to(device)) #the raw outputs (prob,prob,prob,prob)

y_pred_probs= torch.softmax(y_logits, dim=1) #softmaxes the thingy

y_preds= torch.argmax(y_pred_probs, dim=1) # give us which entry has the highest prob

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc= (correct/len(y_pred)) *100
    return acc
#training
epochs = 100

x_blob_train, y_blob_train= x_blob_train.to(device), y_blob_train.to(device)
x_blob_test, y_blob_test=  x_blob_test.to(device), y_blob_test.to(device)
print(y_blob_train.size())
for epoch in range(epochs):
    model_blob.train()
    y_logits= model_blob(x_blob_train)

    y_pred= torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss= loss_fn(y_logits,
                  y_blob_train)

    acc= accuracy_fn(y_true=y_blob_train,
                     y_pred= y_pred)
    opt.zero_grad()

    loss.backward()

    opt.step()

    model_blob.eval()
    with torch.inference_mode():
        test_logits=model_blob(x_blob_test)
        test_pred=torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss= loss_fn(test_logits,y_blob_test)
        test_acc= accuracy_fn(y_blob_test,test_pred)

  '''  if epoch % 10 ==0:
        print(f'Epoch:{epoch}|Loss:{loss:.4f}|Acc{acc}|Test Loss:{test_loss}|Test Accuracy:{test_acc}')'''

  model_blob.eval()
  with