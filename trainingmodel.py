

import torch
import torch.nn
import matplotlib.pyplot as plt

from asd import y_preds_new

w= 2
b= 3
x= torch.arange(0, 1 ,0.02)
y= w*x + b
print(y[:10])

train_split = int(.8 *len(x))
x_t= x[:train_split]
y_t= y[:train_split]
x_q = x[train_split:]
y_q = y[train_split:]

print(len(x_t), len(y_t), len(x_q) , len(y_q))

def plot_predictions(train_data= x_t,
                     train_lable= y_t,
                     test_data= x_q,
                     test_lables = y_q,
                     predictions= y_preds_new):
    plt.figure(figsize=(12,7))
    plt.scatter(train_data, train_lable, c="r", s=15, label="Training data")
    plt.scatter(test_data, test_lables, c="g", s=15, label="testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="b", s=15, label="Predictions")
    plt.legend(prop={"size": 20});

plot_predictions()
plt.show()

