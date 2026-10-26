import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast
import random
import numpy as np

#seeds for reproducibility
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

# Reformatting the dataset so it is readable by the model
class GroebnerDatasetTuples(Dataset):
    def __init__(self, filepath=r"C:\Users\HP\PycharmProjects\PythonProject4\data100reformatted.txt",
                 num_generators=6, num_vars=3, max_terms=10):
        self.data = []
        self.output = []
        self.num_generators = num_generators
        self.num_vars = num_vars
        self.max_terms = max_terms

        with open(filepath, "r") as f:
            lines = f.read().splitlines()
# Ordering|Ideal|Output(s-poly divisions)
        for line in lines:
            ordering_str, ideal_str, output_str = line.split("|")

            # Ordering (stored as tuple like (2))
            ordering = int(ordering_str.strip().strip("()"))

            # semicolon separates polynomials (generators), commas separate monomials
            generators = []
            for poly_str in ideal_str.split(";"):
                poly_str = "[" + poly_str + "]"   # wrap in [] to turn in python list
                monomials = ast.literal_eval(poly_str)

                # Polynomial as tensor [max_terms, num_vars+1]
                g = torch.zeros(max_terms, num_vars + 1)
                for i, (coeff, exps) in enumerate(monomials[:max_terms]):
                    g[i, 0] = coeff
                    g[i, 1:] = torch.tensor(exps)
                generators.append(g)

            # Pad missing generators with zeros
            while len(generators) < num_generators:
                generators.append(torch.zeros(max_terms, num_vars + 1))

            # Stack into one tensor: [num_generators, max_terms, num_vars+1]
            generators = torch.stack(generators)

            self.data.append((generators, torch.tensor(ordering)))
            self.output.append(torch.tensor([float(output_str)], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        generators, ordering = self.data[idx]
        y = self.output[idx]
        return generators, ordering, y


# RNN(using GRU) and FEEDFORWARD
class GroebnerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_generators=5):
        super(GroebnerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_generators = num_generators

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True) #no particular reason why I am using GRU, it's just a popular one
                                                                     #input_size is just num_var + 1 to accommodate the coeff

        # Summary: hidden_size * num_generators + 1 (+1 to accommodate ordering)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size * num_generators + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, generators, ordering):
        """
        B= batch size
        generators: [B, num_generators, max_terms, input_size]
        ordering:   [B] # one ordering value per example in the batch
        """
        batch_size = generators.size(0)
        summaries = []

        for i in range(self.num_generators):
            g = generators[:, i, :, :]               # [B, num_gen, max_terms, input_size]
            _, hidden_n = self.rnn(g)                # hidden_n: [1, B, hidden_size] (_ is the sequence of hidden state after every monomial, hidden_n is the final hidden state (OF THE ith GENERATOR)
            hidden_n = hidden_n.squeeze(0)           # [B, hidden_size], squeeze to get rid of the extra dimension
            summaries.append(hidden_n)

        all_summary = torch.cat(summaries, dim=1)    # [B, hidden_size * num_generators]
        ordering = ordering.view(batch_size, 1).float()

        total_input = torch.cat((all_summary, ordering), dim=1)
        return self.feedforward(total_input)


from torch.utils.data import random_split

#dataset
dataset = GroebnerDatasetTuples(
    filepath=r"../data/data100reformatted.txt",
    num_generators=5,
    num_vars=3,
    max_terms=10
)

# Split 80/20 into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

#Model
model = GroebnerRNN(input_size=4, hidden_size=24, output_size=1, num_generators=5)
first_weight = list(model.parameters())[0][0][:5]  #just to see if the random seed is working
print("Seed check (Machine 1) - first weights:", first_weight.detach().numpy())
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)  #Using Adam but can use SGD to yield similar results
loss_fn = nn.MSELoss() #Mean Squared Error

#Training (just the usual training algorithm)
epochs = 5000 #number of times the model runs the dataset
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for generators, ordering, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(generators, ordering)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

#print out the train and test loss
    if (epoch + 1) % 50 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for generators, ordering, y in test_loader:
                y_pred = model(generators, ordering)
                test_loss += loss_fn(y_pred, y).item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} "
              f"| Test/Validation Loss: {test_loss / len(test_loader):.4f}")

# ===== SAVE TRAINED MODEL =====
save_path = r"C:\Users\HP\PycharmProjects\PythonProject4\saved_models\groebner_model.pth"

torch.save(model.state_dict(), save_path)
print("Model saved to:", save_path)
