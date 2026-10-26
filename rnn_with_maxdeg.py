import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import ast
import random
import numpy as np

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)

# DATASET
class GroebnerDatasetTuples(Dataset):
    def __init__(self, filepath, num_generators=6, num_vars=3, max_terms=10):
        self.data = []
        self.output = []
        self.num_generators = num_generators
        self.num_vars = num_vars
        self.max_terms = max_terms

        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            ordering_str, ideal_str, output_str = line.split("|")
            ordering = int(ordering_str.strip("()"))

            generators = []
            max_degrees = []

            for poly_str in ideal_str.split(";"):
                poly_str = "[" + poly_str + "]"
                monomials = ast.literal_eval(poly_str)

                g = torch.zeros(max_terms, num_vars + 1)
                max_deg = 0
                for i, (coeff, exps) in enumerate(monomials[:max_terms]):
                    g[i, 0] = coeff
                    g[i, 1:] = torch.tensor(exps)
                    max_deg = max(max_deg, sum(exps))  # compute max degree
                generators.append(g)
                max_degrees.append(max_deg)

            while len(generators) < num_generators:
                generators.append(torch.zeros(max_terms, num_vars + 1))
                max_degrees.append(0)

            generators = torch.stack(generators)
            max_degrees = torch.tensor(max_degrees, dtype=torch.float32)

            self.data.append((generators, torch.tensor(ordering), max_degrees))
            self.output.append(torch.tensor([float(output_str)], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        generators, ordering, max_degrees = self.data[idx]
        y = self.output[idx]
        return generators, ordering, max_degrees, y


# RNN and FFWD
class GroebnerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_generators=5):
        super(GroebnerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_generators = num_generators

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        # hidden_size * num_generators + 1 (ordering) + num_generators (max degrees)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size * num_generators + 1 + num_generators, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, generators, ordering, max_degrees):
        batch_size = generators.size(0)
        summaries = []

        for i in range(self.num_generators):
            g = generators[:, i, :, :]
            _, hidden_n = self.rnn(g)
            hidden_n = hidden_n.squeeze(0)
            summaries.append(hidden_n)

        all_summary = torch.cat(summaries, dim=1)
        ordering = ordering.view(batch_size, 1).float()
        max_degrees = max_degrees.view(batch_size, self.num_generators).float()

        total_input = torch.cat((all_summary, ordering, max_degrees), dim=1)
        return self.feedforward(total_input)


# Dataset
dataset = GroebnerDatasetTuples(
    filepath=r"../data/data_formatted222.txt",
    num_generators=5,
    num_vars=3,
    max_terms=10
)

# Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

# Model
model = GroebnerRNN(input_size=4, hidden_size=24, output_size=1, num_generators=5)
first_weight = list(model.parameters())[0][0][:5]
print("Seed check (Machine 1) - first weights:", first_weight.detach().numpy())

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training
epochs = 3000
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for generators, ordering, max_degrees, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(generators, ordering, max_degrees)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for generators, ordering, max_degrees, y in test_loader:
                y_pred = model(generators, ordering, max_degrees)
                test_loss += loss_fn(y_pred, y).item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} "
              f"| Test Loss: {test_loss / len(test_loader):.4f}")
