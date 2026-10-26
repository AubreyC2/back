import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ast
import random
import numpy as np

random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
# DATASET
class GroebnerDatasetTuples(Dataset):
    def __init__(self, filepath=r"C:\Users\HP\PycharmProjects\PythonProject4\divisions_with_two_features_formatted.txt",
                 num_generators=6, num_vars=3, max_terms=10):
        self.data = []
        self.output = []
        self.num_generators = num_generators
        self.num_vars = num_vars
        self.max_terms = max_terms

        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            parts = line.split("|")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 fields per line but got {len(parts)}: {line}")

            ordering_str, ideal_str, max_terms_str, output_str = parts

            # Ordering
            ordering = int(ordering_str.strip().strip("()"))

            # max_terms comes from the line now
            max_terms = int(max_terms_str.strip())
            self.max_terms = max_terms  # optional: update dataset attribute

            generators = []
            for poly_str in ideal_str.split(";"):
                poly_str = "[" + poly_str + "]"
                monomials = ast.literal_eval(poly_str)

                g = torch.zeros(max_terms, self.num_vars + 1)
                for i, (coeff, exps) in enumerate(monomials[:max_terms]):
                    g[i, 0] = coeff
                    g[i, 1:] = torch.tensor(exps)
                generators.append(g)

            while len(generators) < self.num_generators:
                generators.append(torch.zeros(max_terms, self.num_vars + 1))

            generators = torch.stack(generators)

            self.data.append((generators, torch.tensor(ordering)))
            self.output.append(torch.tensor([float(output_str)], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        generators, ordering = self.data[idx]
        y = self.output[idx]
        return generators, ordering, y


# RNN and FFWD
class GroebnerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_generators=5):
        super(GroebnerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_generators = num_generators

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        # Summary: hidden_size * num_generators + 1 (for ordering)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size * num_generators + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, generators, ordering):
        """
        generators: [B, num_generators, max_terms, input_size]
        ordering:   [B]
        """
        batch_size = generators.size(0)
        summaries = []

        for i in range(self.num_generators):
            g = generators[:, i, :, :]               # [B, num_gen, max_terms, input_size]
            _, hidden_n = self.rnn(g)                # hidden_n: [1, B, hidden_size]
            hidden_n = hidden_n.squeeze(0)           # [B, hidden_size]
            summaries.append(hidden_n)

        all_summary = torch.cat(summaries, dim=1)    # [B, hidden_size * num_generators]
        ordering = ordering.view(batch_size, 1).float()

        total_input = torch.cat((all_summary, ordering), dim=1)
        return self.feedforward(total_input)


from torch.utils.data import random_split

#dataset
dataset = GroebnerDatasetTuples(
    filepath=r"../data/divisions_with_two_features_formatted.txt",
    num_generators=5,
    num_vars=3,
    max_terms=10
)

# Split 80/20 into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

#Model
model = GroebnerRNN(input_size=4, hidden_size=24, output_size=1, num_generators=5)
first_weight = list(model.parameters())[0][0][:5]
print("Seed check (Machine 1) - first weights:", first_weight.detach().numpy())
optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
loss_fn = nn.MSELoss()

#Training
epochs = 3000
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

    if (epoch + 1) % 100 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for generators, ordering, y in test_loader:
                y_pred = model(generators, ordering)
                test_loss += loss_fn(y_pred, y).item()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} "
              f"| Test Loss: {test_loss / len(test_loader):.4f}")
