import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import ast, random, numpy as np

random.seed(44)
np.random.seed(44)
torch.manual_seed(44)

# ===============================
# Dataset: Only num_div
# ===============================
def parse_line(line, num_generators=5, num_vars=3, max_terms=10):
    ordering_str, ideal_str, max_terms_str, num_div_str, output_str = line.split("|")
    num_div_val = int(num_div_str)
    output_val = float(output_str)

    # Parse ideal into tensor
    generators = []
    for poly_str in ideal_str.split(";"):
        poly_str = "[" + poly_str + "]"
        monomials = ast.literal_eval(poly_str)
        g = torch.zeros(max_terms, num_vars + 1)
        for i, (coeff, exps) in enumerate(monomials[:max_terms]):
            g[i, 0] = coeff
            g[i, 1:] = torch.tensor(exps)
        generators.append(g)
    while len(generators) < num_generators:
        generators.append(torch.zeros(max_terms, num_vars + 1))
    generators = torch.stack(generators)
    return generators, num_div_val, output_val


class DatasetNumDiv(Dataset):
    def __init__(self, filepath, num_generators=5, num_vars=3, max_terms=10):
        self.data, self.labels = [], []
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            generators, num_div, output_val = parse_line(line, num_generators, num_vars, max_terms)
            extra = torch.tensor([num_div], dtype=torch.float32)
            self.data.append((generators, extra))
            self.labels.append(torch.tensor([output_val], dtype=torch.float32))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return (*self.data[idx], self.labels[idx])


# ===============================
# Model
# ===============================
class GroebnerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_generators=5, extra_size=1):
        super().__init__()
        self.num_generators = num_generators
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size * num_generators + extra_size, 128),
            nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, generators, extra):
        summaries = []
        for i in range(self.num_generators):
            g = generators[:, i, :, :]
            _, hidden = self.rnn(g)
            summaries.append(hidden.squeeze(0))
        all_summary = torch.cat(summaries, dim=1)
        total_input = torch.cat((all_summary, extra), dim=1)
        return self.ff(total_input)


# ===============================
# Train Example
# ===============================
filepath = r"../data/divisions_with_three_features_formatted.txt"
dataset = DatasetNumDiv(filepath, num_generators=5)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=30, shuffle=True)
test_loader = DataLoader(test_set, batch_size=30, shuffle=False)

model = GroebnerRNN(input_size=4, hidden_size=24, num_generators=5, extra_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

for epoch in range(100):
    model.train()
    total_loss = 0
    for g, extra, y in train_loader:
        optimizer.zero_grad()
        y_pred = model(g, extra)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for g, extra, y in test_loader:
                y_pred = model(g, extra)
                test_loss += loss_fn(y_pred, y).item()
        print(f"Epoch {epoch+1} | Train {total_loss/len(train_loader):.4f} | Test {test_loss/len(test_loader):.4f}")
