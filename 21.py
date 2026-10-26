import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import ast
import numpy as np
import random

# -------------------------------
# Seeds for reproducibility
# -------------------------------
random.seed(12)
np.random.seed(12)
torch.manual_seed(12)

# -------------------------------
# Dataset
# -------------------------------
class GroebnerDatasetTuples(Dataset):
    def __init__(self, filepath, num_generators=5, num_vars=3, max_terms=10):
        self.num_generators = num_generators
        self.num_vars = num_vars
        self.max_terms = max_terms

        coeffs_all = []
        exps_all = []
        outputs_all = []

        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        self.lines = lines
        for line in lines:
            ordering_str, ideal_str, output_str = line.split("|")
            outputs_all.append(float(output_str))

            for poly_str in ideal_str.split(";"):
                poly_str = "[" + poly_str + "]"
                monomials = ast.literal_eval(poly_str)
                for coeff, exps in monomials:
                    coeffs_all.append(coeff)
                    exps_all.extend(exps)

        # Dynamic normalization factors
        self.max_coeff = max(abs(c) for c in coeffs_all) if coeffs_all else 1.0
        self.max_exp = max(abs(e) for e in exps_all) if exps_all else 1.0
        self.max_output = max(outputs_all) if outputs_all else 1.0

        # Preprocess dataset
        self.data = []
        self.output = []
        for line in lines:
            ordering_str, ideal_str, output_str = line.split("|")
            ordering = int(ordering_str.strip().strip("()"))

            generators = []
            for poly_str in ideal_str.split(";"):
                poly_str = "[" + poly_str + "]"
                monomials = ast.literal_eval(poly_str)

                g = torch.zeros(max_terms, num_vars + 1)
                for i, (coeff, exps) in enumerate(monomials[:max_terms]):
                    coeff_norm = coeff / self.max_coeff
                    exps_norm = [e / self.max_exp for e in exps]

                    g[i, 0] = coeff_norm
                    g[i, 1:] = torch.tensor(exps_norm)

                generators.append(g)

            while len(generators) < num_generators:
                generators.append(torch.zeros(max_terms, num_vars + 1))

            generators = torch.stack(generators)
            self.data.append((generators, torch.tensor(ordering)))
            self.output.append(torch.tensor([float(output_str)/self.max_output], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        generators, ordering = self.data[idx]
        y = self.output[idx]
        return generators, ordering, y

# -------------------------------
# Model
# -------------------------------
class GroebnerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_generators=5, dropout=0.3):
        super(GroebnerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_generators = num_generators

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size * num_generators + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, generators, ordering):
        batch_size = generators.size(0)
        summaries = []

        for i in range(self.num_generators):
            g = generators[:, i, :, :]
            _, hidden_n = self.rnn(g)
            hidden_n = hidden_n.squeeze(0)
            summaries.append(hidden_n)

        all_summary = torch.cat(summaries, dim=1)
        ordering = ordering.view(batch_size, 1).float()
        total_input = torch.cat((all_summary, ordering), dim=1)
        return self.feedforward(total_input)

# -------------------------------
# Load Dataset
# -------------------------------
dataset = GroebnerDatasetTuples(
    filepath=r"/data/data100reformatted.txt",
    num_generators=5,
    num_vars=3,
    max_terms=10
)

# Split 80/20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=0)

# -------------------------------
# Initialize model
# -------------------------------
model = GroebnerRNN(input_size=4, hidden_size=24, output_size=1, num_generators=5, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

# -------------------------------
# Training loop (no early stopping)
# -------------------------------
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

    # validation
    model.eval()
    test_loss = 0
    test_loss_original = 0
    with torch.no_grad():
        for generators, ordering, y in test_loader:
            y_pred = model(generators, ordering)
            test_loss += loss_fn(y_pred, y).item()

            y_pred_orig = y_pred * dataset.max_output
            y_orig = y * dataset.max_output
            test_loss_original += nn.MSELoss()(y_pred_orig, y_orig).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)
    avg_test_loss_orig = test_loss_original / len(test_loader)

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (norm): {avg_train_loss:.6f} | "
              f"Test Loss (norm): {avg_test_loss:.6f} | Test Loss (orig): {avg_test_loss_orig:.2f}")

# -------------------------------
# Predict on test set (original scale)
# -------------------------------
model.eval()
preds = []
trues = []
with torch.no_grad():
    for generators, ordering, y in test_loader:
        y_pred = model(generators, ordering)
        y_pred_orig = y_pred * dataset.max_output
        y_orig = y * dataset.max_output
        preds.extend(y_pred_orig.squeeze().tolist())
        trues.extend(y_orig.squeeze().tolist())

print("\nSample predictions on original scale:")
for i in range(min(10, len(preds))):
    print(f"Predicted: {preds[i]:.2f} | True: {trues[i]:.2f}")
