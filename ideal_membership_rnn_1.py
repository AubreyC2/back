# ============================================================
# Stable Ideal Membership RNN
# Reads: ideal | polynomial | label
# Predicts: 1 (in ideal), 0 (not in ideal)
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import ast
import random
import numpy as np

# ============================================================
# ====================== USER SETTINGS =======================
# ============================================================

DATA_PATH = "C:/Users/HP/PycharmProjects/PythonProject4/data/ideal_membership_data_5.txt"

NUM_GENERATORS = 5
NUM_VARS = 5
MAX_TERMS = 15

HIDDEN_SIZE = 16        # smaller to avoid overfitting
BATCH_SIZE = 32
EPOCHS = 5000            # can manually change
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
DROPOUT_RATE = 0.2

SEED = 12

# ============================================================
# ====================== SEED CONTROL ========================
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# ====================== DATASET =============================
# ============================================================

class IdealMembershipDataset(Dataset):
    def __init__(self, filepath):

        self.samples = []

        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        ones, zeros = 0, 0

        for line in lines:
            ideal_str, poly_str, label_str = line.split("|")

            # ---------- Parse Ideal ----------
            generators = []
            for poly in ideal_str.split(";"):
                poly = "[" + poly + "]"
                monomials = ast.literal_eval(poly)

                g = torch.zeros(MAX_TERMS, NUM_VARS + 1)
                for i, (coeff, exps) in enumerate(monomials[:MAX_TERMS]):
                    g[i, 0] = coeff
                    g[i, 1:] = torch.tensor(exps)
                generators.append(g)

            while len(generators) < NUM_GENERATORS:
                generators.append(torch.zeros(MAX_TERMS, NUM_VARS + 1))

            generators = torch.stack(generators)

            # ---------- Parse Query Polynomial ----------
            poly = "[" + poly_str + "]"
            monomials = ast.literal_eval(poly)

            query = torch.zeros(MAX_TERMS, NUM_VARS + 1)
            for i, (coeff, exps) in enumerate(monomials[:MAX_TERMS]):
                query[i, 0] = coeff
                query[i, 1:] = torch.tensor(exps)

            label = torch.tensor([float(label_str)], dtype=torch.float32)
            if label.item() == 1.0:
                ones += 1
            else:
                zeros += 1

            self.samples.append((generators, query, label))

        # Compute pos_weight for imbalanced dataset
        self.pos_weight = torch.tensor([zeros / max(ones, 1)])

        print(f"Dataset loaded: {len(self.samples)} samples | 1s: {ones}, 0s: {zeros} | pos_weight: {self.pos_weight.item():.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ============================================================
# ====================== MODEL ===============================
# ============================================================

class IdealMembershipRNN(nn.Module):
    def __init__(self):
        super(IdealMembershipRNN, self).__init__()

        input_size = NUM_VARS + 1

        # Shared GRU for generators
        self.gen_rnn = nn.GRU(input_size, HIDDEN_SIZE, batch_first=True)

        # Separate GRU for query polynomial
        self.query_rnn = nn.GRU(input_size, HIDDEN_SIZE, batch_first=True)

        # MLP with dropout
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * (NUM_GENERATORS + 1), 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1)
        )

    def forward(self, generators, query):
        batch_size = generators.size(0)

        # ---- Encode Generators ----
        gen_summaries = []
        for i in range(NUM_GENERATORS):
            g = generators[:, i, :, :]
            _, h = self.gen_rnn(g)
            gen_summaries.append(h.squeeze(0))

        gen_summary = torch.cat(gen_summaries, dim=1)

        # ---- Encode Query Polynomial ----
        _, hq = self.query_rnn(query)
        query_summary = hq.squeeze(0)

        # ---- Combine ----
        combined = torch.cat((gen_summary, query_summary), dim=1)

        return self.mlp(combined)

# ============================================================
# ====================== TRAINING ============================
# ============================================================

dataset = IdealMembershipDataset(DATA_PATH)

train_size = int(TRAIN_SPLIT * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = IdealMembershipRNN()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)

print("Model initialized.")

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for generators, query, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(generators, query)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for generators, query, labels in test_loader:
            outputs = model(generators, query)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            predictions = torch.sigmoid(outputs) > 0.5
            correct += (predictions.float() == labels).sum().item()
            total += labels.size(0)

    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Test Loss: {test_loss/len(test_loader):.4f} | "
              f"Accuracy: {correct/total:.4f}")

print("Training complete.")