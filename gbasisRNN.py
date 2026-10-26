import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import ast

class GroebnerDatasetTuples(Dataset):
    def __init__(self, filepath=r"C:\Users\HP\PycharmProjects\PythonProject4\gbasisdata\data_tuples.txt", num_generators=6, num_vars=6, max_terms=10):
        self.data = []
        self.output = []
        self.num_generators = num_generators
        self.num_vars = num_vars
        self.max_terms = max_terms

        with open(filepath, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            ordering_str, ideal_str, label_str = line.split("|")

            # Ordering (stored as tuple like (2))
            ordering = int(ordering_str.strip("()"))

            # semicolon separates polynomials, commas separate monomials
            generators = []
            for poly_str in ideal_str.split(";"):
                # Wrap in [] to make python list
                poly_str = "[" + poly_str + "]"
                monomials = ast.literal_eval(poly_str)

                # Represent polynomial as tensor [seq_len, num_vars+1]
                g = torch.zeros(max_terms, num_vars + 1)
                for i, (coeff, exps) in enumerate(monomials[:max_terms]):
                    g[i, 0] = coeff
                    g[i, 1:] = torch.tensor(exps)
                generators.append(g)

            # padding zeros like I was saying earlier to poly with less monomial
            while len(generators) < num_generators:
                generators.append(torch.zeros(max_terms, num_vars + 1))

            self.data.append((generators, torch.tensor(ordering)))
            self.output.append(torch.tensor([float(label_str)], dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        generators, ordering = self.data[idx]
        y = self.output[idx]
        return generators, ordering, y

dataset = GroebnerDatasetTuples(
    filepath=r"../unknown/data_tuples.txt",
    num_generators=6,
    num_vars=6,
    max_terms=10
)

class GroebnerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, summary_size, output_size= 1, num_generators=6):
        super(GroebnerRNN, self).__init__()

        self.hidden_size= hidden_size
        self.num_generator= num_generators

        self.rnn= nn.RNN(input_size, hidden_size)

        self.feedforward= nn.Sequential(
            nn.Linear(summary_size * num_generators + 1, 128), # input has +1 for ordering
            nn.Tanh(),
            nn.Linear(128, output_size),
        )


    def forward(self, generators, monomial_ordering):
        summary = []

        for g in generators:  # g shape: [max_terms, num_vars+1]
            g = g.unsqueeze(0)  # add batch dim → [1, max_terms, input_size]
            output_n, hidden_n = self.rnn(g)
            hidden_n = hidden_n.squeeze(0)  # shape: [hidden_size]
            summary.append(hidden_n)

        all_summary = torch.cat(summary, dim=0)  # shape: [num_generators * hidden_size]

        total_input = torch.cat((all_summary, monomial_ordering.float()), dim=0)

        output = self.feedforward(total_input)
        return output

#training

model = GroebnerRNN(67,2,3)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

dataset = GroebnerDatasetTuples(data, labels)
loader = DataLoader(dataset, batch_size=30, shuffle=True)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for generators, y in loader:
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(generators)  # [B]

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

