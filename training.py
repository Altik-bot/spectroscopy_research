import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
N = 5000
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4



# -----------------------
# LOAD DATA
# -----------------------
X = np.load("spectra.npy")
Y = np.load("labels.npy")

X = np.array([x[::100] for x in X])

# normalize spectra
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

# normalize targets (important for stability)
Y_mean = Y.mean(axis=0)
Y_std = Y.std(axis=0)
Y = (Y - Y_mean) / Y_std

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------
# MODEL
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.relu = nn.ReLU()

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class SpectraResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4),
            nn.ReLU()
        )

        self.block1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool1d(2)

        self.block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool1d(2)

        self.block3 = ResidualBlock(128, 256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.head(x)


# -----------------------
# TRAINING
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpectraResNet().to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)


        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch, "Loss:", total_loss / len(loader))
torch.save(model.state_dict(), "spectra_model.pth")
print("Model saved")
print("Training finished.")