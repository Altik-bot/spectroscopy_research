import torch
import torch.nn as nn
import csv
import time 
# =========================
# NOISE FUNCTION
# =========================
def add_gaussian_noise(x, std=0.01):
    noise = torch.randn_like(x) * std
    return x + noise


# =========================
# MODEL
# =========================
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


# =========================
# VALIDATION
# =========================
def val_loss(model, loader, criterion, device):
    model.eval()
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            total += loss.item()

    return total / len(loader)


# =========================
# TRAIN FUNCTION
# =========================
def train(noise, gen, train_loader, val_loader, y_mean, y_std):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectraResNet().to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("Training started")
    print("Noise level:", noise)

    with open("output.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if gen == 0:
            writer.writerow(["gen", "epoch", "train_loss", "val_loss"])

        for epoch in range(15):
            model.train()
            total_loss = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                # noise injection (training only)
                if noise > 0:
                    x = add_gaussian_noise(x, std=noise)

                pred = model(x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            v_loss = val_loss(model, val_loader, criterion, device)

            writer.writerow([gen, epoch, train_loss, v_loss])

            print("Epoch:", epoch,
                  "Train Loss:", train_loss,
                  "Val Loss:", v_loss)
            print("Cooling started")
            time.sleep(180)
            print("Cooling ended")
            

    torch.save({
        "model_state": model.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std
    }, f"spectra_model_noise_{noise:.2f}.pth")

    print("Model saved")
    print("Training finished")

    return model, y_mean, y_std