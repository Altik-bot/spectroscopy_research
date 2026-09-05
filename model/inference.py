import numpy as np
import torch
import torch.nn as nn
import csv

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

def predict(idx,device,checkpoint,model,X,Y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("spectra_model_full.pth", map_location=device, weights_only=False)
    model = SpectraResNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]



    spectrum = X[idx]
    true_label = Y[idx]



    spectrum_proc = spectrum[::50]
    spectrum_proc = (spectrum_proc - spectrum_proc.mean()) / (spectrum_proc.std() + 1e-8)

    x = torch.tensor(spectrum_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().squeeze()

    pred = pred_norm * y_std + y_mean
    with open("planets.csv", "a") as p:
        writer = csv.writer(p)
        if idx == 0:
            writer.writerow(["idx","log_metallicity","temperature","log_cloud"])
        writer.writerow([idx,pred[0],pred[1],pred[2]])   


    print("Prediction:")
    print("log_metallicity:", pred[0])
    print("temperature:", pred[1])
    print("log_cloud_pressure:", pred[2])

    print("\nGround Truth:")
    print("log_metallicity:", true_label[0])
    print("temperature:", true_label[1])
    print("log_cloud_pressure:", true_label[2])
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("spectra_model_full.pth", map_location=device, weights_only=False)
    model = SpectraResNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    X = np.load("spectra.npy")
    Y = np.load("labels.npy")
    predict(0,device,checkpoint=checkpoint,model = model,X=X,Y=Y)    