import torch
from torch.utils.data import DataLoader
import numpy as np

from training import SpectraResNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



X = np.load("spectra.npy")
Y = np.load("labels.npy")

X = np.array([x[::50] for x in X])
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(Y, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(X, Y)

loader = DataLoader(dataset, batch_size=32, shuffle=False)



checkpoint = torch.load("spectra_model_full.pth", map_location=device, weights_only=False)

model = SpectraResNet().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

y_mean = torch.tensor(checkpoint["y_mean"], dtype=torch.float32)
y_std = torch.tensor(checkpoint["y_std"], dtype=torch.float32)



def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    preds_real = preds * y_std + y_mean
    labels_real = labels

    mae = torch.mean(torch.abs(preds_real - labels_real), dim=0)
    rmse = torch.sqrt(torch.mean((preds_real - labels_real) ** 2, dim=0))

    ss_res = torch.sum((labels_real - preds_real) ** 2, dim=0)
    ss_tot = torch.sum((labels_real - torch.mean(labels_real, dim=0)) ** 2, dim=0)
    r2 = 1 - ss_res / ss_tot

    names = ["Metallicity", "Cloud", "Temperature"]

    for i in range(len(names)):
        print(f"{names[i]}:")
        print(f"  MAE  = {mae[i].item():.4f}")
        print(f"  RMSE = {rmse[i].item():.4f}")
        print(f"  R2   = {r2[i].item():.4f}")
        print()



evaluate(model, loader)
