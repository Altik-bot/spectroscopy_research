from training_auto import train
from eval_auto import evaluate
import time 
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================
X = np.load("spectra.npy")
Y = np.load("labels.npy")

# Downsample spectra
X = np.array([x[::50] for x in X])

# =========================
# OPTIONAL: LOG SCALING (important for metallicity)
# =========================
Y[:, 0] = np.log10(Y[:, 0] + 1e-8)

# =========================
# GLOBAL NORMALIZATION (FIXED)
# =========================
X_mean = X.mean()
X_std = X.std()
X = (X - X_mean) / (X_std + 1e-8)

# =========================
# TARGET NORMALIZATION (SINGLE SOURCE OF TRUTH)
# =========================
y_mean = Y.mean(axis=0)
y_std = Y.std(axis=0)
Y_norm = (Y - y_mean) / y_std

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(Y_norm, dtype=torch.float32)

# =========================
# DATASET + SPLIT
# =========================
dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# =========================
# TRAIN + EVALUATE
# =========================
results = []

for i in [0] + list(range(1, 11)):
    noise_level = i / 100
    print("Noise level:", noise_level)

    model, Y_mean, Y_std = train(
        noise_level,
        i,
        train_loader,
        val_loader,
        y_mean,
        y_std
    )

    # Option A: evaluate with noise
    metrics = evaluate(
        model,
        val_loader,
        Y_mean,
        Y_std,
        noise_std=noise_level
    )
    time.sleep(180)
    results.append({
        "noise": noise_level,

        "mae_metallicity": metrics["mae"][0].item(),
        "mae_cloud": metrics["mae"][1].item(),
        "mae_temperature": metrics["mae"][2].item(),

        "rmse_metallicity": metrics["rmse"][0].item(),
        "rmse_cloud": metrics["rmse"][1].item(),
        "rmse_temperature": metrics["rmse"][2].item(),

        "r2_metallicity": metrics["r2"][0].item(),
        "r2_cloud": metrics["r2"][1].item(),
        "r2_temperature": metrics["r2"][2].item(),
    })

for r in results:
    print(r)