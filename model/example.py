import torch
import numpy as np
import matplotlib.pyplot as plt

from training import SpectraResNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



checkpoint = torch.load("spectra_model_full.pth", map_location=device, weights_only=False)

model = SpectraResNet().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

y_mean = torch.tensor(checkpoint["y_mean"], dtype=torch.float32)
y_std = torch.tensor(checkpoint["y_std"], dtype=torch.float32)


X = np.load("spectra.npy")
Y = np.load("labels.npy")

idx = 0  
spectrum = X[idx]
true_label = Y[idx]

spectrum = spectrum[::50]
spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)

x = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
x = x.to(device)

with torch.no_grad():
    pred = model(x).cpu()

pred_real = pred * y_std + y_mean

names = ["Metallicity", "Cloud", "Temperature"]

print("\nTRUE vs PREDICTED\n")

for i in range(3):
    print(names[i])
    print("True :", float(true_label[i]))
    print("Pred :", float(pred_real[0][i]))
    print()


plt.plot(X[idx][::50])
plt.title("Input Spectrum (Flux vs Wavelength Index)")
plt.xlabel("Wavelength index")
plt.ylabel("Flux")
plt.show()
