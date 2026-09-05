import numpy as np
import torch
import torch.nn as nn
from inference import predict,SpectraResNet
from matplotlib.animation import FuncAnimation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("spectra_model_full.pth", map_location=device, weights_only=False)
model = SpectraResNet().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
X = np.load("spectra.npy")
Y = np.load("labels.npy")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def update(angle):
    ax.view_init(elev=10, azim=angle)

df = pd.read_csv("planets.csv")
df["log_metallicity"] = pd.to_numeric(df["log_metallicity"], errors="coerce")
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
df["log_cloud"] = pd.to_numeric(df["log_cloud"], errors="coerce")
pivot = df.pivot_table(index="log_metallicity", columns="temperature", values="log_cloud")

X = pivot.columns.values   
Y = pivot.index.values     
X, Y = np.meshgrid(X, Y)
Z = pivot.values
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
global ax 
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["log_metallicity"], df["temperature"], df["log_cloud"])

ax.set_xlabel("log_metallicity")
ax.set_ylabel("temperatture")
ax.set_zlabel("log_cloud")
ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)
plt.savefig("temp_metallicity_cloud")
plt.show()    