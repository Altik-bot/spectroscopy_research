import torch
import numpy as np
import csv
from training import ResidualBlock, SpectraResNet

class InferenceEngine:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.checkpoint = torch.load(
        model_path,
        map_location=self.device,
        weights_only=False
        )
        self.model = SpectraResNet().to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()

        self.y_mean = self.checkpoint["y_mean"]
        self.y_std = self.checkpoint["y_std"]

    def predict(self, spectrum):
        spectrum = spectrum[::50]
        spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)

        x = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_norm = self.model(x).cpu().numpy().squeeze()

        return pred_norm * self.y_std + self.y_mean