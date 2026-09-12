import numpy as np

data = np.load("spectra.npy", allow_pickle=True)
lbl = np.load("labels.npy") 
print(data.shape)

idx = 0
one_lbl = lbl[idx]
one_spectrum = data[idx]

np.save("single_spectrum.npy", one_spectrum)
np.save("one_label.npy", one_lbl)