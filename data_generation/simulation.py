import numpy as np
from tqdm import tqdm


N = 5000
from platon.transit_depth_calculator import TransitDepthCalculator
import numpy as np

model = TransitDepthCalculator()

def build_sample():
    T = np.random.uniform(800, 2000)
    metallicity = np.random.uniform(-1, 2)
    cloud = np.random.uniform(1, 6)

    wl, depth, _ = model.compute_depths(
    planet_radius=7e7,
    planet_mass=1.9e27,
    star_radius=6.96e8,
    temperature=T,
    logZ=metallicity,
    cloudtop_pressure=10**cloud
)

    y = np.array([T, metallicity, cloud])
    return wl, depth, y 


X, Y = [], []

for _ in tqdm(range(N)):
    wl, spec, y = build_sample()
    X.append(spec)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)

np.save("spectra.npy", X)
np.save("labels.npy", Y)
