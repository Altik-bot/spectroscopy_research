Exoplanet Atmospheric Retrieval via Deep Learning

Project Overview
This project builds a deep learning system to infer exoplanet atmospheric composition from transmission spectra. It replaces traditional Bayesian atmospheric retrieval methods with fast neural network inference using 1D convolutional architectures.
The system learns a mapping from wavelength dependent transit depth data to chemical abundances such as H2O, CH4, CO2, CO, and NH3.

Scientific Motivation
When an exoplanet passes in front of its host star, starlight filters through the planet’s atmosphere. Molecules absorb specific wavelengths, producing spectral fingerprints.
Standard retrieval methods use Markov Chain Monte Carlo or nested sampling. These approaches are accurate but computationally expensive.
This project uses neural networks to approximate the inverse mapping and produce fast predictions from spectra.

Core Idea
Input:
- Transmission spectrum (wavelength vs transit depth)
Output:
- Atmospheric gas mixing ratios in log scale
Learning task:
- Supervised regression from synthetic spectra to atmospheric composition

Data Pipeline
The project follows this structure:
1. Forward modeling
   - Synthetic spectra generated using tools like TauREx 3 or petitRADTRANS
   - Physics-based radiative transfer simulation
   - Molecular absorption line modeling
2. Noise modeling
   - Gaussian photon noise in ppm range
   - Instrumental effects such as wavelength drift and stellar jitter
   - Telescope specific regimes like JWST and HST noise levels
3. Feature representation
   - Wavelength grid
   - Transit depth values
4. Learning stage
   - 1D CNN or ResNet model
   - Regression head for molecular abundances

Model Architecture
The model uses:
- 1D Convolutional Neural Network or Residual Network
- Local feature extraction across wavelength bands
- Regression head for chemical abundance prediction
Loss function:
- Mean Squared Error or Mean Absolute Error
Evaluation metrics:
- R² score
- MAE per molecule

Noise Study
A key research component is robustness under noise.
Experiments include:
- Clean synthetic spectra baseline
- Increasing Gaussian noise levels
- Instrumental distortion scenarios
Goal:
- Measure degradation in retrieval accuracy as noise increases

Dataset
Training data consists of:
- 10,000+ synthetic spectra
- Multiple exoplanet atmospheric configurations
- Variable temperature pressure profiles
- Randomized chemical abundances
Each sample includes:
- Wavelength array
- Transit depth spectrum
- Ground truth molecular abundances
