import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy.polynomial import Polynomial



def load_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def image_to_spectrum(gray_img):
    spectrum = np.mean(gray_img, axis=0)
    spectrum = spectrum - np.min(spectrum)
    spectrum = spectrum / np.max(spectrum)
    return spectrum

def detect_peaks(spectrum):
    peaks, _ = find_peaks(spectrum, height=0.2, distance=20)
    return peaks

def calibrate(pixels, wavelengths):
    poly = Polynomial.fit(pixels, wavelengths, deg=1)
    return poly.convert()  

def pixel_to_wavelength(poly, length):
    pixels = np.arange(length)
    wavelengths = poly(pixels)
    return wavelengths

def resample_spectrum(wavelengths, spectrum, n_points=1000):
    wl_new = np.linspace(wavelengths.min(), wavelengths.max(), n_points)
    spectrum_new = np.interp(wl_new, wavelengths, spectrum)
    return wl_new, spectrum_new

def process_image(path, known_pixels, known_wavelengths):
    gray = load_image(path)
    spectrum = image_to_spectrum(gray)
    peaks = detect_peaks(spectrum)

    print("Detected peak pixels:", peaks)

    poly = calibrate(known_pixels, known_wavelengths)

    wavelengths = pixel_to_wavelength(poly, len(spectrum))

    wl_new, spec_new = resample_spectrum(wavelengths, spectrum)

    return wl_new, spec_new

if __name__ == "__main__":
    image_path = "photos/IMG_6410.jpeg"
known_pixels = np.array([752 , 2800 ,3615])
known_wavelengths = np.array([557.0, 587.1, 603.2])

wl, spec = process_image(image_path, known_pixels, known_wavelengths)

plt.plot(wl, spec)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Intensity")
plt.savefig("helium5.png")
plt.show()
