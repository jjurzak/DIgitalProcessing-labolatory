import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

np.random.seed(42) 
N = 1000  
white_noise = np.random.normal(0, 1, N)  

mean_estimate = np.mean(white_noise)
variance_estimate = np.var(white_noise)

running_mean = np.cumsum(white_noise) / np.arange(1, N + 1)

frequencies, psd = periodogram(white_noise, fs=1.0)

plt.figure(figsize=(12, 8))

# Plot White Noise Signal
plt.subplot(3, 1, 1)
plt.plot(white_noise, label="White Gaussian Noise")
plt.title("White Gaussian Noise Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

# Plot Running Mean
plt.subplot(3, 1, 2)
plt.plot(running_mean, label="Running Mean", color="orange")
plt.axhline(mean_estimate, color="red", linestyle="--", label="Estimated Mean")
plt.title("Running Mean (Ergodicity Check)")
plt.xlabel("Sample Index")
plt.ylabel("Running Mean")
plt.legend()

# Plot PSD
plt.subplot(3, 1, 3)
plt.semilogy(frequencies, psd, label="PSD (Periodogram)")
plt.title("Power Spectral Density (PSD)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.legend()

plt.tight_layout()
plt.show()

# Print Results
print(f"Estimated Mean: {mean_estimate}")
print(f"Estimated Variance: {variance_estimate}")