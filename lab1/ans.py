import numpy as np
import matplotlib.pyplot as plt

# Parameters (same style as in lab1/main.py)
A = 1.0
sigma = 1.5
L = 3.0

# Frequency axis in Hz
f = np.linspace(-5, 5, 4000)

# Fourier transform convention:
# X(f) = ∫ x(t) * exp(-j 2π f t) dt
# For x_g(t) = A * exp(-t^2 / (2*sigma^2))
X_gauss = A * sigma * np.sqrt(2 * np.pi) * np.exp(-2 * (np.pi**2) * (sigma**2) * (f**2))

# For x_r(t) = 1 on |t| <= L/2 else 0
# X_rect(f) = L * sinc(f*L), where np.sinc(z) = sin(pi*z)/(pi*z)
X_rect = L * np.sinc(f * L)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(f, X_gauss, label='Gaussian spectrum $X_G(f)$', linewidth=2)
plt.plot(f, X_rect, label='Rectangular spectrum $X_R(f)$', linewidth=2)
plt.axhline(0, color='black', linewidth=0.8)
plt.title('Continuous-time spectra')
plt.xlabel('f, Hz')
plt.ylabel('X(f)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(f, np.abs(X_gauss) + 1e-15, label='|$X_G(f)$|', linewidth=2)
plt.semilogy(f, np.abs(X_rect) + 1e-15, label='|$X_R(f)$|', linewidth=2)
plt.title('Magnitude spectra (log scale)')
plt.xlabel('f, Hz')
plt.ylabel('|X(f)|')
plt.grid(True, which='both', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
