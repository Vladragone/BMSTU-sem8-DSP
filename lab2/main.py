import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def dft_slow(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W @ x


def make_rect(N: int, half_width: int) -> np.ndarray:
    n_centered = np.arange(-N // 2, N // 2)
    x_centered = (np.abs(n_centered) <= half_width).astype(float)
    return np.fft.ifftshift(x_centered)


def make_gaussian(N: int, sigma: float) -> np.ndarray:
    n_centered = np.arange(-N // 2, N // 2)
    x_centered = np.exp(-(n_centered ** 2) / (2 * sigma ** 2))
    return np.fft.ifftshift(x_centered)


def remove_twins_by_modulation(x: np.ndarray) -> np.ndarray:
    n = np.arange(x.size)
    return x * ((-1) ** n)


def freq_axis(N: int, fs: float = 1.0) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))


def measure_fft_and_dft(x: np.ndarray):
    start_fft = perf_counter()
    X_fft = np.fft.fft(x)
    fft_time = perf_counter() - start_fft

    start_dft = perf_counter()
    X_dft = dft_slow(x)
    dft_time = perf_counter() - start_dft

    return X_fft, X_dft, fft_time, dft_time


def build_spectrum_data(x: np.ndarray, fs: float = 1.0):
    X_fft, X_dft, fft_time, dft_time = measure_fft_and_dft(x)
    return {
        "f": freq_axis(x.size, fs=fs),
        "fft_shift": np.fft.fftshift(X_fft),
        "dft_shift": np.fft.fftshift(X_dft),
        "fft_time": fft_time,
        "dft_time": dft_time,
    }


def plot_requested_graphs(rect_raw, rect_fixed, gauss_raw, gauss_fixed):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(gauss_raw["f"], np.abs(gauss_raw["dft_shift"]), label="with twins")
    axes[0, 0].plot(gauss_fixed["f"], np.abs(gauss_fixed["dft_shift"]), "--", label="without twins")
    axes[0, 0].set_title("GAUSS: DFT amplitude spectrum")
    axes[0, 0].set_xlabel("f")
    axes[0, 0].set_ylabel("|X(f)|")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(gauss_raw["f"], np.abs(gauss_raw["fft_shift"]), label="with twins")
    axes[0, 1].plot(gauss_fixed["f"], np.abs(gauss_fixed["fft_shift"]), "--", label="without twins")
    axes[0, 1].set_title("GAUSS: FFT amplitude spectrum")
    axes[0, 1].set_xlabel("f")
    axes[0, 1].set_ylabel("|X(f)|")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(rect_raw["f"], np.abs(rect_raw["dft_shift"]), label="with twins")
    axes[1, 0].plot(rect_fixed["f"], np.abs(rect_fixed["dft_shift"]), "--", label="without twins")
    axes[1, 0].set_title("RECT: DFT amplitude spectrum")
    axes[1, 0].set_xlabel("f")
    axes[1, 0].set_ylabel("|X(f)|")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(rect_raw["f"], np.abs(rect_raw["fft_shift"]), label="with twins")
    axes[1, 1].plot(rect_fixed["f"], np.abs(rect_fixed["fft_shift"]), "--", label="without twins")
    axes[1, 1].set_title("RECT: FFT amplitude spectrum")
    axes[1, 1].set_xlabel("f")
    axes[1, 1].set_ylabel("|X(f)|")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    fig.tight_layout()
    plt.show()


def main():
    N = 256
    fs = 1.0

    x_rect = make_rect(N=N, half_width=20)
    x_rect_fixed = remove_twins_by_modulation(x_rect)
    x_gauss = make_gaussian(N=N, sigma=10.0)
    x_gauss_fixed = remove_twins_by_modulation(x_gauss)

    rect_raw_data = build_spectrum_data(x_rect, fs=fs)
    rect_fixed_data = build_spectrum_data(x_rect_fixed, fs=fs)
    gauss_raw_data = build_spectrum_data(x_gauss, fs=fs)
    gauss_fixed_data = build_spectrum_data(x_gauss_fixed, fs=fs)

    print(f'RECT with twins: FFT {rect_raw_data["fft_time"]:.6f} s, DFT {rect_raw_data["dft_time"]:.6f} s')
    print(f'RECT without twins: FFT {rect_fixed_data["fft_time"]:.6f} s, DFT {rect_fixed_data["dft_time"]:.6f} s')
    print(f'GAUSS with twins: FFT {gauss_raw_data["fft_time"]:.6f} s, DFT {gauss_raw_data["dft_time"]:.6f} s')
    print(f'GAUSS without twins: FFT {gauss_fixed_data["fft_time"]:.6f} s, DFT {gauss_fixed_data["dft_time"]:.6f} s')

    plot_requested_graphs(rect_raw_data, rect_fixed_data, gauss_raw_data, gauss_fixed_data)


if __name__ == "__main__":
    main()
