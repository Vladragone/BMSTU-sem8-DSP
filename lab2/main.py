import numpy as np
import matplotlib.pyplot as plt


def dft_slow(x: np.ndarray) -> np.ndarray:
    """
    ДПФ по формуле:
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.size
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-1j * 2 * np.pi * k * n / N)
    return W @ x


def make_rect(N: int, half_width: int) -> np.ndarray:
    """
    rect: 1 внутри [-half_width, half_width], иначе 0
    Определяем на оси n от -N/2 ... N/2-1 и потом "вкладываем" в массив длины N.
    """
    n_centered = np.arange(-N // 2, N // 2)
    x_centered = (np.abs(n_centered) <= half_width).astype(float)
    # Переносим в "обычную" индексацию 0..N-1
    x = np.fft.ifftshift(x_centered)
    return x


def make_gaussian(N: int, sigma: float) -> np.ndarray:
    """
    Гаусс: exp(-n^2/(2*sigma^2))
    Аналогично задаём на центрированной оси, затем ifftshift.
    """
    n_centered = np.arange(-N // 2, N // 2)
    x_centered = np.exp(-(n_centered ** 2) / (2 * sigma ** 2))
    x = np.fft.ifftshift(x_centered)
    return x


def freq_axis(N: int, fs: float = 1.0) -> np.ndarray:
    """
    Частотная ось для отображения (в условных единицах или Гц, если задан fs).
    Для shift-спектра удобно рисовать fftshift(fftfreq).
    """
    return np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))


def plot_signal_and_spectra(x: np.ndarray, title_prefix: str, fs: float = 1.0, do_slow_dft: bool = True):
    """
    Рисует:
    1) сигнал во времени (центрированно)
    2) |FFT| без shift (эффект "близнецов"/нулевая частота в начале массива)
    3) |FFT| с fftshift (исправленный вид)
    4) опционально сравнение с "медленным" ДПФ
    """
    N = x.size

    # "Временная" ось (центрированная для красоты)
    n_centered = np.arange(-N // 2, N // 2)
    x_centered = np.fft.fftshift(x)  # для отображения

    # FFT
    X_fft = np.fft.fft(x)
    X_fft_shift = np.fft.fftshift(X_fft)
    f = freq_axis(N, fs=fs)

    # Медленный ДПФ (может быть тяжёлым для больших N)
    X_dft = None
    if do_slow_dft:
        X_dft = dft_slow(x)
        # для честного сравнения отобразим тоже shift-версию
        X_dft_shift = np.fft.fftshift(X_dft)

    # --- Графики ---
    plt.figure(figsize=(12, 8))

    # 1) сигнал
    plt.subplot(2, 2, 1)
    plt.plot(n_centered, np.real(x_centered))
    plt.title(f"{title_prefix}: сигнал x[n] (центрированно)")
    plt.xlabel("n")
    plt.ylabel("x[n]")
    plt.grid(True)

    # 2) спектр "как есть" (часто воспринимается как с близнецами/неудобно)
    plt.subplot(2, 2, 2)
    plt.plot(np.abs(X_fft))
    plt.title(f"{title_prefix}: |FFT| без fftshift (как возвращает fft)")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid(True)

    # 3) спектр исправленный (нулевая частота в центре)
    plt.subplot(2, 2, 3)
    plt.plot(f, np.abs(X_fft_shift))
    plt.title(f"{title_prefix}: |FFT| с fftshift (исправленный вид)")
    plt.xlabel("f")
    plt.ylabel("|X(f)|")
    plt.grid(True)

    # 4) сравнение FFTshift vs DFTshift (если считали)
    plt.subplot(2, 2, 4)
    plt.plot(f, np.abs(X_fft_shift), label="FFT (shift)")
    if do_slow_dft:
        plt.plot(f, np.abs(X_dft_shift), "--", label="DFT по формуле (shift)")
    plt.title(f"{title_prefix}: сравнение FFT и ДПФ")
    plt.xlabel("f")
    plt.ylabel("|X(f)|")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def remove_twins_by_modulation(x: np.ndarray) -> np.ndarray:
    """
    Вариант, близкий к тому, что описано в задании:
    домножаем сигнал на exp(-j*pi*n), что эквивалентно сдвигу спектра на половину диапазона.
    В дискретном виде удобно:
      x2[n] = x[n] * (-1)^n
    потому что exp(-j*pi*n) = cos(pi*n) - j sin(pi*n) = (-1)^n (для целых n).
    """
    N = x.size
    n = np.arange(N)
    return x * ((-1) ** n)


def main():
    # Параметры
    N = 256
    fs = 1.0  # условная частота дискретизации (можешь поставить реальную)

    # --- rect(x) ---
    x_rect = make_rect(N=N, half_width=20)

    # Показать "близнецов" и исправление через fftshift
    plot_signal_and_spectra(x_rect, "RECT", fs=fs, do_slow_dft=True)

    # Дополнительно: "преобразование перед дискретизацией" (домножение на exp(-j*pi*n))
    x_rect_mod = remove_twins_by_modulation(x_rect)
    plot_signal_and_spectra(x_rect_mod, "RECT после умножения на exp(-j*pi*n)", fs=fs, do_slow_dft=False)

    # --- Gaussian ---
    x_gauss = make_gaussian(N=N, sigma=10.0)
    plot_signal_and_spectra(x_gauss, "GAUSS", fs=fs, do_slow_dft=True)

    x_gauss_mod = remove_twins_by_modulation(x_gauss)
    plot_signal_and_spectra(x_gauss_mod, "GAUSS после умножения на exp(-j*pi*n)", fs=fs, do_slow_dft=False)


if __name__ == "__main__":
    main()