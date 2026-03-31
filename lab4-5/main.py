import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt


def gauspls(x, amplitude, sigma):
    return amplitude * np.exp(-((x / sigma) ** 2))


def impnoise(size, count, amplitude):
    step = size // count
    noise = np.zeros(size)

    for i in range(1, count // 2 + 1):
        right = int(round(size / 2 + i * step))
        left = int(round(size / 2 - i * step))

        if 0 <= right < size:
            noise[right] = amplitude * (0.5 + np.random.rand())
        if 0 <= left < size:
            noise[left] = amplitude * (0.5 + np.random.rand())

    return noise


def buttfilt(diameter, size, filter_type):
    x = np.linspace(-size / 2, size / 2, size)

    if filter_type == "low":
        y = 1 / (1 + (x / diameter) ** 4)
    elif filter_type == "high":
        safe_x = np.where(np.abs(x) < 1e-12, 1e-12, x)
        y = 1 / (1 + (diameter / safe_x) ** 4)
        y[np.abs(x) < 1e-12] = 0.0
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    return y / np.sum(y)


def gaussfilt(sigma, size, filter_type):
    x = np.linspace(-size / 2, size / 2, size)

    if filter_type == "low":
        y = np.exp(-(x**2) / (2 * sigma**2))
    elif filter_type == "high":
        y = 1 - np.exp(-(x**2) / (2 * sigma**2))
    else:
        raise ValueError("filter_type must be 'low' or 'high'")

    return y / np.sum(y)


def build_signals():
    amplitude = 1.0
    sigma = 0.5
    mult = 5
    step = 0.005
    t = np.arange(-mult, mult + step, step)

    clean = gauspls(t, amplitude, sigma)
    gaussian_noise = np.random.normal(0.0, 0.05, size=clean.size)
    impulse_noise = impnoise(clean.size, 7, 0.4)

    noisy_gaussian = clean + gaussian_noise
    noisy_impulse = clean + impulse_noise

    return t, clean, noisy_gaussian, noisy_impulse


def low_frequency_results(clean, noisy_gaussian, noisy_impulse):
    gaussian_kernel = gaussfilt(4, 20, "low")
    butter_kernel = buttfilt(6, 20, "low")

    return {
        "Исходные сигналы": {
            "Без помех": clean,
            "Помеха по Гауссу": noisy_gaussian,
            "Импульсная помеха": noisy_impulse,
        },
        "Гауссовский фильтр": {
            "Без помех": clean,
            "Помеха по Гауссу": filtfilt(gaussian_kernel, [1.0], noisy_gaussian),
            "Импульсная помеха": filtfilt(gaussian_kernel, [1.0], noisy_impulse),
        },
        "Фильтр Баттеруорта": {
            "Без помех": clean,
            "Помеха по Гауссу": filtfilt(butter_kernel, [1.0], noisy_gaussian),
            "Импульсная помеха": filtfilt(butter_kernel, [1.0], noisy_impulse),
        },
    }


def high_frequency_results(clean, noisy_gaussian, noisy_impulse):
    gaussian_kernel = gaussfilt(4, 20, "high")
    butter_kernel = buttfilt(6, 20, "high")

    gaussian_low = gaussfilt(4, 20, "low")
    butter_low = buttfilt(6, 20, "low")

    return {
        "Исходные сигналы": {
            "Без помех": clean,
            "Помеха по Гауссу": noisy_gaussian,
            "Импульсная помеха": noisy_impulse,
        },
        "Гауссовский фильтр": {
            "Без помех": clean,
            "Помеха по Гауссу": noisy_gaussian - filtfilt(gaussian_low, [1.0], noisy_gaussian),
            "Импульсная помеха": noisy_impulse - filtfilt(gaussian_low, [1.0], noisy_impulse),
        },
        "Фильтр Баттеруорта": {
            "Без помех": clean,
            "Помеха по Гауссу": noisy_gaussian - filtfilt(butter_low, [1.0], noisy_gaussian),
            "Импульсная помеха": noisy_impulse - filtfilt(butter_low, [1.0], noisy_impulse),
        },
    }


def plot_window(figure_title, window_title, t, plots):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.canvas.manager.set_window_title(window_title)
    fig.suptitle(figure_title, fontsize=14)

    for ax, (title, series) in zip(axes, plots.items()):
        for label, values in series.items():
            ax.plot(t, values, label=label)
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("A")
        ax.grid(True)
        ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])


def main():
    t, clean, noisy_gaussian, noisy_impulse = build_signals()

    low_plots = low_frequency_results(clean, noisy_gaussian, noisy_impulse)
    high_plots = high_frequency_results(clean, noisy_gaussian, noisy_impulse)

    plot_window("Низкие частоты", "Низкие частоты", t, low_plots)
    plot_window("Высокие частоты", "Высокие частоты", t, high_plots)

    plt.show()


if __name__ == "__main__":
    main()
