import numpy as np
import matplotlib.pyplot as plt

T = 0.1
t = np.linspace(-5, 5, 5000)
ts = np.arange(-5, 5, T)

L = 3

def rect_signal(t, L):
    return np.where(np.abs(t) <= L / 2, 1.0, 0.0)

x_rect = rect_signal(t, L)
x_rect_s = rect_signal(ts, L)

x_rect_rec = np.zeros_like(t)
for k in range(len(ts)):
    x_rect_rec += x_rect_s[k] * np.sinc((t - ts[k]) / T)

A = 1
sigma = 1.5

def gauss_signal(t, A, sigma):
    return A * np.exp(-t**2 / (2 * sigma**2))

x_gauss = gauss_signal(t, A, sigma)
x_gauss_s = gauss_signal(ts, A, sigma)

x_gauss_rec = np.zeros_like(t)
for k in range(len(ts)):
    x_gauss_rec += x_gauss_s[k] * np.sinc((t - ts[k]) / T)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, x_rect, label="Исходный сигнал")
plt.plot(t, x_rect_rec, '--', label="Восстановленный")
markerline, stemlines, baseline = plt.stem(
    ts, x_rect_s,
    markerfmt='o',
    label="Отсчёты"
)
plt.setp(markerline, color='black')
plt.setp(stemlines, visible=False)
plt.setp(baseline, visible=False)
plt.title("Прямоугольный импульс")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t, x_gauss, label="Исходный сигнал")
plt.plot(t, x_gauss_rec, '--', label="Восстановленный")
markerline, stemlines, baseline = plt.stem(
    ts, x_gauss_s,
    markerfmt='o',
    label="Отсчёты"
)
plt.setp(markerline, color='black')
plt.setp(stemlines, visible=False)
plt.setp(baseline, visible=False)
plt.title("Сигнал Гаусса")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
