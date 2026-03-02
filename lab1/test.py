import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Общие параметры
# --------------------------------
T = 1.0                          # шаг дискретизации (для наглядности = 1)
t = np.linspace(-6, 6, 5000)     # "непрерывное" время
ts = np.arange(-5, 6, T)         # точки дискретизации

# --------------------------------
# 1. ОДНА sinc-функция
# --------------------------------
plt.figure(figsize=(6, 4))
plt.plot(t, np.sinc(t), label="sinc(t)")
plt.axhline(0)
plt.axvline(0)
plt.title("1. Одна sinc-функция")
plt.xlabel("t")
plt.ylabel("sinc(t)")
plt.grid()
plt.legend()
plt.show()

# --------------------------------
# 2. Свойство sinc: 1 в центре, 0 в узлах
# --------------------------------
plt.figure(figsize=(6, 4))
plt.plot(t, np.sinc(t), label="sinc(t)")
plt.plot(ts, np.sinc(ts), 'ro', label="sinc(k) в узлах")
plt.title("2. sinc = 1 в центре и 0 в остальных узлах")
plt.xlabel("t")
plt.ylabel("sinc(t)")
plt.grid()
plt.legend()
plt.show()

# --------------------------------
# 3. Один отсчёт → одна sinc
# --------------------------------
x_k = 2.0        # значение одного отсчёта
k0 = 1           # номер отсчёта
t0 = k0 * T

plt.figure(figsize=(6, 4))
plt.plot(t, x_k * np.sinc((t - t0) / T), label="Отсчёт → sinc")
plt.plot(t0, x_k, 'ro', label="Отсчёт")
plt.title("3. Один отсчёт превращается в sinc")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.legend()
plt.show()

# --------------------------------
# 4. Несколько отсчётов (дискретный сигнал)
# --------------------------------
x_samples = np.array([0, 1, 0, 1, 0])
ts_samples = np.arange(-2, 3, T)

plt.figure(figsize=(6, 4))
plt.stem(ts_samples, x_samples, basefmt=" ")
plt.title("4. Дискретные отсчёты")
plt.xlabel("t")
plt.ylabel("x[k]")
plt.grid()
plt.show()

# --------------------------------
# 5. Каждому отсчёту соответствует своя sinc
# --------------------------------
plt.figure(figsize=(6, 4))

for k in range(len(ts_samples)):
    plt.plot(
        t,
        x_samples[k] * np.sinc((t - ts_samples[k]) / T),
        alpha=0.6
    )

plt.title("5. Каждому отсчёту соответствует своя sinc")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.show()

# --------------------------------
# 6. СУММА sinc = восстановленный сигнал
# --------------------------------
x_rec = np.zeros_like(t)
for k in range(len(ts_samples)):
    x_rec += x_samples[k] * np.sinc((t - ts_samples[k]) / T)

plt.figure(figsize=(6, 4))
plt.plot(t, x_rec, label="Восстановленный сигнал")
plt.stem(ts_samples, x_samples, basefmt=" ", linefmt="gray", markerfmt="ro", label="Отсчёты")
plt.title("6. Сумма sinc-функций = сигнал")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid()
plt.legend()
plt.show()