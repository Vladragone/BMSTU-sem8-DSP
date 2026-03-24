import numpy as np
import matplotlib.pyplot as plt

# Гауссов импульс
def gauspls(x, A, s):
    return A * np.exp(-(x / s) ** 2)

# Импульсный шум
def impnoise(size, N, mult):
    step = size // N
    y = np.zeros(size)
    
    for i in range(1, N // 2 + 1):
        y[size // 2 + i * step] = mult * (0.5 + np.random.rand())
        y[size // 2 - i * step] = mult * (0.5 + np.random.rand())
    
    return y

# Фильтр Винера (как в MATLAB коде)
def wiener(x, n):
    return 1 - (n / x) ** 2

# Параметры
A = 1.0
sigma = 0.5

mult = 5
step = 0.005
t = np.arange(-mult, mult, step)

# Идеальный сигнал
x0 = gauspls(t, A, sigma)

# Гауссов шум
NA = 0
NS = 0.05
n1 = np.random.normal(NA, NS, len(x0))
x1 = x0 + n1

# Импульсный шум
count = 7
M = 0.4
n2 = impnoise(len(x0), count, M)
x2 = x0 + n2

# FFT
X1 = np.fft.fft(x1)
N1 = np.fft.fft(n1)

X2 = np.fft.fft(x2)
N2 = np.fft.fft(n2)

# Фильтры
Y1 = wiener(X1, N1)
Y2 = wiener(X2, N2)

# Восстановленные сигналы
y1 = np.fft.ifft(X1 * Y1)
y2 = np.fft.ifft(X2 * Y2)

plt.figure(figsize=(15, 4))

# 1. Исходный сигнал
plt.subplot(1, 3, 1)
plt.plot(t, x0)
plt.title('Исходный сигнал')
plt.grid()

# 2. С шумом
plt.subplot(1, 3, 2)
plt.plot(t, x1)
plt.title('С гауссовым шумом')
plt.grid()

# 3. После фильтрации
plt.subplot(1, 3, 3)
plt.plot(t, np.real(y1))
plt.title('После фильтрации')
plt.grid()

plt.suptitle('Фильтрация гауссова шума фильтром Винера')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 4))

# 1. Исходный сигнал
plt.subplot(1, 3, 1)
plt.plot(t, x0)
plt.title('Исходный сигнал')
plt.grid()

# 2. С шумом
plt.subplot(1, 3, 2)
plt.plot(t, x2)
plt.title('С импульсным шумом')
plt.grid()

# 3. После фильтрации
plt.subplot(1, 3, 3)
plt.plot(t, np.real(y2))
plt.title('После фильтрации')
plt.grid()

plt.suptitle('Фильтрация импульсного шума фильтром Винера')
plt.tight_layout()
plt.show()