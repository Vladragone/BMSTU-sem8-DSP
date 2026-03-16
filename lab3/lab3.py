import numpy as np
import matplotlib.pyplot as plt

def lab03():
    # Входные параметры
    c = 2.0
    sigma = 1.0
    
    # ОДЗ (область допустимых значений)
    t_max = 5
    dt = 0.05
    t = np.arange(-t_max, t_max + dt, dt)
    
    # Генерация сигналов
    x1 = np.concatenate([rectangular(t, c), np.zeros(len(t))])
    x2 = np.concatenate([gaussian(t, sigma), np.zeros(len(t))])
    x3 = np.concatenate([rectangular(t, c/2), np.zeros(len(t))])
    x4 = np.concatenate([gaussian(t, sigma/2), np.zeros(len(t))])
    
    # Свертка
    # Фурье-образ взаимной свертки равен произведению Фурье-образов свертываемых функций.
    y1 = np.fft.ifft(np.fft.fft(x1) * np.fft.fft(x2)) * dt
    y2 = np.fft.ifft(np.fft.fft(x1) * np.fft.fft(x3)) * dt
    y3 = np.fft.ifft(np.fft.fft(x2) * np.fft.fft(x4)) * dt
    
    # Нормализация свёртки (берём только действительную часть)
    y1 = np.real(y1)
    y2 = np.real(y2)
    y3 = np.real(y3)
    
    # Центрирование свёртки
    start = (len(y1) - len(t)) // 2
    y1 = y1[start:start + len(t)]
    y2 = y2[start:start + len(t)]
    y3 = y3[start:start + len(t)]
    
    # Создание фигуры с тремя подграфиками
    plt.figure(figsize=(10, 12))
    
    graph_figure(1, t, x1[:len(t)], x2[:len(t)], y1, 'R + G')
    graph_figure(2, t, x1[:len(t)], x3[:len(t)], y2, 'R + R')
    graph_figure(3, t, x2[:len(t)], x4[:len(t)], y3, 'G + G')
    
    plt.suptitle('Реализация частотного алгоритма вычисления свертки', fontsize=14)
    plt.tight_layout()
    plt.show()

def graph_figure(i, t, x1, x2, y, tit):
    plt.subplot(3, 1, i)
    plt.plot(t, x1, 'r', label='Сигнал 1')
    plt.plot(t, x2, 'b', label='Сигнал 2')
    plt.plot(t, y, 'k', label='Свёртка')
    plt.grid(True)
    plt.title(f'Свёртка {tit}')
    plt.legend()

# Rectangular pulse generation
def rectangular(x, c):
    y = np.zeros_like(x)
    y[np.abs(x) - c < 0] = 1
    y[np.abs(x) == c] = 0.5
    return y

# Gaussian pulse generation
def gaussian(x, sigma):
    return np.exp(-(x / sigma) ** 2)

if __name__ == "__main__":
    lab03()
