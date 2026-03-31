import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Параметры сетки
# =========================
n = 512
T = 10.0
dx = T / n
x = np.linspace(-T/2, T/2, n)

# =========================
# 2. Исходные сигналы
# =========================
u1 = np.exp(-x**2 / 2)
u2 = np.exp(-x**2 / 3)

# =========================
# 3. Добавление шума
# =========================
np.random.seed(0)

noise_p = 0.01

delta_level = noise_p * np.max(u1)
epsilon_level = noise_p * np.max(u1)

delta = delta_level * np.random.randn(n)
epsilon = epsilon_level * np.random.randn(n)

u2_delta = u2 + delta
u1_epsilon = u1 + epsilon

# =========================
# 4. Фурье-образы
# =========================
v1 = np.fft.fft(u1_epsilon)
v2 = np.fft.fft(u2_delta)

m = np.arange(n)

# =========================
# 5. Функции γ(α), β(α), ρ(α)
# =========================
def gamma(alpha):
    denom = (np.abs(v2)**2 * dx**2 + alpha * (1 + (2*np.pi*m/T)**2))**2
    num = (np.abs(v2)**2 * dx**2 *
           np.abs(v1)**2 *
           (1 + (np.pi*m/T)**2))
    return dx/n * np.sum(num / denom)

def beta(alpha):
    denom = (np.abs(v2)**2 * dx**2 + alpha * (1 + (2*np.pi*m/T)**2))**2
    num = (alpha**2 *
           (1 + (2*np.pi*m/T)**2) *
           np.abs(v1)**2)
    return dx/n * np.sum(num / denom)

def rho(alpha):
    return beta(alpha) - (delta_level + epsilon_level * np.sqrt(gamma(alpha)))**2

# =========================
# 6. Поиск α методом невязки
# =========================
def find_alpha_bisect(a, b, tol=1e-6):
    while abs(b - a) > tol:
        c = (a + b) / 2
        if rho(c) > 0:
            b = c
        else:
            a = c
            
    return (a + b) / 2

print("rho(1e-6) =", rho(1e-6))
print("rho(1.0) =", rho(1.0))

alpha = find_alpha_bisect(1e-6, 1.0)

print(f"Найденное alpha = {alpha}")
print(f"Значение функции rho(alpha) = {rho(alpha)}")

# =========================
# 7. Вычисление H(k)
# =========================
H = np.zeros(n, dtype=complex)

for k in range(n):
    sum_val = 0
    for m_idx in range(n):
        denom = (np.abs(v2[m_idx])**2 * dx**2 +
                 alpha * (1 + (2*np.pi*m_idx/T)**2))
        sum_val += (np.exp(-2j*np.pi*k*m_idx/n) *
                    np.conj(v2[m_idx]) *
                    v1[m_idx]) / denom
    H[k] = dx/n * sum_val

# =========================
# 8. Визуализация
# =========================
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.plot(x, u1, label='u1 (ориг.)')
plt.plot(x, u1_epsilon, '--', label='u1 с шумом')
plt.legend()
plt.title('Сигнал u1')

plt.subplot(2,2,2)
plt.plot(x, u2, label='u2 (ориг.)')
plt.plot(x, u2_delta, '--', label='u2 с шумом')
plt.legend()
plt.title('Сигнал u2')

plt.subplot(2,2,3)
plt.plot(np.real(H))
plt.title('Импульсный отклик Re(H)')

plt.subplot(2,2,4)
plt.plot(np.imag(H))
plt.title('Импульсный отклик Im(H)')

plt.tight_layout()
plt.show()

# =========================
# 9. Восстановление сигнала
# =========================
# =========================
# 8–9. Восстановление сигнала
# =========================
u1_restored = np.real(np.fft.ifft(np.fft.fft(u2_delta) * H))

# =========================
# 10. Итоговый график
# =========================
plt.figure(figsize=(10,6))

# сигнал 1 (с шумом)
plt.plot(x, u1_epsilon, '--', label='Сигнал 1 (с шумом)')

# сигнал 2 (с шумом)
plt.plot(x, u2_delta, '--', label='Сигнал 2 (с шумом)')

# восстановленный сигнал (после фильтра Тихонова)
plt.plot(x, u1_restored, label='Восстановленный (Тихонов)')

plt.legend()
plt.title('Восстановление сигнала методом Тихонова')
plt.grid()

plt.show()