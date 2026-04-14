import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


def mygaussignal(x):
    a = 1.0
    sigma = 1.0
    return a * np.exp(-(x ** 2) / (sigma ** 2))


def mean_filter_value(ux, index):
    result = 0.0
    imin = index - 2
    imax = index + 2

    for j in range(imin, imax + 1):
        if 0 <= j < len(ux):
            result += ux[j]

    return result / 5.0


def med_filter_value(ux, index):
    imin = index - 1
    imax = index + 1

    if imin < 0:
        return ux[imax]

    if imax >= len(ux):
        return ux[imin]

    if ux[imax] > ux[imin]:
        return ux[imin]

    return ux[imax]


def run_mean_filter(ux, epsv):
    filtered = ux.copy()
    for i in range(len(filtered)):
        smthm = mean_filter_value(filtered, i)
        if abs(filtered[i] - smthm) > epsv:
            filtered[i] = smthm
    return filtered


def run_med_filter(ux, epsv):
    filtered = ux.copy()
    uxbase = ux.copy()
    for i in range(len(filtered)):
        smthm = med_filter_value(uxbase, i)
        if abs(filtered[i] - smthm) > epsv:
            filtered[i] = smthm
    return filtered


class Lab9App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР 9: Нелинейная фильтрация импульсных помех")
        self.geometry("1300x820")

        self.build_ui()
        self.run_calculation()

    def build_ui(self):
        panel = ttk.Frame(self, padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y)

        self.f_var = self.add_entry(panel, "F:", "3")
        self.dt_var = self.add_entry(panel, "dt:", "0.05")
        self.a_var = self.add_entry(panel, "a:", "0.25")
        self.epsv_var = self.add_entry(panel, "epsv:", "0.05")

        ttk.Button(
            panel,
            text="Построить",
            command=self.run_calculation,
        ).pack(fill="x", ipady=10, pady=(10, 0))

        plot_frame = ttk.Frame(self, padding=(0, 10, 10, 10))
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.ax_mean = self.figure.add_subplot(211)
        self.ax_med = self.figure.add_subplot(212)
        self.figure.subplots_adjust(hspace=0.4)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_entry(self, parent, label_text, default_value):
        ttk.Label(parent, text=label_text).pack(anchor="w")
        var = tk.StringVar(value=default_value)
        ttk.Entry(parent, textvariable=var, width=20).pack(fill="x", pady=(2, 8))
        return var

    def run_calculation(self):
        try:
            f_value = float(self.f_var.get())
            dt = float(self.dt_var.get())
            a = float(self.a_var.get())
            epsv = float(self.epsv_var.get())

            if dt <= 0:
                raise ValueError("dt должен быть больше 0")

            x = np.arange(-f_value, f_value + dt, dt)
            yx = mygaussignal(x)
            ux = mygaussignal(x)
            uxbase = mygaussignal(x)

            px = a * np.random.rand(7)
            pos = np.array([25, 35, 40, 54, 67, 75, 95]) - 1

            for i in range(len(pos)):
                if 0 <= pos[i] < len(ux):
                    ux[pos[i]] = ux[pos[i]] + px[i]
                    uxbase[pos[i]] = uxbase[pos[i]] + px[i]

            noisy_signal = ux.copy()
            ux_mean = run_mean_filter(ux.copy(), epsv)

            ux = mygaussignal(x)
            uxbase = mygaussignal(x)
            for i in range(len(pos)):
                if 0 <= pos[i] < len(ux):
                    ux[pos[i]] = ux[pos[i]] + px[i]
                    uxbase[pos[i]] = uxbase[pos[i]] + px[i]

            ux_med = run_med_filter(ux.copy(), epsv)

            self.ax_mean.clear()
            self.ax_mean.set_title("MEAN-функция фильтрации")
            self.ax_mean.plot(x, yx, label="Исходный гауссовский сигнал")
            self.ax_mean.plot(x, noisy_signal, label="Зашумленный сигнал")
            self.ax_mean.plot(x, ux_mean, label="Сглаженный сигнал")
            self.ax_mean.legend()
            self.ax_mean.grid(True, alpha=0.3)

            self.ax_med.clear()
            self.ax_med.set_title("MED-функция фильтрации")
            self.ax_med.plot(x, yx, label="Исходный гауссовский сигнал")
            self.ax_med.plot(x, noisy_signal, label="Зашумленный сигнал")
            self.ax_med.plot(x, ux_med, label="Сглаженный сигнал")
            self.ax_med.legend()
            self.ax_med.grid(True, alpha=0.3)

            self.canvas.draw()
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))


if __name__ == "__main__":
    app = Lab9App()
    app.mainloop()
