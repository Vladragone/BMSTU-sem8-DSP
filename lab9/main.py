import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import restoration, color

import warnings

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')


class SimpleDeconvApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР 9: Восстановление изображений ('Слепая' деконволюция)")
        self.geometry("1000x600")

        self.original_image = None
        self.build_ui()

    def build_ui(self):
        # --- Верхняя панель с кнопками ---
        top_frame = tk.Frame(self, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = tk.Button(top_frame, text="1. Загрузить картинку", command=self.load_image, font=("Arial", 12))
        self.btn_load.pack(side=tk.LEFT, padx=20)

        self.btn_restore = tk.Button(top_frame, text="2. Восстановить", command=self.restore_image, font=("Arial", 12),
                                     state=tk.DISABLED)
        self.btn_restore.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(top_frame, text="Ожидание загрузки...", font=("Arial", 10, "italic"), fg="gray")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # --- Область с графиками (До и После) ---
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_restored = self.fig.add_subplot(122)

        self.fig.subplots_adjust(wspace=0.1, left=0.05, right=0.95, top=0.9, bottom=0.05)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.clear_axes()

    def clear_axes(self):
        """Очистка осей перед новой отрисовкой"""
        self.ax_orig.clear()
        self.ax_restored.clear()
        self.ax_orig.axis('off')
        self.ax_restored.axis('off')
        self.ax_orig.set_title("Source image (Исходное)")
        self.ax_restored.set_title("Recovered image (Восстановленное)")
        self.canvas.draw()

    def get_motion_psf(self, length, angle):
        """
        Аналог MATLAB fspecial('motion', length, angle).
        Использует SciPy вместо OpenCV.
        """
        # Создаем пустую квадратную матрицу размером length x length
        psf = np.zeros((length, length), dtype=np.float32)

        # Рисуем горизонтальную линию ровно по центру
        psf[length // 2, :] = 1.0

        # Поворачиваем матрицу на заданный угол
        psf = ndimage.rotate(psf, angle, reshape=False, order=1)

        # Убираем возможные отрицательные артефакты после интерполяции при повороте
        psf[psf < 0] = 0

        # Строго нормализуем (сумма всех элементов должна быть равна 1)
        return psf / psf.sum()

    def load_image(self):
        """Загрузка изображения, перевод в ЧБ и нормализация 0..1"""
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg")])
        if not filepath:
            return

        try:
            # Читаем картинку с помощью PIL
            img_pil = Image.open(filepath).convert('RGB')
            # Переводим в numpy массив и градации серого
            img_gray = color.rgb2gray(np.array(img_pil))

            # Эквивалент MATLAB: double(imread(...)) / 255
            # img_gray от scikit-image уже находится в диапазоне [0.0, 1.0]
            self.original_image = img_gray

            self.clear_axes()
            self.ax_orig.imshow(self.original_image, cmap='gray')
            self.ax_orig.set_title("Source image (Исходное)")

            self.canvas.draw()

            self.btn_restore.config(state=tk.NORMAL)
            self.status_label.config(text="Картинка загружена. Нажмите 'Восстановить'.", fg="black")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение:\n{e}")

    def restore_image(self):
        """Процесс восстановления (деконволюции)"""
        if self.original_image is None:
            return

        self.status_label.config(text="Выполняется сложный расчет (Ричардсон-Люси)... Подождите.", fg="blue")
        self.update_idletasks()  # Заставляем интерфейс обновить надпись перед подвисанием

        try:
            # 1. Генерируем матрицу искажения. Как в вашем скрипте: 55, 205
            psf_length = 55
            psf_angle = 205
            psf = self.get_motion_psf(psf_length, psf_angle)

            # 2. Выполняем деконволюцию (восстановление)
            # В scikit-image алгоритм Ричардсона-Люси работает так же круто, как deconvblind
            recovered_img = restoration.richardson_lucy(self.original_image, psf, num_iter=30)

            # 3. Выводим результат
            self.ax_restored.clear()
            self.ax_restored.axis('off')
            # vmin=0, vmax=1 гарантируют, что цвета не "съедут" при нормализации
            self.ax_restored.imshow(recovered_img, cmap='gray', vmin=0, vmax=1)
            self.ax_restored.set_title("Recovered image (Восстановленное)")

            self.canvas.draw()
            self.status_label.config(text="Готово!", fg="green")

        except Exception as e:
            self.status_label.config(text="Ошибка расчета", fg="red")
            messagebox.showerror("Ошибка", f"Сбой в алгоритме:\n{e}")


if __name__ == "__main__":
    app = SimpleDeconvApp()
    app.mainloop()
