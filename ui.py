import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from production_simulator import ProductionSimulator

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Имитационная модель производства 2б")
        self.root.geometry("1200x760")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(background="#f3f4f6")
        self._build_ui()

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(main_pane, padding=(10, 10, 10, 10))
        right_panel = ttk.Frame(main_pane, padding=(10, 10, 10, 10))
        main_pane.add(left_panel, weight=0)
        main_pane.add(right_panel, weight=1)

        param_frame = ttk.LabelFrame(left_panel, text="Параметры модели", padding=12)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        self.entries = {}
        params = [
            ("T", "Горизонт моделирования (время), T", "100"),
            ("dt", "Шаг дискретизации (timestep), dt", "1"),
            ("y_A0", "Начальный запас заготовок A на складе, y_A0", "120"),
            ("y_B0", "Начальный запас заготовок B на складе, y_B0", "150"),
            ("y_CA0", "Начальный запас деталей A в сборке, y_CA0", "20"),
            ("y_CB0", "Начальный запас деталей B в сборке, y_CB0", "30"),
            ("y_11_0", "Начальный буфер A после склада, y_11_0", "40"),
            ("P_A", "Мощность линии A, P_A", "40"),
            ("P_B", "Мощность линии B, P_B", "60"),
            ("P_C", "Мощность сборочного отдела, P_C", "40"),
            ("D11", "Задержка обработки A, D11", "10"),
            ("D21", "Задержка участка B1, D21", "2"),
            ("D22", "Задержка участка B2, D22", "6"),
            ("D23", "Задержка участка B3, D23", "10"),
            ("X1", "Внешний поток в склад A, X1", "15"),
            ("threshold_critical", "Порог пополнения, 0.2*y0", "0.2"),
            ("threshold_stop", "Порог остановки линии, 0.05*y0", "0.05"),
            ("seed", "Случайная семя для воспроизводимости", "123"),
        ]

        self.param_display_names = {key: label for key, label, default in params}
        self.param_descriptions = {
            "T": "Горизонт моделирования. В динамической модели это общий временной интервал, на котором рассчитывается изменение запасов и выпуск продукции. В предметной области это продолжительность оценки работы производства.",
            "dt": "Шаг дискретизации. Определяет частоту обновления внутренних состояний системы. В предметной области это временной шаг, через который контролируются запасы и выполнение операций.",
            "y_A0": "Начальный запас заготовок A на складе. Это начальное значение состояния склада для детали A, от которого начинается модель.",
            "y_B0": "Начальный запас заготовок B на складе. Это начальное значение состояния склада для детали B.",
            "y_CA0": "Начальный запас деталей A в сборочном отделе. Это начальное состояние промежуточного склада перед сборкой.",
            "y_CB0": "Начальный запас деталей B в сборочном отделе. Это начальное состояние промежуточного склада перед сборкой.",
            "y_11_0": "Начальный буфер A после склада и перед сборкой. Это промежуточный уровень, отражающий задержку между складом и сборкой для потока A.",
            "P_A": "Максимальная пропускная способность линии A. В динамике это ограничение на скорость расхода заготовок A.",
            "P_B": "Максимальная пропускная способность линии B. В динамике это ограничение на скорость расхода заготовок B.",
            "P_C": "Максимальная пропускная способность сборочного отдела. Это ограничение на скорость формирования готового изделия из деталей A и B.",
            "D11": "Время технологической задержки для потока A. Используется для моделирования накопления и освобождения буфера A.",
            "D21": "Время задержки на первом участке линии B. Это технологическое время обработки B на первом этапе.",
            "D22": "Время задержки на втором участке линии B. Это технологическое время обработки B на втором этапе.",
            "D23": "Время задержки на третьем участке линии B. Это технологическое время обработки B на третьем этапе.",
            "X1": "Постоянный внешний поток поставки (в единицах в шаг) на склад A. В предметной области это внешний вход материалов/сырья в производство.",
            "threshold_critical": "Критический уровень запаса, при достижении которого запускается пополнение. В предметной области это сигнал нехватки запаса на складе.",
            "threshold_stop": "Уровень запаса, при котором линия останавливается из-за недостатка материала. В предметной области это минимально безопасный запас для продолжения производства.",
            "seed": "Случайная семя. Используется для воспроизводимости случайных пополнений склада.",
        }

        for index, (name, label, default) in enumerate(params):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(param_frame, text=f"{label}:", anchor=tk.W).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=4)
            entry = ttk.Entry(param_frame, width=14)
            entry.insert(0, default)
            entry.grid(row=row, column=col + 1, sticky="w", pady=4)
            entry.bind("<FocusIn>", lambda event, name=name: self.show_param_description(name))
            entry.bind("<Button-1>", lambda event, name=name: self.show_param_description(name))
            self.entries[name] = entry
        param_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(3, weight=1)

        actions = ttk.Frame(left_panel)
        actions.pack(fill=tk.X, pady=(0, 12))
        run_button = ttk.Button(actions, text="Запустить модель", command=self.run_model)
        run_button.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=2)
        reset_button = ttk.Button(actions, text="Сбросить параметры", command=self.reset_parameters)
        reset_button.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        metrics_button = ttk.Button(actions, text="Показатели эффективности", command=self.open_metrics_window)
        metrics_button.grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        adaptive_button = ttk.Button(actions, text="Адаптивный алгоритм", command=self.open_adaptive_results_window)
        adaptive_button.grid(row=0, column=3, sticky="ew", padx=(2, 0), pady=2)
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)

        graph_frame = ttk.LabelFrame(left_panel, text="Графики", padding=12)
        graph_frame.pack(fill=tk.X, pady=(0, 10))
        self.curve_flags = {}
        self.curve_names = [
            ("y_A", "Запас A на складе"),
            ("y_B", "Запас B на складе"),
            ("y_CA", "Запас A в сборке"),
            ("y_CB", "Запас B в сборке"),
            ("G", "Выпуск готовой продукции"),
            ("pA", "Буфер A (D11)"),
            ("pB1", "Буфер B1 (D21)"),
            ("pB2", "Буфер B2 (D22)"),
            ("pB3", "Буфер B3 (D23)"),
            ("fA", "Темп расхода A"),
            ("fB", "Темп расхода B"),
            ("g", "Темп сборки G"),
        ]
        for key, label in self.curve_names:
            var = tk.BooleanVar(value=(key in ["y_A", "y_B", "y_CA", "y_CB", "G"]))
            chk = ttk.Checkbutton(graph_frame, text=label, variable=var)
            chk.pack(anchor=tk.W, pady=1)
            self.curve_flags[key] = var

        info_frame = ttk.LabelFrame(left_panel, text="Описание переменных", padding=12)
        info_frame.pack(fill=tk.BOTH, expand=True)
        self.param_desc_text = tk.Text(info_frame, wrap="word", height=8, borderwidth=0, background="#f8f9fb")
        self.param_desc_text.insert("1.0", "Выберите параметр модели слева, чтобы увидеть его описание.\n")
        self.param_desc_text.configure(state="disabled")
        self.param_desc_text.pack(fill=tk.BOTH, expand=False, pady=(0, 8))

        description = (
            "Состояние (уровень):\n"
            "  y_A, y_B — уровни запасов заготовок A и B на складе;\n"
            "  y_CA, y_CB — уровни деталей A и B в сборочном отделе;\n"
            "  pA, pB1, pB2, pB3 — промежуточные буферы, учитывающие технологические задержки;\n"
            "  G — накопленный выпуск готовой продукции.\n\n"
            "Темпы изменения состояния:\n"
            "  fA, fB — скорость расхода заготовок A и B со склада;\n"
            "  g — скорость сборки готовой продукции из деталей A и B.\n\n"
            "Задержки и взаимодействия:\n"
            "  D11, D21, D22, D23 — длительности технологической обработки (задержки) для потоков;\n"
            "  выходной поток буфера = текущий буфер / D.\n"
            "  Это моделирует накопление и освобождение запаса во времени.\n\n"
            "Другие параметры:\n"
            "  P_A, P_B — мощности линий A и B;\n"
            "  P_C — мощность сборочного отдела;\n"
            "  threshold_critical — уровень, при котором запускается пополнение склада;\n"
            "  threshold_stop — уровень, при котором линия останавливается.\n"
        )
        self.description_text = tk.Text(info_frame, wrap="word", height=10, borderwidth=0, background="#f8f9fb")
        self.description_text.insert("1.0", description)
        self.description_text.configure(state="disabled")
        self.description_text.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(right_panel)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_parameters(self):
        defaults = {
            "T": "100",
            "dt": "1",
            "y_A0": "120",
            "y_B0": "150",
            "y_CA0": "20",
            "y_CB0": "30",
            "y_11_0": "40",
            "P_A": "40",
            "P_B": "60",
            "P_C": "40",
            "D11": "10",
            "D21": "2",
            "D22": "6",
            "D23": "10",
            "X1": "15",
            "threshold_critical": "0.2",
            "threshold_stop": "0.05",
            "seed": "123",
        }
        for name, value in defaults.items():
            self.entries[name].delete(0, tk.END)
            self.entries[name].insert(0, value)

    def get_simulation_parameters(self):
        kwargs = {
            key: float(self.entries[key].get())
            for key in [
                "y_A0",
                "y_B0",
                "y_CA0",
                "y_CB0",
                "y_11_0",
                "P_A",
                "P_B",
                "P_C",
                "D11",
                "D21",
                "D22",
                "D23",
                "X1",
                "threshold_critical",
                "threshold_stop",
            ]
        }
        T = float(self.entries["T"].get())
        dt = float(self.entries["dt"].get())
        seed = int(float(self.entries["seed"].get()))
        return kwargs, T, dt, seed

    def run_model(self):
        try:
            kwargs, T, dt, seed = self.get_simulation_parameters()
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный формат параметров. Введите числовые значения.")
            return

        simulator = ProductionSimulator(**kwargs)
        result = simulator.simulate(T=T, dt=dt, seed=seed)
        self.plot_result(result)

    def open_metrics_window(self):
        try:
            kwargs, T, dt, seed = self.get_simulation_parameters()
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный формат параметров. Введите числовые значения.")
            return
        MetricsWindow(self.root, kwargs, T, dt, seed)

    def open_adaptive_results_window(self):
        try:
            kwargs, T, dt, seed = self.get_simulation_parameters()
        except ValueError:
            messagebox.showerror("Ошибка", "Неверный формат параметров. Введите числовые значения.")
            return
        AdaptiveResultsWindow(self.root, kwargs, T, dt)

    def show_param_description(self, name):
        display_name = self.param_display_names.get(name, name)
        description = self.param_descriptions.get(
            name,
            "Описание для этого параметра отсутствует."
        )
        self.param_desc_text.configure(state="normal")
        self.param_desc_text.delete("1.0", tk.END)
        self.param_desc_text.insert(
            "1.0",
            f"{display_name}\n\nПеременная: {name}\n\n{description}"
        )
        self.param_desc_text.configure(state="disabled")

    def plot_result(self, result):
        self.ax1.clear()
        self.ax2.clear()
        time = result["time"]

        lines1 = []
        lines2 = []
        if self.curve_flags["y_A"].get():
            lines1.append((time, result["y_A"], "blue", "y_A — запас A"))
        if self.curve_flags["y_B"].get():
            lines1.append((time, result["y_B"], "green", "y_B — запас B"))
        if self.curve_flags["y_CA"].get():
            lines1.append((time, result["y_CA"], "red", "y_CA — детали A в сборке"))
        if self.curve_flags["y_CB"].get():
            lines1.append((time, result["y_CB"], "orange", "y_CB — детали B в сборке"))
        if self.curve_flags["pA"].get():
            lines2.append((time, result["pA"], "purple", "pA — буфер A"))
        if self.curve_flags["pB1"].get():
            lines2.append((time, result["pB1"], "brown", "pB1 — буфер B1"))
        if self.curve_flags["pB2"].get():
            lines2.append((time, result["pB2"], "magenta", "pB2 — буфер B2"))
        if self.curve_flags["pB3"].get():
            lines2.append((time, result["pB3"], "cyan", "pB3 — буфер B3"))
        if self.curve_flags["fA"].get():
            lines2.append((time, result["fA"], "navy", "fA — темп расхода A"))
        if self.curve_flags["fB"].get():
            lines2.append((time, result["fB"], "darkgreen", "fB — темп расхода B"))
        if self.curve_flags["g"].get():
            lines2.append((time, result["g"], "black", "g — темп сборки"))
        if self.curve_flags["G"].get():
            lines2.append((time, result["G"], "gray", "G — выпуск готовой продукции"))

        for t, series, color, label in lines1:
            self.ax1.plot(t, series, color=color, label=label)
        self.ax1.set_title("Запасы в системе и сборочном отделе")
        self.ax1.set_xlabel("Время")
        self.ax1.set_ylabel("Уровень")
        self.ax1.grid(True)
        if lines1:
            self.ax1.legend(loc="best")

        for t, series, color, label in lines2:
            self.ax2.plot(t, series, color=color, label=label)
        self.ax2.set_title("Промежуточные буферы и выпуск")
        self.ax2.set_xlabel("Время")
        self.ax2.set_ylabel("Количество")
        self.ax2.grid(True)
        if lines2:
            self.ax2.legend(loc="best")

        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()


class MetricsWindow:
    METRIC_LABELS = {
        "G_total": "Общий выпуск G(T)",
        "q_avg": "Средний темп выпуска",
        "I_avg_A": "Средний запас A",
        "I_avg_B": "Средний запас B",
        "downtime_frac_A": "Доля времени простоя A",
        "downtime_frac_B": "Доля времени простоя B",
        "stop_prob_A": "Вероятность остановки A",
        "stop_prob_B": "Вероятность остановки B",
    }

    PARAMETER_LABELS = {
        "X1": "Внешний поток X1",
        "P_A": "Мощность P_A",
        "P_B": "Мощность P_B",
        "P_C": "Мощность P_C",
        "D11": "Задержка D11",
        "D21": "Задержка D21",
        "D22": "Задержка D22",
        "D23": "Задержка D23",
        "threshold_stop": "Порог остановки",
    }

    def __init__(self, root, base_kwargs, T, dt, seed):
        self.base_kwargs = base_kwargs
        self.T = T
        self.dt = dt
        self.seed = seed
        self.repeats = 10
        self.steps = 7
        self.window = tk.Toplevel(root)
        self.window.title("Показатели эффективности")
        self.window.geometry("900x700")
        self.window.configure(background="#f3f4f6")
        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self.window, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_frame, text="Метрика:", width=16).grid(row=0, column=0, sticky="w", padx=2, pady=4)
        self.metric_var = tk.StringVar(value="G_total")
        metric_menu = ttk.Combobox(top_frame, textvariable=self.metric_var, values=list(self.METRIC_LABELS.keys()), state="readonly", width=24)
        metric_menu.grid(row=0, column=1, sticky="w", padx=2, pady=4)

        ttk.Label(top_frame, text="Параметр по оси X:", width=16).grid(row=1, column=0, sticky="w", padx=2, pady=4)
        self.param_var = tk.StringVar(value="X1")
        param_menu = ttk.Combobox(top_frame, textvariable=self.param_var, values=list(self.PARAMETER_LABELS.keys()), state="readonly", width=24)
        param_menu.grid(row=1, column=1, sticky="w", padx=2, pady=4)

        ttk.Label(top_frame, text="Число точек:", width=16).grid(row=2, column=0, sticky="w", padx=2, pady=4)
        self.points_entry = ttk.Entry(top_frame, width=10)
        self.points_entry.insert(0, "7")
        self.points_entry.grid(row=2, column=1, sticky="w", padx=2, pady=4)

        ttk.Label(top_frame, text="Повторов на точку:", width=16).grid(row=3, column=0, sticky="w", padx=2, pady=4)
        self.repeats_entry = ttk.Entry(top_frame, width=10)
        self.repeats_entry.insert(0, "10")
        self.repeats_entry.grid(row=3, column=1, sticky="w", padx=2, pady=4)

        run_button = ttk.Button(top_frame, text="Построить зависимость", command=self.draw_metric_curve)
        run_button.grid(row=0, column=2, rowspan=2, sticky="ew", padx=8, pady=4)

        info_frame = ttk.Frame(self.window, padding=(10, 0, 10, 10))
        info_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=info_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_metric_curve(self):
        try:
            points = int(self.points_entry.get())
            repeats = int(self.repeats_entry.get())
            if points < 2 or repeats < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Введите целые числа: точки >= 2, повторы >= 1.")
            return

        self.repeats = repeats
        param_name = self.param_var.get()
        metric_name = self.metric_var.get()

        x_values = self._parameter_range(param_name, points)
        y_values = []
        y_stds = []

        for x in x_values:
            metrics = []
            for i in range(repeats):
                run_kwargs = dict(self.base_kwargs)
                run_kwargs[param_name] = x
                sim = ProductionSimulator(**run_kwargs)
                result = sim.simulate(T=self.T, dt=self.dt, seed=self.seed + i)
                metrics.append(sim.calculate_metrics(result)[metric_name])
            y_values.append(float(np.mean(metrics)))
            y_stds.append(float(np.std(metrics)))

        self.ax.clear()
        self.ax.plot(x_values, y_values, marker="o", color="tab:blue", label=self.METRIC_LABELS[metric_name])
        self.ax.fill_between(x_values, np.array(y_values) - np.array(y_stds), np.array(y_values) + np.array(y_stds), color="tab:blue", alpha=0.2)
        self.ax.set_title(f"{self.METRIC_LABELS[metric_name]} vs {self.PARAMETER_LABELS[param_name]}")
        self.ax.set_xlabel(self.PARAMETER_LABELS[param_name])
        self.ax.set_ylabel(self.METRIC_LABELS[metric_name])
        self.ax.grid(True)
        self.ax.legend()
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

    def _parameter_range(self, param_name, points):
        base = float(self.base_kwargs.get(param_name, 1.0))
        if param_name in ["threshold_stop"]:
            low = max(0.01, base * 0.5)
            high = min(0.4, base * 1.5)
        elif param_name in ["D11", "D21", "D22", "D23"]:
            low = max(1.0, base * 0.5)
            high = base * 1.5
        else:
            low = max(1.0, base * 0.5)
            high = base * 1.5
        return np.linspace(low, high, points)


class AdaptiveResultsWindow:    
    METRIC_DISPLAY_NAMES = {
        "G_total": "Общий выпуск G(T)",
        "q_avg": "Средний темп выпуска",
        "I_avg_A": "Средний запас A",
        "I_avg_B": "Средний запас B",
        "downtime_frac_A": "Доля времени простоя A",
        "downtime_frac_B": "Доля времени простоя B",
        "stop_prob_A": "Вероятность остановки A",
        "stop_prob_B": "Вероятность остановки B",
    }

    def __init__(self, root, base_kwargs, T, dt):
        self.base_kwargs = base_kwargs
        self.T = T
        self.dt = dt
        
        self.window = tk.Toplevel(root)
        self.window.title("Адаптивный алгоритм: результаты")
        self.window.geometry("1100x650")
        self.window.configure(background="#f3f4f6")
        
        self.simulator = ProductionSimulator(**base_kwargs)
        self.results = None
        
        self._build_ui()
        self._run_adaptive_algorithm()

    def _build_ui(self):
        """Построить интерфейс окна."""
        # Фрейм для информации и прогресса
        info_frame = ttk.Frame(self.window, padding=10)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(info_frame, text="Статус:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(info_frame, text="Инициализация...", foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Фрейм для таблицы
        table_frame = ttk.Frame(self.window, padding=10)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Scrollbar для таблицы
        scrollbar = ttk.Scrollbar(table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview таблица
        columns = ("Показатель", "Среднее", "Станд. отклонение", "Минимум", "Максимум")
        self.tree = ttk.Treeview(table_frame, columns=columns, height=18, yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.tree.yview)
        
        # Определение столбцов
        self.tree.column("#0", width=0, stretch=tk.NO)
        self.tree.column("Показатель", anchor=tk.W, width=180)
        self.tree.column("Среднее", anchor=tk.CENTER, width=140)
        self.tree.column("Станд. отклонение", anchor=tk.CENTER, width=140)
        self.tree.column("Минимум", anchor=tk.CENTER, width=140)
        self.tree.column("Максимум", anchor=tk.CENTER, width=140)
        
        # Заголовки
        self.tree.heading("#0", text="", anchor=tk.W)
        self.tree.heading("Показатель", text="Показатель", anchor=tk.W)
        self.tree.heading("Среднее", text="Среднее", anchor=tk.CENTER)
        self.tree.heading("Станд. отклонение", text="Станд. отклонение", anchor=tk.CENTER)
        self.tree.heading("Минимум", text="Минимум", anchor=tk.CENTER)
        self.tree.heading("Максимум", text="Максимум", anchor=tk.CENTER)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Фрейм для информации об алгоритме
        info_panel = ttk.LabelFrame(self.window, text="Информация об алгоритме", padding=10)
        info_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.info_text = tk.Text(info_panel, height=4, wrap=tk.WORD, borderwidth=0, background="#f8f9fb")
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.configure(state="disabled")

    def _run_adaptive_algorithm(self):
        """Запустить адаптивный алгоритм в отдельном потоке."""
        import threading
        
        def run():
            def progress_callback(msg):
                self.window.after(0, lambda: self._update_status(msg))
            
            try:
                self.results = self.simulator.adaptive_algorithm(
                    T=self.T,
                    dt=self.dt,
                    quality_metric="G_total",
                    epsilon=0.2,
                    N_initial=50,
                    alpha_level=0.05,
                    max_iterations=10,
                    progress_callback=progress_callback
                )
                self.window.after(0, self._populate_table)
            except Exception as e:
                self.window.after(0, lambda: self._show_error(str(e)))
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def _update_status(self, msg):
        """Обновить статус."""
        self.status_label.config(text=msg, foreground="blue")
        self.window.update_idletasks()

    def _populate_table(self):
        """Заполнить таблицу результатами."""
        if not self.results or "metrics" not in self.results:
            self.status_label.config(text="Ошибка: результаты не получены", foreground="red")
            return
        
        # Очистить таблицу
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        metrics = self.results["metrics"]
        
        # Добавить строки
        for metric_key, metric_data in metrics.items():
            display_name = self.METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
            values = (
                display_name,
                f"{metric_data['mean']:.6f}",
                f"{metric_data['std']:.6f}",
                f"{metric_data['min']:.6f}",
                f"{metric_data['max']:.6f}",
            )
            self.tree.insert("", "end", values=values)
        
        # Обновить информацию об алгоритме
        info = (
            f"Статус: {self.results['status']} | "
            f"Число реализаций: N={self.results['N']} | "
            f"Итерации адаптации: {self.results['iterations']} | "
            f"Показатель качества: {self.results['quality_metric']} | "
            f"Относительная ошибка: ε={self.results['epsilon']} | "
            f"Уровень значимости: α={self.results['alpha_level']}"
        )
        
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", info)
        self.info_text.configure(state="disabled")
        
        self.status_label.config(text="Вычисления завершены", foreground="green")

    def _show_error(self, error_msg):
        """Показать ошибку."""
        self.status_label.config(text=f"Ошибка: {error_msg}", foreground="red")
        messagebox.showerror("Ошибка", f"Ошибка при выполнении алгоритма:\n{error_msg}")
