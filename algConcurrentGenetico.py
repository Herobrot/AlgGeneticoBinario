import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading


class AlgGenetico:
    def __init__(
        self,
        delta,
        min,
        max,
        iteration,
        pop_max,
        pop_min,
        crossover_rate,
        mutation_rate,
        bit_mutattion_rate,
        x,
    ):
        self.x = x
        self.delta = delta
        self.intervalo = [min, max]
        self.iteration = iteration
        self.population = []
        self.pop_max = pop_max
        self.pop_min = pop_min
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.bit_mutation_rate = bit_mutattion_rate
        self.fitness = []
        self.offspring = []
        self.n_points = math.ceil((max - min) / delta) + 1
        self.bits = math.ceil(math.log2(self.n_points))
        self.grid = [min + i * delta for i in range(self.n_points)]
        self.best_solution = None
        self.best_fitness = float("-inf")
        self.best_x = None
        self.worse_solution = None
        self.worse_fitness = float("inf")
        self.lock = threading.Lock()
        self.delta_system = (self.intervalo[1] - self.intervalo[0]) / (2**self.bits - 1)

        self.temp_population = []
        self.temp_fitness = []

    def fx(self, x):
        return 0.1 * x * math.log(1 + abs(x)) * math.cos(x) ** 2

    def binary_to_decimal(self, binary):
        return int(binary, 2)

    def decimal_to_binary(self, decimal):
        return f"{decimal:0{self.bits}b}"

    def decode_individual(self, binary):
        index = self.binary_to_decimal(binary)
        return self.intervalo[0] + index * self.delta_system

    def initialize_population(self):
        with ThreadPoolExecutor() as executor:
            self.population = list(
                executor.map(
                    lambda _: self.decimal_to_binary(
                        random.randint(0, self.n_points - 1)
                    ),
                    range(self.pop_max),
                )
            )
            self.fitness = list(
                executor.map(
                    lambda individual: self.fx(self.decode_individual(individual)),
                    self.population,
                )
            )

    def adjust_population_size(self, iteration):
        target_size = int(
            self.pop_min + (self.pop_max - self.pop_min) * 
            ((self.iteration - iteration) / self.iteration)
        )
        
        with self.lock:
            if len(self.population) > target_size:
    
                sorted_pairs = sorted(
                    zip(self.population, self.fitness),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.population, self.fitness = zip(*sorted_pairs[:target_size])
                self.population = list(self.population)
                self.fitness = list(self.fitness)

    def process_pair(self, _):
        with self.lock:
            parent1, parent2 = self.select_parents()
        
        child1, child2 = self.crossover(parent1, parent2)
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)
        return child1, child2

    def select_parents(self):
        total_fitness = sum(self.fitness)
        probabilities = [fit / total_fitness for fit in self.fitness]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents.copy()

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.bits - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            mutated = list(individual)
            for bit in range(len(mutated)):
                if random.random() < self.bit_mutation_rate:
                    mutated[bit] = "1" if mutated[bit] == "0" else "0"
            return "".join(mutated)
        return individual

    def evolve(self, current_iteration):
        with ThreadPoolExecutor() as executor:
            pairs = list(executor.map(self.process_pair, range(len(self.population) // 2)))
            new_population = [child for pair in pairs for child in pair]        

            new_fitness = list(
                executor.map(
                    lambda individual: self.fx(self.decode_individual(individual)),
                    new_population
                )
            )

        with self.lock:
            self.population = new_population[:len(self.population)]
            self.fitness = new_fitness[:len(self.population)]
            
            current_best_idx = np.argmax(self.fitness)
            current_best_fitness = self.fitness[current_best_idx]

            current_worse_idx = np.argmin(self.fitness)
            current_worse_fitness = self.fitness[current_worse_idx]

            if current_worse_fitness < self.worse_fitness:
                self.worse_fitness = current_worse_fitness
                self.worse_solution = self.population[current_worse_idx]
                self.worse_x = self.decode_individual(self.worse_solution)
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[current_best_idx]
                self.best_x = self.decode_individual(self.best_solution)

        self.adjust_population_size(current_iteration)


def init_alg_gen(
    delta,
    interval,
    iteration,
    pop_max,
    pop_min,
    crossover_rate,
    mutation_rate,
    bit_mutation_rate,
    plot_ax,
    best_fx_label,
    best_x_label,
    fitness_ax,
):
    with ThreadPoolExecutor() as executor:
        x_values = np.arange(interval[0], interval[1] + delta, delta)
        y_values = list(
            executor.map(
                lambda x: 0.1 * x * math.log(1 + abs(x)) * math.cos(x) ** 2, x_values
            )
        )

    plot_ax.clear()
    plot_ax.plot(x_values, y_values, label="f(x)", color="blue")
    plot_ax.set_title("Evolución de la Población")
    plot_ax.set_xlabel("X")
    plot_ax.set_ylabel("f(x)")
    plot_ax.legend()

    fitness_ax.clear()
    fitness_ax.set_title("Evolución de la Aptitud")
    fitness_ax.set_xlabel("Iteraciones")
    fitness_ax.set_ylabel("Aptitud")

    canvas.draw()

    alg_gen = AlgGenetico(
        delta,
        interval[0],
        interval[1],
        iteration,
        pop_max,
        pop_min,
        crossover_rate,
        mutation_rate,
        bit_mutation_rate,
        0,
    )
    alg_gen.initialize_population()

    best_fitness_history = []
    worst_fitness_history = []
    avg_fitness_history = []

    def update_gui():
        nonlocal best_fitness_history, worst_fitness_history, avg_fitness_history

        current_best = max(alg_gen.fitness)
        current_worst = min(alg_gen.fitness)
        current_avg = sum(alg_gen.fitness) / len(alg_gen.fitness)

        best_fitness_history.append(current_best)
        worst_fitness_history.append(current_worst)
        avg_fitness_history.append(current_avg)

        plot_ax.clear()
        plot_ax.plot(x_values, y_values, label="f(x)", color="blue")
        plot_ax.set_title("Algoritmo Genético y f(x)")
        plot_ax.set_xlabel("X")
        plot_ax.set_ylabel("f(x)")

        for ind, fit in zip(alg_gen.population, alg_gen.fitness):
            x = alg_gen.decode_individual(ind)
            y = fit
            plot_ax.plot(x, y, "kx")
        
        if alg_gen.best_x is not None:
            plot_ax.plot(alg_gen.best_x, alg_gen.best_fitness, "mx")

        if alg_gen.worse_x is not None:
            plot_ax.plot(alg_gen.worse_x, alg_gen.worse_fitness, "rx")

        plot_ax.legend()

        fitness_ax.clear()
        fitness_ax.set_title("Evolución de la Aptitud")
        fitness_ax.set_xlabel("Iteraciones")
        fitness_ax.set_ylabel("Aptitud")

        iterations = range(1, len(best_fitness_history) + 1)
        fitness_ax.plot(iterations, best_fitness_history, label="Mejor", color="green")
        fitness_ax.plot(iterations, worst_fitness_history, label="Peor", color="red")
        fitness_ax.plot(iterations, avg_fitness_history, label="Promedio", color="blue")
        fitness_ax.legend()

        best_x_label.configure(
            text=f"Mejor x: {alg_gen.best_x:.6f}", 
            style=header_type
        )
        best_fx_label.configure(
            text=f"Mejor f(x): {alg_gen.best_fitness:.6f}", 
            style=header_type
        )
        delta_system_label.configure(
            text=f"Delta del sistema: {alg_gen.delta_system:.6f}", 
            style=header_type
        )
        num_points_label.configure(
            text=f"Cantidad de puntos: {alg_gen.n_points}", 
            style=header_type
        )
        num_bits_label.configure(
            text=f"Cantidad de bits: {alg_gen.bits}", 
            style=header_type
        )
        string_bits_label.configure(
            text=f"Cadena de bits: {alg_gen.best_solution}", 
            style=header_type
        )

        canvas.draw()

    def evolution_process():
        for _ in range(iteration):
            alg_gen.evolve(iteration)
            root.after(0, update_gui)
        root.after(0, lambda: start_button.config(state="normal"))

    start_button.config(state="disabled")
    threading.Thread(target=evolution_process, daemon=True).start()


def start_algorithm():
    try:
        delta = float(delta_entry.get())
        interval = (float(interval_min_entry.get()), float(interval_max_entry.get()))
        iteration = int(iteration_entry.get())
        pop_max = int(population_max_entry.get())
        pop_min = int(population_min_entry.get())
        crossover_rate = float(crossover_rate_entry.get())
        mutation_rate = float(mutation_rate_entry.get())
        bit_mutation_rate = float(bit_mutation_rate_entry.get())

        init_alg_gen(
            delta,
            interval,
            iteration,
            pop_max,
            pop_min,
            crossover_rate,
            mutation_rate,
            bit_mutation_rate,
            ax,
            best_fx_label,
            best_x_label,
            fitness_ax,
        )
    except ValueError as _:
        tk.messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos")


# Configuración inicial de la ventana
root = tk.Tk()
root.title("Algoritmo Genetico Concurrente - 223200")
root.geometry("1200x900")  # Aumentado el ancho para acomodar ambas columnas

# Crear un frame contenedor principal
container_frame = ttk.Frame(root, padding="10")
container_frame.grid(row=0, column=0, sticky='nsew')

# Configurar los pesos de las columnas y filas del root
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Crear el frame para los controles (izquierda)
main_frame = ttk.Frame(container_frame, padding="10")
main_frame.grid(row=0, column=0, sticky='nsew')

# Configurar los pesos de las columnas del container_frame
container_frame.grid_columnconfigure(0, weight=0)  # columna de controles
container_frame.grid_columnconfigure(1, weight=1)  # columna de gráficas

# Crear los estilos
style = ttk.Style()
header_type = 'Header.TLabel'
style.configure('Large.TLabel', font=('Helvetica', 12))
style.configure(header_type, font=('Helvetica', 14, 'bold'))
style.configure('Large.TButton', font=('Helvetica', 12))

# Campos de entrada
fields = [
    ("Delta", "delta_entry"),
    ("Intervalo Min", "interval_min_entry"),
    ("Intervalo Max", "interval_max_entry"),
    ("Iteraciones", "iteration_entry"),
    ("Población Max", "population_max_entry"),
    ("Población Min", "population_min_entry"),
    ("Crossover Rate", "crossover_rate_entry"),
    ("Mutation Rate", "mutation_rate_entry"),
    ("Probabilidad por bit", "bit_mutation_rate_entry"),
]

entries = {}
for i, (label, var_name) in enumerate(fields):
    ttk.Label(main_frame, text=label, style='Large.TLabel').grid(
        row=i, column=0, sticky=tk.W, pady=5, padx=10
    )
    entry = ttk.Entry(main_frame, width=25, font=('Helvetica', 11))
    entry.grid(row=i, column=1, pady=5, padx=10)
    entries[var_name] = entry

# Desempaquetar las entradas
(
    delta_entry,
    interval_min_entry,
    interval_max_entry,
    iteration_entry,
    population_max_entry,
    population_min_entry,
    crossover_rate_entry,
    mutation_rate_entry,
    bit_mutation_rate_entry,
) = entries.values()

# Botón de inicio
start_button = ttk.Button(
    main_frame, 
    text="Iniciar", 
    command=start_algorithm, 
    style='Large.TButton'
)
start_button.grid(
    row=len(fields), 
    column=0, 
    columnspan=2, 
    pady=25,
    padx=10,
    sticky='ew'
)

# Labels de resultados
best_x_label = ttk.Label(main_frame)
best_fx_label = ttk.Label(main_frame)
delta_system_label = ttk.Label(main_frame)
num_points_label = ttk.Label(main_frame)
num_bits_label = ttk.Label(main_frame)
string_bits_label = ttk.Label(main_frame)

result_labels = [
    (best_x_label, "Mejor x: ---"),
    (best_fx_label, "Mejor f(x): ---"),
    (delta_system_label, "Delta del sistema: ---"),
    (num_points_label, "Cantidad de puntos: ---"),
    (num_bits_label, "Cantidad de bits: ---"),
    (string_bits_label, "Cadena de bits: ---")
]

for i, (label, text) in enumerate(result_labels):
    label.configure(
        text=text, 
        style=header_type,
        padding=(10, 5)
    )
    label.grid(
        row=len(fields) + 1 + i, 
        column=0, 
        columnspan=2, 
        sticky=tk.W, 
        pady=8,
        padx=10
    )

# Configurar las gráficas
fig, (ax, fitness_ax) = plt.subplots(2, 1, figsize=(8, 8))
fig.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.4)

# Crear y posicionar el canvas de matplotlib
canvas = FigureCanvasTkAgg(fig, master=container_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)

root.mainloop()
