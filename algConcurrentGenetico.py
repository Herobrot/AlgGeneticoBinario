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
        population,
        crossover_rate,
        mutation_rate,
        bit_mutattion_rate,
        x,
    ):
        self.x = x
        self.delta = delta
        self.intervalo = [min, max]
        self.iteration = iteration
        self.population = population
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
        self.lock = threading.Lock()
        self.delta_system = 0

    def fx(self, x):
        return 0.1 * x * math.log(1 + abs(x)) * math.cos(x) ** 2

    def binary_to_decimal(self, binary):
        return int(binary, 2)

    def decimal_to_binary(self, decimal):
        return f"{decimal:0{self.bits}b}"

    def decode_individual(self, binary):
        delta_system = self.calculate_delta_system()
        index = self.binary_to_decimal(binary)
        return self.intervalo[0] + index * delta_system  # A + i * deltaSystem

    def calculate_delta_system(self):
        return (self.intervalo[1] - self.intervalo[0]) / (2**self.bits - 1)

    def initialize_population(self):
        with ThreadPoolExecutor() as executor:
            self.population = list(
                executor.map(
                    lambda _: self.decimal_to_binary(
                        random.randint(0, self.n_points - 1)
                    ),
                    range(self.population),
                )
            )
            self.fitness = list(
                executor.map(
                    lambda individual: self.fx(self.decode_individual(individual)),
                    self.population,
                )
            )

    def select_parents(self):
        with self.lock:
            total_fitness = sum(self.fitness)
            probabilities = [fit / total_fitness for fit in self.fitness]
            parents = random.choices(self.population, weights=probabilities, k=2)
            return parents

    def process_pair(self, _):
        parent1, parent2 = self.select_parents()
        child1, child2 = self.crossover(parent1, parent2)
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)
        return child1, child2

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

    def evolve(self):
        with ThreadPoolExecutor() as executor:
            pairs = list(
                executor.map(self.process_pair, range(len(self.population) // 2))
            )

            new_population = [child for pair in pairs for child in pair]

            self.population = new_population[: len(self.population)]

            self.fitness = list(
                executor.map(
                    lambda individual: self.fx(self.decode_individual(individual)),
                    self.population,
                )
            )
            current_best = max(self.fitness)
            if current_best > self.best_fitness:
                self.best_fitness = current_best
                self.best_solution = self.population[self.fitness.index(current_best)]


def init_alg_gen(
    delta,
    interval,
    iteration,
    population,
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
    plot_ax.set_title("Algoritmo Genético y f(x)")
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
        population,
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

        best_individual = alg_gen.population[np.argmax(alg_gen.fitness)]
        best_x = (alg_gen.decode_individual(best_individual))
        best_y = alg_gen.fx(best_x)

        plot_ax.plot(best_x, best_y, "rx")
        
        fitness_ax.clear()
        fitness_ax.set_title("Evolución de la Aptitud")
        fitness_ax.set_xlabel("Iteraciones")
        fitness_ax.set_ylabel("Aptitud")
        
        iterations = range(1, len(best_fitness_history) + 1)
        fitness_ax.plot(iterations, best_fitness_history, label="Mejor", color="green")
        fitness_ax.plot(iterations, worst_fitness_history, label="Peor", color="red")
        fitness_ax.plot(iterations, avg_fitness_history, label="Promedio", color="blue")
        fitness_ax.legend()

        best_x_label.config(text=f"Mejor x: {best_x}")
        best_fx_label.config(text=f"Mejor f(x): {best_y:.4f}")
        delta_system_label.config(text=f"Delta del sistema: {alg_gen.calculate_delta_system():.4f}")
        num_points_label.config(text=f"Cantidad de puntos: {alg_gen.n_points}")
        num_bits_label.config(text=f"Cantidad de bits: {alg_gen.bits}")
        string_bits_label.config(text=f"Cadena de bits: {best_individual}")

        canvas.draw()

    def evolution_process():
        for _ in range(iteration):
            alg_gen.evolve()
            # Programar la actualización de GUI en el hilo principal
            root.after(0, update_gui)
        root.after(0, lambda: start_button.config(state="normal"))

    start_button.config(state="disabled")
    threading.Thread(target=evolution_process, daemon=True).start()


def start_algorithm():
    try:
        delta = float(delta_entry.get())
        interval = (float(interval_min_entry.get()), float(interval_max_entry.get()))
        iteration = int(iteration_entry.get())
        population = int(population_entry.get())
        crossover_rate = float(crossover_rate_entry.get())
        mutation_rate = float(mutation_rate_entry.get())
        bit_mutation_rate = float(bit_mutation_rate_entry.get())

        init_alg_gen(
            delta,
            interval,
            iteration,
            population,
            crossover_rate,
            mutation_rate,
            bit_mutation_rate,
            ax,
            best_fx_label,
            best_x_label,
            fitness_ax
        )
    except ValueError as _:
        tk.messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos")


root = tk.Tk()
root.title("Algoritmo Genetico Concurrente - 223200")
root.geometry("900x900")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

fields = [
    ("Delta", "delta_entry"),
    ("Intervalo Min", "interval_min_entry"),
    ("Intervalo Max", "interval_max_entry"),
    ("Iteraciones", "iteration_entry"),
    ("Población", "population_entry"),
    ("Crossover Rate", "crossover_rate_entry"),
    ("Mutation Rate", "mutation_rate_entry"),
    ("Probabilidad por bit", "bit_mutation_rate_entry"),
]

entries = {}
for i, (label, var_name) in enumerate(fields):
    ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
    entry = ttk.Entry(main_frame, width=20)
    entry.grid(row=i, column=1, pady=2)
    entries[var_name] = entry

(
    delta_entry,
    interval_min_entry,
    interval_max_entry,
    iteration_entry,
    population_entry,
    crossover_rate_entry,
    mutation_rate_entry,
    bit_mutation_rate_entry,
) = entries.values()

start_button = ttk.Button(main_frame, text="Iniciar", command=start_algorithm)
start_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

best_x_label = ttk.Label(main_frame, text="Mejor x: ---")
best_x_label.grid(row=len(fields) + 1, column=0, columnspan=2, sticky=tk.W, pady=5)

best_fx_label = ttk.Label(main_frame, text="Mejor f(x): ---")
best_fx_label.grid(row=len(fields) + 2, column=0, columnspan=2, sticky=tk.W, pady=5)

delta_system_label = ttk.Label(main_frame, text="Delta del sistema: ---")
delta_system_label.grid(
    row=len(fields) + 3, column=0, columnspan=2, sticky=tk.W, pady=5
)

num_points_label = ttk.Label(main_frame, text="Cantidad de puntos: ---")
num_points_label.grid(row=len(fields) + 4, column=0, columnspan=2, sticky=tk.W, pady=5)

num_bits_label = ttk.Label(main_frame, text="Cantidad de bits: ---")
num_bits_label.grid(row=len(fields) + 5, column=0, columnspan=2, sticky=tk.W, pady=5)

string_bits_label = ttk.Label(main_frame, text="Cadena de bits: ---")
string_bits_label.grid(row=len(fields) + 6, column=0, columnspan=2, sticky=tk.W, pady=5)


fig, (ax, fitness_ax) = plt.subplots(2, 1, figsize=(6, 4))
fig.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

root.mainloop()
