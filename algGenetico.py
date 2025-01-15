import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class AlgGenetico:
    def __init__(self, delta, min, max, iteration, population, crossover_rate, mutation_rate, x):
        self.x = x
        self.delta = delta
        self.intervalo = [min, max]
        self.iteration = iteration
        self.population = population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = []
        self.offspring = []
        self.n_points = int((max - min) / delta) + 1
        self.bits = math.ceil(math.log2(self.n_points))
        self.grid = [min + i * delta for i in range(self.n_points)]

    def fx(self, x):
        return 0.1 * math.log10(1 + abs(x)) * math.cos(x) ** 2

    def binary_to_decimal(self, binary):
        return int(binary, 2)

    def decimal_to_binary(self, decimal):
        return f"{decimal:0{self.bits}b}"

    def decode_individual(self, binary):
        index = self.binary_to_decimal(binary)
        return self.grid[min(index, len(self.grid) - 1)]

    def initialize_population(self):
        self.population = [self.decimal_to_binary(random.randint(0, self.n_points - 1)) for _ in range(self.population)]
        self.fitness = [self.fx(self.decode_individual(individual)) for individual in self.population]

    def select_parents(self):
        total_fitness = sum(self.fitness)
        probabilities = [fit / total_fitness for fit in self.fitness]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.bits - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            bit = random.randint(0, self.bits - 1)
            mutated = list(individual)
            mutated[bit] = '1' if individual[bit] == '0' else '0'
            return ''.join(mutated)
        return individual

    def evolve(self):
        new_population = []
        for _ in range(len(self.population) // 2):
            parent1, parent2 = self.select_parents()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])

        self.population = new_population[:len(self.population)]
        self.fitness = [self.fx(self.decode_individual(individual)) for individual in self.population]

def init_alg_gen(delta, interval, iteration, population, crossover_rate, mutation_rate, plot_ax):
    x_values = np.arange(interval[0], interval[1] + delta, delta)
    y_values = [0.1 * math.log10(1 + abs(x)) * math.cos(x) ** 2 for x in x_values]

    plot_ax.clear()
    plot_ax.plot(x_values, y_values, label="f(x)", color="blue")
    plot_ax.set_title("Algoritmo Genético y f(x)")
    plot_ax.set_xlabel("X")
    plot_ax.set_ylabel("f(x)")
    plot_ax.legend()

    canvas.draw()

    alg_gen = AlgGenetico(delta, interval[0], interval[1], iteration, population, crossover_rate, mutation_rate, 0)
    alg_gen.initialize_population()

    for _ in range(iteration):
        alg_gen.evolve()
        
        best_individual = alg_gen.population[np.argmax(alg_gen.fitness)]
        best_x = alg_gen.decode_individual(best_individual)
        best_y = alg_gen.fx(best_x)
        
        plot_ax.plot(best_x, best_y, 'rx')  
        canvas.draw()

def start_algorithm():
    delta = float(delta_entry.get())
    interval = (float(interval_min_entry.get()), float(interval_max_entry.get()))
    iteration = int(iteration_entry.get())
    population = int(population_entry.get())
    crossover_rate = float(crossover_rate_entry.get())
    mutation_rate = float(mutation_rate_entry.get())

    init_alg_gen(delta, interval, iteration, population, crossover_rate, mutation_rate, ax)

root = tk.Tk()
root.title("Algoritmo Genetico - 223200")
root.geometry("800x400")

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
]

entries = {}
for i, (label, var_name) in enumerate(fields):
    ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
    entry = ttk.Entry(main_frame, width=20)
    entry.grid(row=i, column=1, pady=2)
    entries[var_name] = entry

(delta_entry, interval_min_entry, interval_max_entry, iteration_entry,
 population_entry, crossover_rate_entry, mutation_rate_entry) = entries.values()

start_button = ttk.Button(main_frame, text="Iniciar", command=start_algorithm)
start_button.grid(row=len(fields), column=0, columnspan=2, pady=30)


fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

root.mainloop()


