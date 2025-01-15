import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
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

    def fx(self, x):
        return 0.1*math.log10(1+abs(x))*math.cos(x)*math.cos(x)
    
    def selection(self, population):
        return 0
    
def initAlgGen(delta, interval, iteration, population, crossover_rate, mutation_rate, fitness, plot_ax):
    algGenetico = AlgGenetico(delta, interval[0], interval[1], iteration, population, crossover_rate, mutation_rate, fitness)
    plot_ax.clear()
    plot_ax.set_title("Evolución del Algoritmo Genético")
    plot_ax.set_xlabel("Iteraciones")
    plot_ax.set_ylabel("Fitness")

    x_vals = []
    y_vals = []

    for i in range(iteration):
        fitness_value = fitness + random.uniform(-delta, delta)
        x_vals.append(i)
        y_vals.append(fitness_value)

        plot_ax.plot(x_vals, y_vals, 'bo-', label="Fitness" if i == 0 else "")
        plot_ax.legend()
        canvas.draw()

        time.sleep(0.1)  # Simulación de tiempo entre iteraciones

def start_algorithm():
    delta = float(delta_entry.get())
    interval = (float(interval_min_entry.get()), float(interval_max_entry.get()))
    iteration = int(iteration_entry.get())
    population = int(population_entry.get())
    crossover_rate = float(crossover_rate_entry.get())
    mutation_rate = float(mutation_rate_entry.get())
    fitness = float(fitness_entry.get())

    initAlgGen(delta, interval, iteration, population, crossover_rate, mutation_rate, fitness, ax)

root = tk.Tk()
root.title("Algoritmo Genetico - 223200")
root.geometry("800x600")

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
    ("Fitness Inicial", "fitness_entry"),
]

entries = {}
for i, (label, var_name) in enumerate(fields):
    ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
    entry = ttk.Entry(main_frame, width=20)
    entry.grid(row=i, column=1, pady=2)
    entries[var_name] = entry

(delta_entry, interval_min_entry, interval_max_entry, iteration_entry,
 population_entry, crossover_rate_entry, mutation_rate_entry,
 fitness_entry, offspring_entry) = entries.values()

start_button = ttk.Button(main_frame, text="Iniciar", command=start_algorithm)
start_button.grid(row=len(fields), column=0, columnspan=2, pady=10)


fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

root.mainloop()


