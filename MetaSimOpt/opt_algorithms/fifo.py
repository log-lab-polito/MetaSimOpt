import numpy as np
import math
from MetaSimOpt.utils.data_utils import _sort_features
from .utils import initialize_population

class Fifo:
    def __init__(self, hyper):

        self.population_size = hyper.get('population_size', 1)
        self.n_iterations = hyper.get('n_iterations', 500)
        self.mod_handler = hyper.get('mod_handler', None)
        self.early_stop = math.ceil(hyper.get('early_stop', 0.25) * self.n_iterations)

        if hasattr(self.mod_handler.metamodel, "compute_lengths"):
            self.solution_length = int(self.mod_handler.metamodel.compute_lengths(self.mod_handler.x[0]))
        
        self.population = initialize_population(population_size=self.population_size, solution_length=self.solution_length, shuffle=False)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.no_improve_count = 0
        self.best_fitness_per_iteration = []
        self.all_fitness_per_iteration = []


    def run(self, print_prog = True):
        data = _sort_features(idx = self.population, x = self.mod_handler.x)
        fitness_scores = self.mod_handler.predict(x = data).squeeze()
        self.best_solution = self.population[0]
        self.best_fitness = fitness_scores

        self.best_fitness_per_iteration = [fitness_scores for _ in range(self.n_iterations)]
        self.all_fitness_per_iteration = [fitness_scores for _ in range(self.n_iterations)]

        return self.best_solution, self.best_fitness
