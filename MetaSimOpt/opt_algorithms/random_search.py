import math
from MetaSimOpt.utils.data_utils import _sort_features
from .utils import initialize_population

class RandomSearch:

    def __init__(self, hyper):

        self.population_size = hyper.get('population_size', 25)
        self.n_iterations = hyper.get('n_iterations', 500)
        self.mod_handler = hyper.get('mod_handler', None)
        self.early_stop = math.ceil(hyper.get('early_stop', 0.25) * self.n_iterations)

        if hasattr(self.mod_handler.metamodel, "compute_lengths"):
            self.solution_length = int(self.mod_handler.metamodel.compute_lengths(self.mod_handler.x[0])[0])
        else:
            self.solution_length = 50

        self.population = []
        self.best_solution = None
        self.best_fitness = float("inf")
        self.no_improve_count = 0
        self.best_fitness_per_iteration = []
        self.all_fitness_per_iteration = []


    def evolve(self):

        self.population = initialize_population(population_size=self.population_size, solution_length=self.solution_length)
        data = _sort_features(idx = self.population, x = self.mod_handler.x)
        fitness_scores = self.mod_handler.predict(x = data).squeeze()
        sorted_pairs = sorted(zip(fitness_scores, self.population), key=lambda x: x[0])
        sorted_fitness_scores, sorted_population = zip(*sorted_pairs)

        # Save best solution if improved
        current_best_fitness = sorted_fitness_scores[0]
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = sorted_population[0]
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        return fitness_scores


    def run(self, print_prog = True):

        for iteration in range(self.n_iterations):
            fitness_scores = self.evolve()
            self.best_fitness_per_iteration.append(self.best_fitness)
            self.all_fitness_per_iteration.append(fitness_scores.tolist())

            if print_prog:
                print(f"Iteration {iteration + 1} | Best fitness: {self.best_fitness:.2f} | No improvement: {self.no_improve_count}")

            if self.no_improve_count >= self.early_stop:
                print(f"Early stopping triggered at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_fitness