import numpy as np
import math
from MetaSimOpt.utils.data_utils import _sort_features
from .mutation_methods import swap_mutation, scramble_mutation, inverse_mutation
from .utils import initialize_population

    
class MultiStartSimulatedAnnealing:

    def __init__(self, hyper):
        self.population_size = hyper.get('population_size', 25)
        self.n_iterations = hyper.get('n_iterations', 500)
        self.temperature = hyper.get('temperature', 50)
        self.cooling_rate = hyper.get('cooling_rate', 0.9)
        self.mutation_rate = hyper.get('mutation_rate', 0.1)
        self.mutation_method = hyper.get('mutation_method', "swap")        
        self.mod_handler = hyper.get('mod_handler', None)
        self.early_stop = math.ceil(hyper.get('early_stop', 0.25) * self.n_iterations)

        if hasattr(self.mod_handler.metamodel, "compute_lengths"):
            self.solution_length = int(self.mod_handler.metamodel.compute_lengths(self.mod_handler.x[0])[0])
        else:
            self.solution_length = 50

        self.population = initialize_population(population_size=self.population_size, solution_length=self.solution_length)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.no_improve_count = 0
        self.best_fitness_per_iteration = []
        self.all_fitness_per_iteration = []


    def mutate(self):
        if self.mutation_method == "swap":
            return swap_mutation(self.population, self.mutation_rate)
        elif self.mutation_method == "scramble":
            return scramble_mutation(self.population, self.mutation_rate)
        elif self.mutation_method == "inverse":
            return inverse_mutation(self.population, self.mutation_rate)
        else:
            raise NotImplementedError(f"Mutation method '{self.mutation_method}' not implemented")
        

    def acceptance_probability(self, fitness_scores, new_fitness_scores, new_population):
        updated_fitness_scores = fitness_scores.copy()
        updated_population = self.population.copy()

        # create masks
        better_mask = new_fitness_scores < fitness_scores
        worse_mask = ~better_mask

        # calculate probabilities for the worse
        delta = fitness_scores[worse_mask] - new_fitness_scores[worse_mask]
        acceptance_probs = np.exp(np.clip(delta / self.temperature, -700, 0))
        random_values = np.random.rand(np.sum(worse_mask))
        accept_worse = random_values < acceptance_probs
        worse_indices = np.where(worse_mask)[0]
        accepted_worse_indices = worse_indices[accept_worse]

        # update
        updated_fitness_scores[better_mask] = new_fitness_scores[better_mask]
        updated_population[better_mask] = new_population[better_mask]
        updated_fitness_scores[accepted_worse_indices] = new_fitness_scores[accepted_worse_indices]
        updated_population[accepted_worse_indices] = new_population[accepted_worse_indices]

        return updated_fitness_scores, updated_population


    def annealing_chain(self,fitness_scores):

        new_population = np.array(self.mutate())
        data = _sort_features(idx = new_population, x = self.mod_handler.x)
        new_fitness_scores = self.mod_handler.predict(x = data).squeeze()

        best_idx = np.argmin(new_fitness_scores)
        current_best_fitness = new_fitness_scores[best_idx]
        current_best_solution = new_population[best_idx]

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = current_best_solution
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        
        fitness_scores, self.population = self.acceptance_probability(fitness_scores, new_fitness_scores, new_population)

        # update temperature
        self.temperature = max(self.temperature * self.cooling_rate, 1e-8)

        return fitness_scores


    def run(self, print_prog = True):
        
        data = _sort_features(idx = self.population, x = self.mod_handler.x)
        fitness_scores = self.mod_handler.predict(x = data).squeeze()

        for iteration in range(self.n_iterations):
            fitness_scores = self.annealing_chain(fitness_scores)
            self.best_fitness_per_iteration.append(self.best_fitness)
            self.all_fitness_per_iteration.append(fitness_scores.tolist())

            if print_prog:
                print(f"Iteration {iteration + 1} | Best fitness: {self.best_fitness:.2f} | No improvement: {self.no_improve_count}")

            if self.no_improve_count >= self.early_stop:
                print(f"Early stopping triggered at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_fitness