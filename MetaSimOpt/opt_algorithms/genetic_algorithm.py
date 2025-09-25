import numpy as np
import random
import math
from MetaSimOpt.utils.data_utils import _sort_features
from .mutation_methods import swap_mutation, scramble_mutation, inverse_mutation
from .utils import initialize_population


class GeneticAlgorithm:

    @staticmethod
    def roulette_wheel_selection(population, fitness_scores, num_selected):
        max_score = np.max(fitness_scores)
        fitness_scores_norm = max_score - np.array(fitness_scores)
        total_fitness = np.sum(fitness_scores_norm)

        if total_fitness == 0:
            probabilities = np.full(len(fitness_scores), 1 / len(fitness_scores))
        else:
            probabilities = fitness_scores_norm / total_fitness

        selected_indices = np.random.choice(len(population), size=num_selected, p=probabilities)
        selected_individuals = [population[i] for i in selected_indices]
        selected_fitnesses = [fitness_scores[i] for i in selected_indices]

        return selected_individuals, selected_fitnesses 


    @staticmethod
    def tournament_selection(population, fitness_scores, num_selected, tournament_size=3):
        selected_individuals = []
        selected_fitnesses = []
        pop_fitness = list(zip(population, fitness_scores))
        for _ in range(num_selected):
            competitors = random.sample(pop_fitness, tournament_size)
            winner = min(competitors, key=lambda x: x[1])  # winner is (individual, fitness)
            selected_individuals.append(winner[0])
            selected_fitnesses.append(winner[1])
        return selected_individuals, selected_fitnesses


    @staticmethod
    def pmx_crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(1, size - 2), 2))

        child1 = parent1[:start] + parent2[start:end] + parent1[end:]
        child2 = parent2[:start] + parent1[start:end] + parent2[end:]

        mapping1 = {parent2[i]: parent1[i] for i in range(start, end)}
        mapping2 = {parent1[i]: parent2[i] for i in range(start, end)}

        for i in list(range(start)) + list(range(end, size)):
            while child1[i] in mapping1:
                child1[i] = mapping1[child1[i]]
            while child2[i] in mapping2:
                child2[i] = mapping2[child2[i]]

        return child1, child2


    @staticmethod
    def order_crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        # Initialize children with None
        child1 = [None] * size
        child2 = [None] * size

        # Copy slice from parents
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        # Fill the remaining positions with order from the other parent
        def fill_remaining(child, parent):
            current_pos = end
            for gene in list(parent[end:]) + list(parent[:end]):
                if gene not in child:
                    if current_pos >= size:
                        current_pos = 0
                    child[current_pos] = gene
                    current_pos += 1
            return child

        child1 = fill_remaining(child1, parent2)
        child2 = fill_remaining(child2, parent1)

        return child1, child2

    def __init__(self, hyper):
        
        self.population_size = hyper.get('population_size', 25)
        self.n_iterations = hyper.get('n_iterations', 500)
        self.crossover_rate = hyper.get('crossover_rate', 0.8)
        self.elite_size = hyper.get('elite_size', 2)
        self.mutation_rate = hyper.get('mutation_rate', 0.1)
        self.selection_method = hyper.get('selection_method', "roulette_wheel")
        self.crossover_method = hyper.get('crossover_method', "ox")
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


    def selection(self, population, fitness_scores):
        num_selected = len(population) - self.elite_size

        if self.selection_method == "roulette_wheel":
            return GeneticAlgorithm.roulette_wheel_selection(population, fitness_scores, num_selected)
        elif self.selection_method == "tournament":
            return GeneticAlgorithm.tournament_selection(population, fitness_scores, num_selected)
        else:
            raise NotImplementedError(f"Selection method '{self.selection_method}' not implemented")


    def crossover(self, parent1, parent2):
        if self.crossover_method == "pmx":
            return GeneticAlgorithm.pmx_crossover(parent1, parent2)
        elif self.crossover_method == "ox":
            return GeneticAlgorithm.order_crossover(parent1, parent2)
        else:
            raise NotImplementedError(f"Crossover method '{self.crossover_method}' not implemented")
    

    def mutate(self, offspring):
        if self.mutation_method == "swap":
            return swap_mutation(offspring, self.mutation_rate)
        elif self.mutation_method == "scramble":
            return scramble_mutation(offspring, self.mutation_rate)
        elif self.mutation_method == "inverse":
            return inverse_mutation(offspring, self.mutation_rate)
        else:
            raise NotImplementedError(f"Mutation method '{self.mutation_method}' not implemented")


    def evolve(self):
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

        # Elitism
        elite = list(sorted_population[:self.elite_size])
        selected, _ = self.selection(sorted_population, sorted_fitness_scores)

        # Crossover
        offspring = []
        while len(offspring) < self.population_size - self.elite_size:
            parent1, parent2 = random.sample(selected, 2)
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()  # No crossover, parents copied directly
            offspring.extend([child1, child2])

        # Mutation
        offspring = self.mutate(offspring)

        # New generation
        self.population = np.array(elite + offspring)

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