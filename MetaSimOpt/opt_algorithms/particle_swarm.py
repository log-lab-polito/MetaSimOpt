import numpy as np
import random
import math
from MetaSimOpt.utils.data_utils import _sort_features
from .utils import initialize_population

class PartSwarmOpt:

    def __init__(self, hyper):

        self.population_size = hyper.get('population_size', 20)
        self.n_iterations = hyper.get('n_iterations', 500)
        self.w = hyper.get('w', 0.5)
        self.c1 = hyper.get('c1', 0.5)
        self.c2 = hyper.get('c2', 0.5)    
        self.mod_handler = hyper.get('mod_handler', None)
        self.early_stop = math.ceil(hyper.get('early_stop', 0.25) * self.n_iterations)

        if hasattr(self.mod_handler.metamodel, "compute_lengths"):
            self.solution_length = int(self.mod_handler.metamodel.compute_lengths(self.mod_handler.x[0])[0])
        else:
            self.solution_length = 50

        self.population = initialize_population(population_size=self.population_size, solution_length=self.solution_length)
        self.particles = [self.create_particle(sol) for sol in self.population]
        self.best_solution = None
        self.best_fitness = float("inf")
        self.no_improve_count = 0
        self.best_fitness_per_iteration = []
        self.all_fitness_per_iteration = []

    
    def create_particle(self, sequence):
        velocity = []
        n_swaps = self.solution_length
        for _ in range(n_swaps):
            first_pos = 0
            second_pos = 0
            while (first_pos == second_pos):
                first_pos = random.randint(0, self.solution_length-1)
                second_pos = random.randint(0, self.solution_length-1)
            swap = (first_pos, second_pos, 'w')
            velocity.append(swap)
        
        return {'sequence': sequence, 'fitness' : np.inf, 'best_sequence': [el for el in sequence], 'best_fitness' : np.inf, 'velocity': velocity}


    def update_velocities(self):

        for particle in self.particles:
            temp_velocity = particle['velocity']
            particle['velocity'] = []
            temp_sequence_1 = [el for el in particle['sequence']] # for personal best
            temp_sequence_2 = [el for el in particle['sequence']] # for global best

            # personal velocity
            for swap in temp_velocity:
                swap = (swap[0],swap[1])
                if random.random() <= self.w*random.random():
                    particle['velocity'].append(swap)

            # personal best
            for i in range(len(particle['sequence'])): 
                if temp_sequence_1[i] != particle['best_sequence'][i]:
                    swap = (i, particle['best_sequence'].index(temp_sequence_1[i]))
                    if random.random() <= self.c1*random.random():
                        particle['velocity'].append(swap)
                    temp_sequence_1[swap[0]], temp_sequence_1[swap[1]] = temp_sequence_1[swap[1]], temp_sequence_1[swap[0]]
            
            # global best
            for i in range(len(particle['sequence'])):
                if self.best_solution is not None and temp_sequence_2[i] != self.best_solution[i]:
                    index = int(np.where(self.best_solution == temp_sequence_2[i])[0][0])
                    swap = (i, index)
                    if random.random() <= self.c2*random.random():
                        particle['velocity'].append(swap)
                    temp_sequence_2[swap[0]], temp_sequence_2[swap[1]] = temp_sequence_2[swap[1]], temp_sequence_2[swap[0]]


    def update_positions(self):
        pop = []
        for particle in self.particles:
            for swap in particle['velocity']:
                particle['sequence'][swap[0]], particle['sequence'][swap[1]] = particle['sequence'][swap[1]], particle['sequence'][swap[0]]
            pop.append(particle['sequence'])
        self.population = np.array(pop)


    def evolve(self):

        # update particles velocity position
        self.update_velocities()
        self.update_positions()

        # calculate fitness and update best_sequence and best_fitness for each particle
        data = _sort_features(idx = self.population, x = self.mod_handler.x)
        fitness_scores = self.mod_handler.predict(x = data).squeeze()
        for particle, fitness_value in zip(self.particles, fitness_scores):
            particle['fitness'] = fitness_value
            if particle['fitness'] <= particle['best_fitness']:
                particle['best_sequence'] = [el for el in particle['sequence']]
                particle['best_fitness'] = particle['fitness']

        # select best particle
        sorted_particles = sorted(self.particles, key=lambda pp: pp['fitness'])
        gbest_particle = sorted_particles[0]

        # Check for improvement in the best fitness
        if sorted_particles[0]['fitness'] < self.best_fitness:
            gbest_particle = sorted_particles[0]
            self.best_solution = gbest_particle['sequence']
            self.best_fitness = gbest_particle['fitness']
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
    