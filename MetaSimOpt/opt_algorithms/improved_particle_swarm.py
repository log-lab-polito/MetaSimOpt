import numpy as np
import random
import math
from MetaSimOpt.utils.data_utils import _sort_features
from .mutation_methods import swap_mutation, scramble_mutation, inverse_mutation
from .utils import initialize_population

class ImprovPartSwarmOpt:

    def __init__(self, hyper):

        self.population_size = hyper.get('population_size', 20)
        self.n_iterations = hyper.get('n_iterations', 500)
        self.w = hyper.get('w', 0.1)
        self.c1 = hyper.get('c1', 0.5)
        self.c2 = hyper.get('c2', 0.5)
        self.c3 = hyper.get('c3', 0.5)
        self.mutation_rate = hyper.get('mutation_rate', 0.1)
        self.mutation_method = hyper.get('mutation_method', 'swap')
        self.stag_max = hyper.get('stag_max', 10)
        self.alpha = hyper.get('alpha', 0.5)
        self.beta = hyper.get('beta', 0.8)          
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
        self.worst_solution = None
        self.worst_fitness = float("inf")
        self.no_improve_count = 0
        self.n_elites = math.floor(self.alpha * self.population_size)
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

        mu = {
            'mu_c1' : random.random(),
            'mu_c3' : random.random(),
            'mu_c2' : random.random()
            }

        coefficients = {
                'w'  : 1,
                'c1' : 1,
                'c2' : 1,
                'c3' : 1
            }
        
        return {'sequence': sequence, 'fitness' : np.inf, 'best_sequence': [el for el in sequence], 'best_fitness' : np.inf, 'best_local': [el for el in sequence], 'forgetting' : 0, 'velocity': velocity, 'mu' : mu, 'coef': coefficients}


    def update_velocities(self):

        for particle in self.particles:
            temp_velocity = particle['velocity']
            particle['velocity'] = []
            temp_sequence_1 = [el for el in particle['sequence']] # for personal best
            temp_sequence_2 = [el for el in particle['sequence']] # for local best
            temp_sequence_3 = [el for el in particle['sequence']] # for global best

            # personal velocity
            for swap in temp_velocity:
                swap = (swap[0],swap[1],'w')
                if random.random() <= self.w:
                    particle['velocity'].append(swap)

            # personal best
            for i in range(len(particle['sequence'])): 
                if temp_sequence_1[i] != particle['best_sequence'][i]:
                    swap = (i, particle['best_sequence'].index(temp_sequence_1[i]))
                    if random.random() <= self.c1 * particle['coef']['c1']:
                        particle['velocity'].append(swap)
                    temp_sequence_1[swap[0]], temp_sequence_1[swap[1]] = temp_sequence_1[swap[1]], temp_sequence_1[swap[0]]

            # global best
            swaps = []
            for i in range(len(particle['sequence'])):
                if self.best_solution is not None and temp_sequence_3[i] != self.best_solution[i]:
                    index = int(np.where(self.best_solution == temp_sequence_3[i])[0][0])
                    swap = (i, index)
                    swaps.append(swap)
                    temp_sequence_3[swap[0]], temp_sequence_3[swap[1]] = temp_sequence_3[swap[1]], temp_sequence_3[swap[0]]
            
            n_swaps = len(swaps)
            new_n_swaps = math.floor(n_swaps*(abs(1-particle['forgetting'])))
            if new_n_swaps < n_swaps:
                for j in range(new_n_swaps):
                    if random.random() <= self.c2 * particle['coef']['c2']:
                        particle['velocity'].append(swaps[j])  
            else:
                i = 0
                while new_n_swaps:
                    if i == n_swaps:
                        i = 0
                    if random.random() <= self.c2 * particle['coef']['c2']:
                        particle['velocity'].append(swaps[i]) 
                    i += 1
                    new_n_swaps -= 1

            # local best
            swaps = []
            for i in range(len(particle['sequence'])): 
                if temp_sequence_2[i] != particle['best_local'][i]:
                    index = int(np.where(particle['best_local'] == temp_sequence_2[i])[0][0])
                    swap = (i, index)
                    swaps.append(swap)
                    temp_sequence_2[swap[0]], temp_sequence_2[swap[1]] = temp_sequence_2[swap[1]], temp_sequence_2[swap[0]]
                
            n_swaps = len(swaps)
            new_n_swaps = math.floor(n_swaps*(abs(1-particle['forgetting'])))
            if new_n_swaps < n_swaps:
                for j in range(new_n_swaps):
                    if random.random() <= self.c3 * particle['coef']['c3']:
                        particle['velocity'].append(swaps[j])
            else:
                i = 0
                while new_n_swaps:
                    if i == n_swaps:
                        i = 0
                    if random.random() <= self.c3 * particle['coef']['c3']:
                        particle['velocity'].append(swaps[i]) 
                    i += 1
                    new_n_swaps -= 1
            

    def update_positions(self):
        pop = []
        for particle in self.particles:
            for swap in particle['velocity']:
                particle['sequence'][swap[0]], particle['sequence'][swap[1]] = particle['sequence'][swap[1]], particle['sequence'][swap[0]]
            pop.append(particle['sequence'])
        self.population = np.array(pop)


    def update_forgetting_ability(self):

        for part in self.particles:
            part['dist'] = part['fitness'] - self.best_fitness

        sorted_particles = sorted(self.particles, key=lambda pp: pp['dist'])
        n_part = len(self.particles)
        threshold = math.floor(n_part * self.beta)

        forg_abilities = []

        for i in range(n_part):
            if i+1 < threshold:
                forg_abilities.append(0)
            else:
                forg_abilities.append((math.e**((i+1)/n_part))*(self.worst_fitness-self.best_fitness)*0.01)

        for i, part in enumerate(sorted_particles):
            part['forgetting'] = forg_abilities[i]


    def normalise_list(self,lst):

        min_val = min(lst)
        max_val = max(lst)
        normalised = [(x - min_val) / (max_val - min_val) for x in lst]
        return normalised


    def update_coefficients(self, mu_k_elite, iter = 1):

        c1s = []
        c2s = []
        c3s = []
            
        for particle in self.particles:
            particle['mu']['mu_c1'] = self.beta * particle['mu']['mu_c1'] + (1-self.beta) * (np.mean([k['mu_c1'] for k in mu_k_elite]))
            particle['mu']['mu_c2'] = self.beta * particle['mu']['mu_c2'] + (1-self.beta) * (np.mean([k['mu_c2'] for k in mu_k_elite]))
            particle['mu']['mu_c3'] = self.beta * particle['mu']['mu_c3'] + (1-self.beta) * (np.mean([k['mu_c3'] for k in mu_k_elite]))

            c1s.append(np.random.normal(particle['mu']['mu_c1'],0.1))
            c2s.append(np.random.normal(particle['mu']['mu_c2'],0.1))
            c3s.append(np.random.normal(particle['mu']['mu_c3'],0.1))

        c1s = self.normalise_list(c1s)
        c2s = self.normalise_list(c2s)
        c3s = self.normalise_list(c3s)

        for i, p in enumerate(self.particles):
            p['coef']['w'] = 1
            p['coef']['c1'] = c1s[i]
            p['coef']['c2'] = c2s[i]
            p['coef']['c3'] = c3s[i]


    def update_neighbours(self):
        
        random.shuffle(self.particles)

        for i, particle in enumerate(self.particles):
            neighbours = []
            if i == 0: # first element
                neighbours.append(self.particles[-1])
                neighbours.append(self.particles[0])
                neighbours.append(self.particles[1])
            elif i == len(self.particles)-1: # last element
                neighbours.append(self.particles[-2])
                neighbours.append(self.particles[-1])
                neighbours.append(self.particles[0])
            else: # other elements
                neighbours.append(self.particles[i-1])
                neighbours.append(self.particles[i])
                neighbours.append(self.particles[i+1])

            lbest_particle = min(neighbours, key=lambda pp: pp['fitness'])
            particle['best_local'] = lbest_particle['sequence']
    

    def update(self, sorted_particles, iter):
        elites = [part for part in sorted_particles[:self.n_elites]]
        mu_elite_particles = [part['mu'] for part in elites]
        
        # update particles coefficients
        self.update_forgetting_ability()
        self.update_coefficients(mu_elite_particles, iter)
        self.update_neighbours()


    def mutate(self):
        if self.mutation_method == "swap":
            return swap_mutation(self.population, self.mutation_rate)
        elif self.mutation_method == "scramble":
            return scramble_mutation(self.population, self.mutation_rate)
        elif self.mutation_method == "inverse":
            return inverse_mutation(self.population, self.mutation_rate)
        else:
            raise NotImplementedError(f"Mutation method '{self.mutation_method}' not implemented")


    def evolve(self):

        # update particles velocity position
        self.update_velocities()
        self.update_positions()
        
        # Mutation
        self.population = np.array(self.mutate())
        for i,individual in enumerate(self.particles):
            individual['sequence'] = self.population[i]

        # calculate fitness and update best_sequence and best_fitness for each particle
        data = _sort_features(idx = self.population, x = self.mod_handler.x)
        fitness_scores = self.mod_handler.predict(x = data).squeeze()
        for particle, fitness_value in zip(self.particles, fitness_scores):
            particle['fitness'] = fitness_value
            if particle['fitness'] <= particle['best_fitness']:
                particle['best_sequence'] = [el for el in particle['sequence']]
                particle['best_fitness'] = particle['fitness']

        # select elites particles
        sorted_particles = sorted(self.particles, key=lambda pp: pp['fitness'])
        
        gbest_particle = sorted_particles[0]
        gworst_particle = sorted_particles[-1]
        self.worst_solution = gworst_particle['sequence']
        self.worst_fitness = gworst_particle['fitness']

        # Check for improvement in the best fitness
        if sorted_particles[0]['fitness'] < self.best_fitness:
            gbest_particle = sorted_particles[0]
            self.best_solution = gbest_particle['sequence']
            self.best_fitness = gbest_particle['fitness']
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        return fitness_scores, sorted_particles


    def run(self, print_prog = True):
        for iteration in range(self.n_iterations):
            fitness_scores, sorted_particles = self.evolve()
            self.best_fitness_per_iteration.append(self.best_fitness)
            self.all_fitness_per_iteration.append(fitness_scores.tolist())

            if self.no_improve_count >= self.stag_max or iteration == 0:
                self.update(sorted_particles, iteration)

            if print_prog:
                print(f"Iteration {iteration + 1} | Best fitness: {self.best_fitness:.2f} | No improvement: {self.no_improve_count}")

            if self.no_improve_count >= self.early_stop:
                print(f"Early stopping triggered at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_fitness


