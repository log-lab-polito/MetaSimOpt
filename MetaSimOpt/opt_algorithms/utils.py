import random
import numpy as np

def initialize_population(population_size, solution_length, shuffle = True):
    pop = []
    while len(pop) < population_size:
        individual = list(range(solution_length))
        if shuffle:
            random.shuffle(individual)
            if individual not in pop:
                pop.append(individual)
        else:
            pop.append(individual)
    return np.array(pop)