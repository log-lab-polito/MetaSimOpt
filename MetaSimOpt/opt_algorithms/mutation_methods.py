import random

def swap_mutation(population, mutation_rate):
    mutated = []
    for individual in population:
        individual = individual.copy()
        if random.random() < mutation_rate:
            pos1 = random.randint(0, len(individual) - 1)
            pos2 = random.randint(0, len(individual) - 1)
            individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
        mutated.append(individual)
    return mutated


def scramble_mutation(population, mutation_rate):
    mutated = []
    for individual in population:
        individual = individual.copy()
        if random.random() < mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            subsequence = individual[start:end]
            random.shuffle(subsequence)
            individual[start:end] = subsequence
        mutated.append(individual)
    return mutated


def inverse_mutation(population, mutation_rate):
    mutated = []
    for individual in population:
        individual = individual.copy()
        if random.random() < mutation_rate:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end] = individual[start:end][::-1]
        mutated.append(individual)
    return mutated