import numpy as np
import time


def FSA(population, fitness_function, lb, ub, max_iter):
    # Flamingo Search Algorithm (FSA)
    pop_size, dim = population.shape
    mp_ratio = 0.3

    fitness = np.array([fitness_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]
    convergence = np.zeros((max_iter))
    ct = time.time()
    for t in range(max_iter):
        rand_val = np.random.rand()
        mp1 = int(mp_ratio * pop_size)
        mp2 = int(rand_val * mp1 * (1 - mp_ratio))
        mp3 = pop_size - mp1 - mp2

        # Update positions for the first group
        for i in range(mp1):
            for j in range(dim):
                population[i][j] = population[i][j] + np.random.rand() * (best_individual[j] - population[i][j])  # the position update using equation (3)

        # Update positions for the second group
        for i in range(mp1, mp1 + mp2):
            for j in range(dim):
                population[i][j] = population[i][j] + np.random.rand() * (best_individual[j] - population[i][j])  # the position update using equation (2)

        # Update positions for the third group
        for i in range(mp1 + mp2, pop_size):
            for j in range(dim):
                population[i][j] = population[i][j] + np.random.rand() * (best_individual[j] - population[i][j])  # the position update using equation (3)

        # Boundary detection
        population = np.clip(population, lb, ub)

        # Calculate new fitness values and find the best individual
        fitness = np.array([fitness_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_individual = population[best_idx]
        convergence[t] = best_fitness
    ct = time.time() - ct
    return best_fitness, convergence, best_individual, ct