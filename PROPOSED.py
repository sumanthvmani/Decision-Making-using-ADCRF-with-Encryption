import time
import numpy as np


# (EPFOA) Enhanced Piranha Foraging Optimization Algorithm
def PROPOSED(population, fitness_function, lb, ub, max_iter):
    # Proposed position update is done at line 28
    population_size, dimension = population.shape
    # Initialize parameters
    M = 0.5  # Probability of hunger
    N = 0.75  # Probability for blood concentration impact
    C = 5  # Constant parameter
    G = 9  # Coefficient of foraging ability
    fitness = np.apply_along_axis(fitness_function, 1, population)
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]
    convergence = np.zeros((max_iter))
    # Iterative process
    ct = time.time()
    for iteration in range(max_iter):
        for i in range(population_size):
            if np.random.rand() < M:
                # Non-linear cosine factor
                S = C * np.cos(np.pi * iteration / max_iter)
                # Proposed position update is done here
                # E = np.random.rand()
                E = fitness[i] / (np.max(fitness) + np.min(fitness))
                if E < N:
                    # Localized group attack pattern
                    prey_position = best_solution
                    random_indices = np.random.choice(population_size, size=population_size // 2, replace=False)
                    local_group = population[random_indices]
                    local_update = np.sum(local_group - prey_position, axis=0)
                    population[i] = prey_position + S * local_update
                else:
                    # Bloodthirsty cluster attack pattern
                    prey_position = best_solution
                    F = G / (np.linalg.norm(prey_position - population[i]) ** 2 + 1e-9)
                    movement = np.random.uniform(-2, 2, size=dimension) * F
                    population[i] = prey_position + S * movement
            else:
                # Scavenging foraging pattern
                random_agent = population[np.random.randint(0, population_size)]
                direction = np.random.uniform(-1, 1, size=dimension)
                population[i] = random_agent + direction

            # Boundary handling
            population[i] = np.clip(population[i], lb, ub)

        # Update fitness and best solution
        fitness = np.apply_along_axis(fitness_function, 1, population)
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]
        convergence[iteration] = best_fitness
    ct = time.time() - ct
    return best_fitness, convergence, best_solution, ct
