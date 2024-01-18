"""
@File: GA.py
@author: Zijia Zhao
@Describtion: 
"""
import numpy as np

class Individual:
    def __init__(self, action_size, action_bound):
        self.action_size = action_size
        self.action_bound = action_bound
        self.actions = np.random.uniform(-action_bound, action_bound, action_size)

    def mutate(self, mutation_rate=0.1):
        mutation = np.random.normal(0, 1, self.action_size) * mutation_rate
        self.actions = np.clip(self.actions + mutation, -self.action_bound, self.action_bound)

    def crossover(self, other):
        child = Individual(self.action_size, self.action_bound)
        for i in range(self.action_size):
            child.actions[i] = np.random.choice([self.actions[i], other.actions[i]])
        return child

class Population:
    def __init__(self, size, action_size, action_bound):
        self.individuals = [Individual(action_size, action_bound) for _ in range(size)]

    def evolve(self, get_fitness, mutation_rate=0.1, crossover_rate=0.7):
        new_population = []
        fitness_scores = [get_fitness(ind) for ind in self.individuals]
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]

        for _ in range(len(self.individuals)//2):
            parents = np.random.choice(self.individuals, 2, p=probabilities)
            if np.random.rand() < crossover_rate:
                child1 = parents[0].crossover(parents[1])
                child2 = parents[1].crossover(parents[0])
            else:
                child1, child2 = parents

            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            new_population.extend([child1, child2])

        self.individuals = new_population
