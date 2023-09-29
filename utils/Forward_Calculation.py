import numpy as np
import sympy as sp

class VNetwork:
    C = 0.8
    n = 0.5

    def __init__(self,pin:float,pout:float,node_num:int):
        self.pin = pin
        self.pout = pout
        self.node_num = node_num
        self.population_size = 100
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.population = np.random.rand(self.population_size,node_num) * 20 - 10
        pass

    def flow(self,p1,p2):
        result = self.C * np.abs(p1 - p2) ** self.n * np.sign(p1 - p2)
        return result
    
    def objective(self,population):
        Qin = self.flow(self.pin,population[:,0])
        Qout = self.flow(population[:,-1],self.pout)
        Q12 = self.flow(population[:,0],population[:,1])
        Q23 = self.flow(population[:,1],population[:,2])
        Q13 = self.flow(population[:,0],population[:,2])

        eq1 = np.abs(Qin + Q12 + Q13)
        eq2 = np.abs(Q12 + Q23)
        eq3 = np.abs(Q13 + Q23 + Qout)

        result = eq1 + eq2 + eq3
        return result
    
    def select_parent(self,population):
        fitness = self.objective(population)
        fitness_prob = 1/(1+fitness)
        fitness_prob = fitness_prob/np.sum(fitness_prob)
        parent_index = np.random.choice(np.arange(self.population_size),size=self.population_size // 2,p=fitness_prob)
        return parent_index
    
    def crossover(self,population):
        offspring = []
        parent_index = self.select_parent(population)
        parent = self.population[parent_index]
        for i in range(0,len(parent),2):
            parent1 = parent[i]
            parent2 = parent[i+1]
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1,self.node_num)
                child1 = np.concatenate((parent1[:crossover_point],parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point],parent1[crossover_point:]))
            else:
                child1 = parent1
                child2 = parent2
            offspring.extend([child1,child2])
        offspring = np.array(offspring)
        return offspring
    
    def mutation(self,population):
        offspring = self.crossover(population)
        for i in range(len(offspring)):
            if np.random.rand() < self.mutation_rate:
                mutation_point = np.random.randint(self.node_num)
                offspring[i][mutation_point] += np.random.rand() * 2
        return offspring
    
    def main(self):
        offspring = self.mutation(self.population)
        offspring_fitness = self.objective(offspring)
        population_fitness = self.objective(self.population)
        combined_population = np.concatenate((self.population,offspring))
        combined_fitness = np.concatenate((population_fitness,offspring_fitness))
        next_population_index = np.argsort(combined_fitness)[:self.population_size]
        fitness = combined_fitness[next_population_index]
        self.population = combined_population[next_population_index]
        print('Best fitness:',fitness[0])
