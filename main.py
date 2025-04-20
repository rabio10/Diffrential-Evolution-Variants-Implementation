import numpy as np

population_size = 5
problem_dimension = 2
Stopping_Criteria = Maximum_Num_iterations = 2
Scaling_Factor = 0.5
Crossover_Probability = 0.7

def function_evaluation(point):
    """
    point : is a tuple of (x,y)
    """
    return np.square(point[0]) + np.square(point[1])

def initialize_pop(pop_size):
    pop = []
    for i in range(pop_size):
        x = np.random.uniform(-5,5,2)
        pop.append(x)

    return pop
