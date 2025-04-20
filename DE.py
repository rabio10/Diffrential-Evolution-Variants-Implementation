import numpy as np
import matplotlib.pyplot as plt


class DE:
    def __init__(self, pop_size, prob_dimension, scaling_factor,Pcr, variant="DE" ,target_vector_selection_strategy="rand", number_differentials=1):
        """
        variant : "DE" , "CODE", ....
        target_vector_selection_strategy : "rand" , "best" , "current-to-best" , "current-to-rand"
        number_differentials : 1 or 2
        """
        self.population_size = pop_size
        self.problem_dimension = prob_dimension
        self.Maximum_Num_iterations = 0
        self.Scaling_Factor = scaling_factor
        self.Crossover_Probability = Pcr
        self.pop = []
        self.target_vector_selection_strategy = target_vector_selection_strategy
        self.number_of_differentials = number_differentials
        self.variant = variant

    
    def best_in_pop(self):
        evals = []
        for vect in self.pop:
            ev = function_evaluation(vect)
            evals.append(ev)
        best_value = min(evals)
        index_best_value = evals.index(best_value)
        return best_value, index_best_value, evals

    def initialize_pop_custom(self):
        pop = [
            np.array([1.7667, -4.1337]),
            np.array([4.8071, 0.6642]),
            np.array([-2.9232, -4.0439]),
            np.array([-4.3747, -4.7421]),
            np.array([-1.6587, 0.5680])
        ]
        self.pop = pop
        return pop

    def initialize_pop(self):
        pop = []
        for i in range(self.population_size):
            x = np.random.uniform(-5,5,self.problem_dimension)
            pop.append(x)
        self.pop = pop
        return pop
    
    def current_to_best_mutation(self, vect_i):
        """
        returns the trial vector
        """
        # the best in pop
        index_of_selected_best = self.best_in_pop()[1]
        # generte two randomly
        x1 = -1
        x2 = -1
        while x1 == x2 or x1 == index_of_selected_best or x2 == index_of_selected_best:
            x1 = np.random.randint(0,self.population_size)
            x2 = np.random.randint(0,self.population_size)
        # calculate the trial vvector
        trial_vector = vect_i + self.Scaling_Factor * ( self.pop[index_of_selected_best] - vect_i ) + self.Scaling_Factor * ( self.pop[x1] - self.pop[x2] )

        return trial_vector
    
    def current_to_rand_mutation(self, vect_i):
        # generate random scaling factor
        rand_F = np.random.randint(0,1)
        # generate 3 randomly
        x1 = -1
        x2 = -1
        x3 = -1
        while x1 == x2 or x1 == x3 or x2 == x3:
            x1 = np.random.randint(0,self.population_size)
            x2 = np.random.randint(0,self.population_size)
            x3 = np.random.randint(0,self.population_size)
        # calculate the trial vector
        trial_vector = vect_i + rand_F * (self.pop[x1] - vect_i) + self.Scaling_Factor * (self.pop[x2] - self.pop[x3])
        
        return trial_vector

    def rand_mutation(self, strategy, num_diff):
        """
        4 possible schemes : rand/1 , best/1 , rand/2 , best/2 
        strategy : "rand" or "best"
        num_diff : 1 or 2
        """
        # compute trial_vector depending on strategy : rand/1 , best/1 , rand/2 , best/2
        if strategy == "rand":
            index_of_selected = np.random.randint(0,self.population_size)
        elif strategy == "best":
            index_of_selected = self.best_in_pop()[1]
        
        target_vector = self.pop[index_of_selected]

        # select two/four vectors randomely (insure mutually diffrent)
        x1 = -1
        x2 = -1
        while x1 == x2 or x1 == index_of_selected or x2 == index_of_selected:
            x1 = np.random.randint(0,self.population_size)
            x2 = np.random.randint(0,self.population_size)

        # mutate
        trial_vector = target_vector + self.Scaling_Factor * (self.pop[x1] - self.pop[x2])

        if num_diff == 2:
            x3 = -1
            x4 = -1
            while x3 == x4 or x3 == x2 or x3 == x1 or x4 == x1 or x4 == x2 or x3 == index_of_selected or x4 == index_of_selected:
                x3 = np.random.randint(0,self.population_size)
                x4 = np.random.randint(0,self.population_size)
            
            # add to trial vector
            trial_vector += self.Scaling_Factor * (self.pop[x3] - self.pop[x4])
        
        return trial_vector

    def boundary_verification(self, vector):
        if vector[0] < -5:
            vector[0] = -5
        elif vector[0] > 5:
            vector[0] = 5
        if vector[1] < -5:
            vector[1] = -5
        elif vector[1] > 5:
            vector[1] = 5
        return vector
    
    def mutation(self, vect_i):
        """
        target_vector_selection_strategy : it can be "rand" or "best"
        """
        # compute trial_vector depending on strategy : current-to-best/1 , current-to-rand/1
        if self.target_vector_selection_strategy == "current-to-best":
            # fucntion call
            trial_vector = self.current_to_best_mutation(vect_i)

        elif self.target_vector_selection_strategy == "current-to-rand":
            # function call
            trial_vector = self.current_to_rand_mutation(vect_i)
        else:
            # compute trial_vector depending on strategy : rand/1 , best/1 , rand/2 , best/2
            trial_vector = self.rand_mutation(self.target_vector_selection_strategy, self.number_of_differentials)

        if trial_vector[0] < -5:
            trial_vector[0] = -5
        elif trial_vector[0] > 5:
            trial_vector[0] = 5
        if trial_vector[1] < -5:
            trial_vector[1] = -5
        elif trial_vector[1] > 5:
            trial_vector[1] = 5

        return trial_vector

    def crossover(self, parent_vector, trial_vector):
        """
        returns the new vector , either is new or juste parent vector
        """
        is_crossover = np.random.random()
        if self.Crossover_Probability < is_crossover:
            new_vect = parent_vector.copy() 
            # decide n and L 
            # n: index of first element to take from the trial_vector
            # L: length of elements taken from trial_vector
            while(True):
                n = np.random.randint(0, self.problem_dimension)
                L = np.random.randint(1, self.problem_dimension+1)
                if self.problem_dimension - n >= L:
                    break
            
            for i in range(L):
                new_vect[n+i] = trial_vector[n+i]
            # decide who's best (to live or die)
            new_vect_evaluation = function_evaluation(new_vect)
            parent_eval = function_evaluation(parent_vector)
            if new_vect_evaluation < parent_eval:
                return new_vect
            else:
                return parent_vector
        else:
            return parent_vector
            
    def next_generation(self):
        new_gen = []
        if self.variant == "DE":
            for parent_vect in self.pop:
                trial_vect = self.mutation(parent_vect)
                new_vect = self.crossover(parent_vect, trial_vect)
                new_gen.append(new_vect)
            self.pop = new_gen
        
        elif self.variant == "CODE":
            # select parametrs randomly
            parameter_candidate_pool = [(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)] # (F , CR)
            strategies = [("rand", 1), ("rand", 2), ("current-to-rand", 1)]
            for parent_vect in self.pop:
                new_vects = []
                evals_of_new_vects = []
                # do the 3 strategies
                for st in strategies:
                    selected_params = parameter_candidate_pool[np.random.randint(0,3)]
                    self.Scaling_Factor = selected_params[0]
                    self.Crossover_Probability = selected_params[1]
                    # do mutation in 3 schemes
                    self.target_vector_selection_strategy = st[0]
                    self.number_of_differentials = st[1]
                    trial_vect = self.mutation(parent_vect)
                    # crossover
                    new_vect = self.crossover(parent_vect, trial_vect)
                    ev = function_evaluation(new_vect)
                    # add to list
                    new_vects.append(new_vect)
                    evals_of_new_vects.append(ev)
                # take the best of the three
                evals_of_new_vects = np.array(evals_of_new_vects)
                index_of_best_of_three = np.argmin(evals_of_new_vects)
                # append to the new gen
                new_gen.append(new_vects[index_of_best_of_three])

            self.pop = new_gen
                
        return new_gen
    

    def do_evolution(self, num_generations, verbose=False):
        self.Maximum_Num_iterations = num_generations
        self.initialize_pop()
        evals_of_generations = []
        if self.variant == "JDE":
            # the JDE works on classic DE with adaptif parametrs
            self.variant = "DE"
            new_CR = self.Crossover_Probability
            new_F = self.Scaling_Factor
            F_l = 0.1
            F_u = 0.9
            tau = 0.1
            for i in range(num_generations):
                best = self.best_in_pop()
                if verbose:
                    print(f"Gen {i}, the best : {self.pop[best[1]]} with f = {best[0]}")
                    print(np.array(self.pop))
                    print(np.array(self.best_in_pop()[2]))
                evals_of_generations.append(best[0])
                self.next_generation()
                # new F 
                rand = [np.random.uniform() for i in range(4)]
                if rand[1] < tau:
                    new_F = F_l + rand[0] * F_u
                else:
                    new_F = self.Scaling_Factor
                self.Scaling_Factor = new_F
                # new CR
                if rand[3] < tau:
                    new_CR = rand[2]
                else:
                    new_CR = self.Crossover_Probability
                self.Scaling_Factor = new_CR
        
        else:
            for i in range(num_generations):
                best = self.best_in_pop()
                if verbose:
                    print(f"Gen {i}, the best : {self.pop[best[1]]} with f = {best[0]}")
                    print(np.array(self.pop))
                    print(np.array(self.best_in_pop()[2]))
                evals_of_generations.append(best[0])
                self.next_generation()

        return evals_of_generations



def function_evaluation(point):
        """
        point : is a tuple of (x,y)
        """
        return np.square(point[0]) + np.square(point[1])
