from DE import *

if __name__ == "__main__":
    num_generations = 100
    evals = []
    # DE/rand/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="rand", number_differentials=1)
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"DE/rand/1"))
    # DE/best/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="best", number_differentials=1)
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"DE/best/1"))
    # DE/rand/2
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="rand", number_differentials=2)
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"DE/rand/2"))
    # DE/best/2
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="best", number_differentials=2)
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"DE/best/2"))
    # DE/current-to-best/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="current-to-best", number_differentials=1)
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"DE/current-to-best/1"))
    # DE/current-to-rand/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="current-to-rand", number_differentials=1)
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"DE/current-to-rand/1"))
    # CODE
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="CODE")
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"CODE"))
    # JDE
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="JDE")
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"JDE"))
    # JADE
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="JADE")
    evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"JADE"))

    only_opt_evals = []
    for i in range(len(evals)):
        plt.plot(range(num_generations),evals[i][0], label = evals[i][1])
        only_opt_evals = min(evals[i][0])
    plt.legend()
    plt.show()
    only_opt_evals = np.array(only_opt_evals)
    print(f"the optimal evaluation : {np.min(only_opt_evals)} with {evals[np.argmin(only_opt_evals)][1]}")

    # temp execution 
    # qualité de solution