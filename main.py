from DE import *

if __name__ == "__main__":
    num_generations = 100
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="JDE",target_vector_selection_strategy="rand", number_differentials=2)
    evals_of_generations = differential_evolution.do_evolution(num_generations,verbose=False)

    print(f"the optimal evaluation : {evals_of_generations[-1]}")

    plt.plot(range(num_generations),evals_of_generations)
    plt.show()
