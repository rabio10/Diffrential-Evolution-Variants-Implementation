from DE import *
from JADE import *
from problem_data import *

num_evals_jade = 0
num_generations = 100
prob_dimension = 30
nbr_installation = 15
pop_size = 50
scaling_F = 0.5
Pcr = 0.7


def f(I):
    global num_evals_jade
    num_evals_jade += 1

    nbreInstall =3
    nbreClients =4 # prob dimension = 7
    Demande = [30, 89, 78, 99]
    Capacity = [400, 600, 300]
    CoutAffect =[[50, 40, 30],   
                [70, 60, 20],
                [10, 80, 90],
                [10, 80, 90]]
    CoutOuvert= [3000, 8000, 5000]
    B= 40000
    problem1= Problem(nbreInstall, nbreClients,Demande,  Capacity, CoutAffect, CoutOuvert, B)
    return problem1.penalty(I)


i_execution = 0 

def run(prob_data):

    evals = []
    final_result = [] # [ DE/rand/1 : [best_eval , num_func_evals] , DE/best/1 : [best_eval , num_func_evals] , ... ]
    # DE/rand/1
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data ,variant="DE",target_vector_selection_strategy="rand", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/rand/1"))
    
    # DE/best/1
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data ,variant="DE",target_vector_selection_strategy="best", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/best/1"))

    # DE/rand/2
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation, prob_data,variant="DE",target_vector_selection_strategy="rand", number_differentials=2)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/rand/2"))
    
    # DE/best/2
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data,variant="DE",target_vector_selection_strategy="best", number_differentials=2)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/best/2"))
    
    # DE/current-to-best/1
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data,variant="DE",target_vector_selection_strategy="current-to-best", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/current-to-best/1"))
    
    # DE/current-to-rand/1
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data,variant="DE",target_vector_selection_strategy="current-to-rand", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/current-to-rand/1"))
    
    # CODE
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data,variant="CODE")
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"CODE"))
    
    # JDE
    differential_evolution = DE(pop_size, prob_dimension, scaling_F, Pcr,nbr_installation,prob_data,variant="JDE")
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"JDE"))
    
    # JADE
    #differential_evolution = DE(50, 2, 0.5, 0.7,variant="JADE")
    #evals.append((differential_evolution.do_evolution(num_generations,verbose=False),"JADE"))
    #searchSpace=[[0,3],[0,3], [0,3],    [1,3], [1,3], [1,3],[1,3]]
    problem1= Problem(prob_data.searchSpace, nbr_installation, prob_dimension-nbr_installation,prob_data.Demande,  prob_data.Capacity, prob_data.CoutAffect, prob_data.CoutOuvert, prob_data.B)
    modelJade = JADE(problem1, pop_size, prob_dimension, scaling_F, Pcr,prob_data.searchSpace,0 , problem1.penalty, 0.9, nbr_installation)
    list_evals_gen = modelJade.main(num_generations,populationTest=prob_data.pop)[1]
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = num_evals_jade
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"JADE"))


    only_opt_evals = []
    for i in range(len(evals)):
        plt.plot(range(num_generations),evals[i][0], label = evals[i][1])
        only_opt_evals.append(min(evals[i][0]))
    plt.legend()
    plt.savefig("plots/plot_exec_"+ str(i_execution))
    #plt.show()
    plt.close()
    only_opt_evals = np.array(only_opt_evals)
    print(f"the optimal evaluation : {np.min(only_opt_evals)} with {evals[np.argmin(only_opt_evals)][1]}")
    return final_result
    
if __name__ == "__main__":
    # problem data
    prob_data = problem_data(nbr_installation, prob_dimension - nbr_installation)

    final_res = run(prob_data)
    for i in range(2):
        res = run(prob_data)
        for v,w,i in zip(final_res, res,range(len(final_res))):
            #print(f"===> v : ", v)
            #print(f"===> w : ", w)
            container = np.vstack([v,w])
            final_res[i] = container
        i_execution += 1
    i_execution = 0
    #final_res = np.array(final_res)
    print(final_res)

    # calcule de avrg , best , std 
    stats = []
    for m in final_res:
        best = np.min(m,axis=0)[0]
        avg = np.average(m,axis=0)[0]
        std = np.std(m,axis=0)[0]
        nbr_ev = np.average(m,axis=0)[1]
        stats.append([best, avg, std, nbr_ev])
    print("")
    np.set_printoptions(suppress=True, precision=5)
    print(np.array(stats))
