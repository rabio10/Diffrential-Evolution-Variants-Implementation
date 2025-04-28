from DE import *
from JADE import *

num_evals_jade = 0

def f(I):
    global num_evals_jade
    num_evals_jade += 1
    return sum(i**2 for i in I)

i_execution = 0 

def run():
    num_generations = 100
    evals = []
    final_result = [] # [ DE/rand/1 : [best_eval , num_func_evals] , DE/best/1 : [best_eval , num_func_evals] , ... ]
    # DE/rand/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="rand", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/rand/1"))
    
    # DE/best/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="best", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/best/1"))

    # DE/rand/2
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="rand", number_differentials=2)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/rand/2"))
    
    # DE/best/2
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="best", number_differentials=2)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/best/2"))
    
    # DE/current-to-best/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="current-to-best", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/current-to-best/1"))
    
    # DE/current-to-rand/1
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="DE",target_vector_selection_strategy="current-to-rand", number_differentials=1)
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"DE/current-to-rand/1"))
    
    # CODE
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="CODE")
    list_evals_gen = differential_evolution.do_evolution(num_generations,verbose=False)
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = differential_evolution.num_execution_obj_func
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"CODE"))
    
    # JDE
    differential_evolution = DE(50, 2, 0.5, 0.7,variant="JDE")
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
    
    modelJade = JADE(50, 2, 0.5, 0.7,[[-5,5],[-5,5], [4,8], [3, 8]], 0 , f, 0.9)
    list_evals_gen = modelJade.main(num_generations,populationTest=None)[1]
    list_evals_gen = np.array(list_evals_gen)
    best_eval_of_run = list_evals_gen.max()
    num_func_evals = num_evals_jade
    metric_of_variant_one_run = np.array([best_eval_of_run, num_func_evals])     # build metrics of the variant of one run  :  
    final_result.append(metric_of_variant_one_run)
    evals.append((list_evals_gen,"JADE"))


    only_opt_evals = []
    for i in range(len(evals)):
        plt.plot(range(num_generations),evals[i][0], label = evals[i][1])
        only_opt_evals = min(evals[i][0])
    plt.legend()
    plt.savefig("plots/plot_exec_"+ str(i_execution))
    #plt.show()
    only_opt_evals = np.array(only_opt_evals)
    print(f"the optimal evaluation : {np.min(only_opt_evals)} with {evals[np.argmin(only_opt_evals)][1]}")
    return final_result
    
if __name__ == "__main__":
    final_res = run()
    for i in range(3):
        res = run()
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
