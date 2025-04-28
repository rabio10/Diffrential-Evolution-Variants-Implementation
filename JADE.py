import numpy as np
import random 
import matplotlib.pyplot as plt
from DE import *


class DiffEvolution:
    def __init__(self, populationSize, problemDim,
                  scalingFactor, crossOverProb,  searchSpace, stopCriteria, fitnessFunt):
        self.popSize = populationSize
        self.probDim = problemDim
        self.scalingFact = scalingFactor
        self.crossOverProb = crossOverProb
        #self.costFunct = costFunct
        self.searchSpace = searchSpace
        self.fitFunct = fitnessFunt
        #searchSpace= [[3, 6],[4, 7],[0,1],[9,8]] example for problemDim=4


    def Initialization(self,):
        pop=[]
        np.random.seed(5)
        for i in range(self.popSize):
            indiv=[]
            index = 0
            for j in range(self.probDim):
                new= round(random.randint(self.searchSpace[index][0], self.searchSpace[index][1] ),2)
                #print(new)
                indiv.append(new)
                index +=1
            pop.append(indiv)
        np.random.seed()
        return pop


    def Mutation_indiv(self, parentVect, population, strat= 'best'):
        population2 = population.copy()
        population2.remove(parentVect)

        if strat=='rand':
            targetVector = random.choices(population2)[0]
        elif strat == 'best':
            fitFunctList = [self.fitFunct(indiv) for indiv in population2]
            mini = min(fitFunctList)
            targetVector = population2[fitFunctList.index(mini)]

        population2.remove(targetVector)

        randSol1 = random.choices(population2)[0]
        population2.remove(randSol1)
        randSol2 = random.choices(population2)[0]

        #Calculate Trial Vector:
        mutantVector = [round(targetVector[i]+ self.scalingFact*(randSol1[i]-randSol2[i]),2) for i in range(len(targetVector))]

        #searchSpace Verification and Correction
        for i in range(len(mutantVector)-1):
            if mutantVector[i] < self.searchSpace[i][0] :
                mutantVector[i] = self.searchSpace[i][0]
            elif mutantVector[i] > self.searchSpace[i][1] :
                mutantVector[i] = self.searchSpace[i][1]


        return mutantVector


    def CrossOver(self, parentVect, mutantVect, type= 'bin'):
        trialVect = [0 for k in range(len(parentVect))]
        if type == 'bin':
            randIndex = random.choices(range(len(parentVect)-1))
            #print(trialVect)
            for i in range(self.probDim):
                #print(i, trialVector[i], mutantVect[i])
                randIndex2 = random.random()
                if randIndex2 >= self.crossOverProb or i==randIndex:
                    trialVect[i] = mutantVect[i]
                else:
                    trialVect[i] = parentVect[i]

        elif type == 'exponential':
            n= random.choices(range(self.probDim))[0]
            for i in range(n, self.probDim):
                print(i)
                rand_i = random.random()
                if rand_i < self.crossOverProb:
                    trialVect[i] = mutantVect[i]
                else:
                    trialVect[i] = parentVect[i]

        return trialVect

    #On traite un probleme de minimisation
    #fitFunct(Vactor)--> number 
    def Selection(self, parentVect, trialVect):
        selectedVect = parentVect
        if self.fitFunct(parentVect)> self.fitFunct(trialVect):
            selectedVect = trialVect
            #print(self.fitFunct(parentVect), self.fitFunct(trialVect) )
        return selectedVect
   


    def main(self, nbrGenerations, populationTest):
        L={}
        E =[]

        if populationTest:
            population = populationTest
        else:
            population = self.Initialization()

        for n in range(nbrGenerations):
            L[n]=population

            #Calculate minFitFunct for eachGeneration
            evalGen= [self.fitFunct(i) for i in population]
            evalMin= min(evalGen)
            E.append(evalMin)

            populationCopy = []
            for indiv in population:
                mutant= self.Mutation_indiv( indiv, population)
                trial = self.CrossOver(indiv, mutant)
                selected = self.Selection(indiv, trial)

                populationCopy.append(selected)
            population = populationCopy

        #print('list des min de chaque gen', E)
        return L, E



   
    def Graphe(self, num_generation, populationTest):
            evals_of_generations=self.main(num_generation, populationTest)[1]
            plt.plot(range(num_generation),evals_of_generations)
            plt.show()
    
#JADE
#to add the truncated values to the CRi


class JADE(DiffEvolution):
    def __init__(self,Problem, populationSize, problemDim, scalingFactor, crossOverProb, searchSpace, stopCriteria, fitnessFunt, p, NbrInstallations):
        super().__init__(populationSize, problemDim, scalingFactor, crossOverProb, searchSpace, stopCriteria, fitnessFunt)
        self.SCR = None
        self.SF = None

        self.p=p


        self.uF= None
        self.uCR = None

        self.CR_list=[]
        self.F_list=[]


        #adapting to the problem:
        self.nbrInstall= NbrInstallations
    #Initialisation same as parent Class.. no need to overwrite it
    
    def ParameterGeneration(self,):

        self.CR_list = [random.uniform(self.uCR, 0.1) for i in range(self.popSize)]

        self.F_list = [random.uniform(0, 1.2) for i in range(self.popSize//3)]+[random.uniform(self.uF, 0.1) for i in range(self.popSize//3, self.popSize)]

    def Mutation_indiv(self, IndexParentVect, population, topPercent):
        parentVect = population[IndexParentVect]

        X_best = random.choice(topPercent)

        populationCopy= population.copy()
        populationCopy.remove(parentVect)

        randomIndiv1= random.choice(populationCopy)
        populationCopy.remove(randomIndiv1)
        randomIndiv2= random.choice(populationCopy)

        mutantVect= [0 for i in range(self.probDim)]
        for para in range(self.probDim):
            mutantVect[para] = int(parentVect[para] + self.F_list[IndexParentVect]*(X_best[para] -
                    parentVect[para])+ self.F_list[IndexParentVect]*(randomIndiv1[para] - randomIndiv2[para]))
        return mutantVect


#exemple d'indiv: [3, 1, 0,    1, 2, 1, 3]
    #nbrInstall = 3
    def CrossOver(self, population, IndexParentVect, mutantVect):
        parentVectP1= population[IndexParentVect][0: self.nbrInstall] #[3, 1, 0]
        parentVectP2= population[IndexParentVect][self.nbrInstall:] #[1, 2, 1, 3]

        trialVect=[0 for i in range(len(mutantVect))]

        #for part1 (les installations dans l'indiv)
        j_rand1= random.randint(0, self.nbrInstall-1) #0,1,2
        for i in range(self.nbrInstall): #0,1,2
            prob= random.random()
            if i== j_rand1 or prob< self.CR_list[IndexParentVect]:
                trialVect[i] = mutantVect[i]
            else: 
                trialVect[i] = parentVectP1[i]
        #print(trialVect)
        #return trialVect
    
        #for Prt2 (les clients dans l'indiv)
        j_rand2= random.randint(self.nbrInstall, self.probDim-1)
        for i in range(self.nbrInstall, self.probDim):
            prob= random.random()
            if i== j_rand2 or prob< self.CR_list[IndexParentVect]:
                trialVect[i] = mutantVect[i]
            else: 
                trialVect[i] = parentVectP2[i-self.nbrInstall]
        #print(trialVect)    

        #print(trialVect)
        return trialVect
    

    #Selection same as DiffeEvolution class with the update parameters


    def Selection(self,population, IndexParentVect, trialVect):
        parentVect= population[IndexParentVect]
        selectedVect  = super().Selection(parentVect, trialVect)
        if selectedVect==trialVect: #if true: successful Update 
            self.SCR.append(self.CR_list[IndexParentVect])
            self.SF.append(self.F_list[IndexParentVect])
        return selectedVect  
    



    def main(self, nbrGenerations, populationTest, c=0.1):
        L={}
        E =[]

        if populationTest:
            population = populationTest
        else:
            population = self.Initialization()
            #print('initialisation',population)
            
        self.SCR = []
        self.SF = []
        self.uF= 0.5
        self.uCR = 0.5




        def ArithmeticMean(I):
            return sum(i for i in I)/ len(I)

        def LehmerMean(I):
            return  sum(i**2 for i in I)/sum(i for i in I)

        for n in range(nbrGenerations):
            L[n]=population


            #Calculate minFitFunct for eachGeneration
            evalGen= [self.fitFunct(i) for i in population]
            evalMin= min(evalGen)
            E.append(evalMin)

            PopulationCopy= []
            #initiation of self.SCR and self.SF
            self.ParameterGeneration()
           #print(self.CR_list)
           #print(self.uCR)

#here: lets define the top p of the population:
            

            eval_Dict= {i: self.fitFunct(population[i]) for i in range(len(population))}

            #topPercentList222 = sorted(eval_Dict.items(), key=lambda item: item[1])
            #print('len liste ordonnes',len(topPercentList222))

            topPercentList = sorted(eval_Dict.items(), key=lambda item: item[1])[: int(self.p*len(population))]
            #sorted_dict=[(index: fit(popoulaion[index])] top p 


            topPercent = [[item[0] for item in topPercentList ]]

            for IndexIndiv in range(self.popSize):
                #print('indiv debut',population[IndexIndiv])

                mutantVect= self.Mutation_indiv(IndexIndiv, population, topPercent)
                #print('mutant vect',mutantVect)

                trialVect = self.CrossOver(population, IndexIndiv, mutantVect)
                #print('trial vect',trialVect)

                #here the self.SCR and self.SF are updated
                selectedVect = self.Selection(population, IndexIndiv, trialVect)
                #print('selected indiv',selectedVect)
                
                PopulationCopy.append(selectedVect)

           #print(len(PopulationCopy))
            population = PopulationCopy

            #updation uCR and uF

            self.uCR= (1-c)* self.uCR + c* ArithmeticMean(self.SCR)
            self.uF = (1-c)* self.uF + c* LehmerMean(self.SF)

        return L,E




