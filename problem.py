import numpy as np

class Problem:
    def __init__(self, nbreInstall, nbreClients,Demande,  Capacity, CoutAffect, CoutOuvert, B):
        #Cout = array dim : nbreClients*  nbreInstall
        self.nbrInstall = nbreInstall #3--> les indices des install: 1, 2, 3
        self.nbrClients = nbreClients #4--> les indices des clients:  1, 2, 3, 4
        self.Capacity = Capacity #[400, 500, 300]--> niveau 1 a une capacite 400, niveau 1 a une capacite 500... etc
        self.CoutAffect = CoutAffect # [[50, 40, 30],   CoutAffec du client 1 al'inst 2 est 40
                                      # [70, 60, 20],
                                      # [10, 80, 90],
                                      #[10, 80, 90]]
        self.CoutOuvert = CoutOuvert #[3000, 8000, 5000]--> 3000 est le cout pour level 1, 8000 pour level 2
        self.B= B
        self.Demande= Demande #[30, 89, 78, 99]
        
        #exemple d'indiv: [3, 1, 0,    1, 2, 1, 3]
    

    def penalty (self, indiv, coeff1=5, coeff2= 5, coeff3= 5):
        Install = indiv[0: self.nbrInstall+1]
        Clients = indiv[self.nbrInstall:]

        #obj:
        obj_indiv= 0
        for i in range(0, self.nbrClients):  # 0,1,2,3
            install_j= indiv[i+ self.nbrInstall] 
            #print(install_j)  #1, 2, 1, 3

            Cout_j = self.CoutAffect[i][install_j-1]

            d_i= self.Demande[i]

            obj_indiv = obj_indiv + d_i* Cout_j
        
        #exemple d'indiv: [3, 1, 0,    1, 2, 1, 3]


        #violation1:
        Violation1=0
        for j in range(1, self.nbrInstall+1):  #1,2,3

            #sommes des demandes affectes a une installaj
            SumDemande_j= 0
            for i in range(0, self.nbrClients): #0,1,2,3 --> 3,4,5,6
                #print(indiv[i+ self.nbrInstall])
                if indiv[i+ self.nbrInstall]==j:
                    SumDemande_j += self.Demande[i]
        
            #somme des capacite pour l'install j
            level_j= indiv[j-1] 
            SumCapacite_j = self.Capacity[level_j-1]

            maxi= max(0, SumDemande_j- SumCapacite_j)
            Violation1 += maxi 


        #Violation2:
        Violation2=0
        #compter le nombre dinstallations qui sont affectes a des cliens et leurs niveau est 0.
        for i in range(0, self.nbrClients):  #0,1,2,3
            install_j = indiv[i+self.nbrInstall]  #indiv: 3,4,5,6
            level_j = indiv[install_j-1]
            if level_j == 0:
                Violation2 +=1 

        #violation3:
        
        sum3=0
        for j in range(1,self.nbrInstall+1): #1,2,3
            level_j = indiv[j-1]
            sum3 = sum3 + self.CoutOuvert[level_j -1]
        violation3= max(0, sum3 - self.B)
        
        
        pen= obj_indiv+ coeff1* Violation1 + coeff2* Violation2+ coeff3* violation3
        return pen  


        

