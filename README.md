﻿# Diffrential-Evolution-Variants-Implementation
This project focuses on the study and implementation of a differential evolution (DE) algorithm and its variants to efficiently solve the static multi-capacity localization problem with a budget constraint. 
Differential evolution, introduced by Stern and Price (1997), is a powerful differential search technique inspired by the principles of natural evolution, which has excelled in solving continuous optimization problems and more recently discrete combinatorial problems.

The variants that are implemented in this project are : 
* The standard différentielle évolution (DE)
  * DE/rand/1
  * DE/best/1
  * DE/rand/2
  * DE/best/2 
  * DE/current-to-best/1 
  * DE/current-to-rand/1 
* The composite differential evolution (CODE)
* The self-adaptive control parameters differential evolution (JDE) 
* The adaptive differential evolution with optional external archive(JADE)

This project is focused on solving the static multi-capacity localization problem with a budget constraint :

<img width="710" height="763" alt="image" src="https://github.com/user-attachments/assets/44e9e2d9-9f5b-4368-ae37-5c661dae3b20" />

with variables: 

<img width="728" height="237" alt="image" src="https://github.com/user-attachments/assets/3c2d736b-524e-47ff-885b-0a0ed43e67b8" />

## Bibliography

- [**Storn, R. & Price, K. (1997).** _Differential Evolution – A simple and efficient adaptive scheme for global optimization over continuous spaces._ Journal of Global Optimization, 11(4), 341–359.](https://www.metabolic-economics.de/pages/seminar_theoretische_biologie_2007/literatur/schaber/Storn1997JGlobOpt11.pdf)

- [**Georgioudakis, M. & Plevris, V. (2020).** _A Comparative Study of Differential Evolution Variants in Constrained Structural Optimization._ Frontiers in Built Environment, 6:102.](
https://scispace.com/pdf/a-comparative-study-of-differential-evolution-variants-in-1rjx48thti.pdf)

- [**Zhang, J. (2007).** _JADE: Self-adaptive differential evolution with fast and reliable convergence performance._ IEEE Transactions on Evolutionary Computation.](https://matlabtools.com/wp-content/uploads/2017/11/p121-3.pdf)
