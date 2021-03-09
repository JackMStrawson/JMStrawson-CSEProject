#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.linalg
from scipy import optimize
import yaml
import timeit

from assignment13 import OneDimReservoir


# In[129]:


class SolvePermeabilityNewtonKrylov(OneDimReservoir):
    
    def __init__(self, inputs, history):
        
        #stores input dictionary as class attribute, either read from a yaml
        #file or directly from a Python dictonary
        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.load(f, yaml.FullLoader)
        else:
            self.inputs = inputs
        
        #Define attribute
        self.history = history
        self.k_array_length = len(self.history)
        
        #Initiate OneDimReservoir and obtain initial guess
        self.reservoir = OneDimReservoir(self.inputs)
        self.initial_guess = self.reservoir.k

        return
    
    def run_res_sim(self, k):
        #Runs through OneDimReservoir to obtain final pressure values
        
        #Change permeability values

        self.reservoir.k = k 
        
        #Reapply initial conditions
        
        self.reservoir.apply_initial_conditions()
        
        #Solve for pressures with new permeabilities
        
        self.reservoir.fill_matrices()
        
        self.reservoir.solve()
        
        return self.reservoir.get_solution()
    
    def optimize_k(self):
        #Use scipy optimizer to solve for each grid blocks permeability
        
        #Initiate functions and attributes
        run_res_sim = self.run_res_sim
        history = self.history
        k = self.initial_guess
        
        #Fine real permeability
        optimized_k = optimize.newton_krylov(
            lambda k:(run_res_sim(k) - history)**2, k)
                
        return optimized_k


# In[132]:


# Solution = OneDimReservoir('Hetero(100-200)mD.yml')
# k = Solution.k
# Solution.solve()
# SolutionPressure = Solution.get_solution()


# In[133]:


# match = SolvePermeabilityNewtonKrylov('80mD.yml', SolutionPressure)
# %prun match.optimize_k()


# In[134]:


# match = Solve_permeability_newton_krylov('80mD.yml', SolutionPressure)
# %timeit match.optimize_k()
# newk = match.optimize_k()
# print(np.array(k))
# print(newk.tolist())
# for i in range(len(newk)):
#     print(abs(newk[i]-k[i]))


# In[ ]:




