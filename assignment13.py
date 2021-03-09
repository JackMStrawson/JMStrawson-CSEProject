#!/usr/bin/env python
# coding: utf-8
# %%

# # Assignment 12
# 
# Consider the reservoir shown below with the given properties that has been discretized into 4 equal grid blocks.
# 
# ![image](images/grid.png)
# 
# Below is a skeleton of a Python class that can be used to solve for the pressures in the reservoir.  The class is actually written generally enough that it can account for an arbitrary number of grid blocks, but we will only test cases with 4.  The class takes a Python dictonary of input parameters as an initialization argument.  An example of a complete set of input parameters is shown in the `setup()` function of the tests below.
# 
# Several simple useful functions are already implemented, your task is to implement the functions `compute_transmisibility()`, `compute_accumulation()`, `fill_matrices()` and `solve_one_step()`.  `fill_matrices()` should correctly populate the $\mathbf{T}$, $\mathbf{B}$ matrices as well as the vector $\vec{Q}$.   These should also correctly account for the application of boundary conditions.  Only the boundary conditions shown in the figure will be tested, but in preparation for future assignments, you may wish to add the logic to the code such that arbitrary pressure/no flow boundary conditions can be applied to either side of the one-dimensional reservoir. You may need to use the `'conversion factor'` for the transmissibilities. `solve_one_step()` should solve a single time step for either the explicit or implicit methods depending on which is specified in the input parameters. The $\vec{p}{}^{n+1}$ values should be stored in the class attribute `self.p`.  If this is implemented correctly, you will be able to then use the `solve()` function to solve the problem up to the `'number of time steps'` value in the input parameters.
# 
# This time, in preparation for solving much larger systems of equations in the future, use the `scipy.sparse` module to create sparse matrix data structures for $\mathbf{T}$ and $\mathbf{B}$.  The sparsity of the matrix $\mathbf{T}$ is tested, so please assign this matrix to a class attribute named exactly `T`.  Use `scipy.sparse.linalg.spsolve()` for the linear solution of the `'implicit'` method implementation.
# 
# Once you have the tests passing, you might like to experiment with viewing several plots with different time steps, explicit vs. implicit, number of grid blocks, etc.  To assist in giving you a feel for how they change the character of the approximate solution.  I have implemented a simple plot function that might help for this.

# %%


import numpy as np
import yaml
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


# %%


class OneDimReservoir():
    
    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            finite differences.
        '''
        
        #stores input dictionary as class attribute, either read from a yaml file
        #or directly from a Python dictonary
        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.load(f, yaml.FullLoader)
        else:
            self.inputs = inputs
        
        #computes delta_x
        self.Nx = self.inputs['numerical']['number of grids']['x']
        
        if 'delta x' in self.inputs['numerical']:
            self.delta_x = self.inputs['numerical']['delta x']
        else:
            self.delta_x = np.ones(self.Nx) * (self.inputs['reservoir']['length'] / float(self.Nx))
        
        # reservoir properties 
        if 'cross sectional area' in self.inputs['reservoir']:
            if isinstance(self.inputs['reservoir']['cross sectional area'], list) == True:
                self.Ai = self.inputs['reservoir']['cross sectional area']
            else:
                self.Ai = np.ones(self.Nx) * self.inputs['reservoir']['cross sectional area']
        else:
            pass
        
        if isinstance(self.inputs['reservoir']['permeability'], list) == True:
            self.k = self.inputs['reservoir']['permeability']
        else:
            self.k = np.ones(self.Nx) * self.inputs['reservoir']['permeability']
        
        self.phi =  self.inputs['reservoir']['porosity']
        
        # fluid properties 
        self.mu = self.inputs['fluid']['water']['viscosity']
        
        self.Balpha = self.inputs['fluid']['water']['formation volume factor']
        
        self.ct = self.inputs['fluid']['water']['compressibility']
        
        #gets delta_t from inputs
        self.delta_t = self.inputs['numerical']['time step']
        
        #computes transmissibility 
        self.compute_transmissibility(0,0) 
        
        #computes accumulation 
        self.compute_accumulation(0)
        
        #applies the initial reservoir pressues to self.p
        self.apply_initial_conditions()
        
        #calls fill matrix method (must be completely implemented to work)
        self.fill_matrices()
        
        #create an empty list for storing data if plots are requested
        if 'plots' in self.inputs:
            self.p_plot = []
            
        return
        
    def compute_transmissibility(self, i, j):
        '''
            Computes the transmissibility.
        '''
        #Complete implementation here
        
        Ti = ((2 * self.k[i] * self.Ai[i] * self.k[j] * self.Ai[j]) / (self.k[i] * self.Ai[i] * self.delta_x[j] + self.k[j] * self.Ai[j] *         self.delta_x[i])) * 1/(self.mu * self.Balpha)
        
        return (Ti)
    
    
    def compute_accumulation(self, i):
        '''
            Computes the accumulation.
        '''
        #Complete implementation here
        
        Bscalar = self.Ai[i] * self.delta_x[i] * self.phi * self.ct / self.Balpha
        
        return (Bscalar)
    
    def fill_matrices(self):
        '''
            Fills the matrices A, I, and \vec{p}_B and applies boundary
            conditions.
        '''
        #Complete implementation here
        
        #Right boundary conditions
        if self.inputs['boundary conditions']['right']['type'] == 'prescribed pressure':
            TN = self.compute_transmissibility(self.Nx - 1, self.Nx - 1)
            QBr = 2 * TN * self.inputs['boundary conditions']['right']['value'] 
        else:
            TN = 0
            QBr = self.inputs['boundary conditions']['right']['value'] * self.mu * self.delta_x[0] / self.k[0]
        
        #Left boundary conditions
        if self.inputs['boundary conditions']['left']['type'] == 'prescribed pressure':
            T0 = self.compute_transmissibility(0,0)
            QBl = 2 * T0 * self.inputs['boundary conditions']['left']['value']
        else:
            T0 = 0
            QBl = self.inputs['boundary conditions']['left']['value'] * self.mu * self.delta_x[self.Nx - 1] / self.k[self.Nx - 1]
        
        #Compute accumulation matrix
        Bmatrix = np.zeros([self.Nx, self.Nx])
        for i in range(0,self.Nx):
            Bmatrix[i][i] = self.compute_accumulation(i)
        self.B = scipy.sparse.csr_matrix(Bmatrix)
        
        #Compute inverse of accumulation matrix
        Bmatrixinv = np.zeros([self.Nx, self.Nx])
        for i in range(0, self.Nx):
            Bmatrixinv[i][i] = 1/self.compute_accumulation(i) 
        self.Binv = scipy.sparse.csr_matrix(Bmatrixinv)
        
        #Compute transmissibility matrix
        Tmatrix = np.zeros([self.Nx,self.Nx])
        for i in range(0,self.Nx):
            for j in range(0,self.Nx):
                if i == j:
                    if i == 0:
                        Tmatrix[0][0] = 2 * T0 + self.compute_transmissibility(i, i + 1)
                    elif i == self.Nx - 1:
                        Tmatrix[self.Nx - 1][self.Nx - 1] = self.compute_transmissibility(i - 1, i) + 2 * TN
                    else:
                        Tmatrix[i][j] = self.compute_transmissibility(i - 1, i) + self.compute_transmissibility(i, i + 1)
                elif j == i+1:
                    Tmatrix[i][j] = -self.compute_transmissibility(i, i + 1)
                elif j == i-1:
                    Tmatrix[i][j] = -self.compute_transmissibility(i - 1, i)
        Tmatrix = Tmatrix * self.inputs['conversion factor']
        self.T = scipy.sparse.csr_matrix(Tmatrix)
        
        #Compute Flowrates
        Q = np.zeros(self.Nx)
        Q[0] = QBl 
        Q[self.Nx-1] = QBr
        self.Q =  Q * self.inputs['conversion factor']
        
        return
            
                
    def apply_initial_conditions(self):
        '''
            Applies initial pressures to self.p
        '''
        
        N = self.Nx
        
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']

        return
                
                
    def solve_one_step(self):
        '''
            Solve one time step using either the implicit or explicit method
        '''
        #Complete implementation here
        
        if self.inputs['numerical']['solver'] == 'implicit':
            self.p = scipy.sparse.linalg.spsolve((self.T + 1/self.delta_t * self.B), ((1/self.delta_t * self.B).dot(self.p) +  self.Q))      
        
        elif self.inputs['numerical']['solver'] == 'explicit':
            self.product = self.Q - (self.T).dot(self.p)
            self.newBinv = self.delta_t  * self.Binv
            self.p = self.p +  (self.newBinv).dot(self.product)
        
        elif 'mixed method' in self.inputs['numerical']['solver']:
            theta = self.inputs['numerical']['solver']['mixed method']['theta']
            self.p = scipy.sparse.linalg.spsolve(( (1- theta) * self.T + 1 / self.delta_t * self.B), ((1 / self.delta_t * self.B - theta * self.T).dot(self.p) +  self.Q))
       
        return
            
            
    def solve(self):
        '''
            Solves until "number of time steps"
        '''
        
        for i in range(self.inputs['numerical']['number of time steps']):
            self.solve_one_step()
            
            if i % self.inputs['plots']['frequency'] == 0:
                self.p_plot += [self.get_solution()]
                
        return
                
    def plot(self):
        '''
           Crude plotting function.  Plots pressure as a function of grid block #
        '''
        
        if self.p_plot is not None:
            for i in range(len(self.p_plot)):
                plt.plot(self.p_plot[i])
        
        return
            
    def get_solution(self):
        '''
            Returns solution vector
        '''
        return self.p


# # Example code execution
# 
# If you'd like to run your code in the notebook, perhaps creating a crude plot of the output, you can uncomment the following lines of code in the cell below.  You can also inspect the contents of `inputs.yml` and change the parameters to see how the solution is affected.

# %%


# import matplotlib.pyplot as plt
# %matplotlib inline
# implicit = OneDimReservoir('inputs.yml')
# implicit.solve()
# implicit.plot()


# %%




