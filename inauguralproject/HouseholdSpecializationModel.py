
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt


class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()         # creates sort of a dictionary for all parameters
        sol = self.sol = SimpleNamespace()         # creates sort of a dictionary for all solutions 

        # b. defining preferences as specified in the model
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. defining household production as specified in the model
        par.alpha = 0.5
        par.sigma = 1.0

        # d. defining wages as specified in the model
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)    #this is for the plot in question 2 

        # e. targets (for the regression in Question 3 onwards)
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. definig empty matrixes for the solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        sol.alphasol = np.nan
        sol.sigmasol = np.nan

        sol.sigma_when_alpha_fixed = np.nan


    H = 0       #defining H so that the for-loop below does run 
    def calc_utility(self,LM,HM,LF,HF):                 
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 0:
            H = np.fmin(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))
                                                        
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility


    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # creates all combinations of hours worked in market and at home by M and F
    
        LM = LM.ravel() # turns the utput into a vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set the value of utility to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)    # This returns the index of the maximizing eleement of u 
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt 


    def solve(self,do_print=False):
        """ solve model continously """

        #create a SimpleNamespace to store optimal values 
        opt = SimpleNamespace()

        #define an objective function we are going to optimize. Basically the negative utility function we can minimize
        def value_of_choice(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)
        
        #set constraints, bounds and initial value
        constraints = ({'type': 'ineq', 'fun': lambda HF, LF: 24-HF-LF}, {'type': 'ineq', 'fun': lambda HM, LM: 24-HM-LM})
        bounds = ((0,24), (0,24), (0,24), (0,24))
        initial_guess = [6, 6, 6, 6]

        #call on the minimize-solver from the scipy/optimize package
        solution_continuous = optimize.minimize(value_of_choice, initial_guess, method='Nelder-Mead', bounds=bounds, constraints=constraints)

        #assigning the optimal values to the SimpleNamespace
        opt.LM, opt.HM, opt.LF, opt.HF = solution_continuous.x

        return opt


    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        #loop over the vector of female wage and change the value of wF to whereever we are in the vector 
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF

            #solve the model with the discrete solver if keyword argument above is true
            if discrete == True:
                optimal = self.solve_discrete()
            
            #use the contiuous solver if keyword argument above is false 
            else:
                optimal = self.solve()

            #store the resulting values 
            sol.LM_vec[i] = optimal.LM
            sol.HM_vec[i] = optimal.HM
            sol.LF_vec[i] = optimal.LF
            sol.HF_vec[i] = optimal.HF

        return sol.LM_vec, sol.HM_vec, sol.LF_vec, sol.HF_vec


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return sol.beta0, sol.beta1
    

    def printing_estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma and print out their values"""

        par = self.par
        sol = self.sol

        #Defining our objective function which we are minimizing 
        def objective_function(x):
            par.alpha = x[0]
            par.sigma = x[1]
            self.solve_wF_vec()
            self.run_regression()
            return(par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2

        #creating bounds and initial guess
        bounds = [(0, 1), (0, 5)]
        initial_guess = [0.8, 1]

        #call the solver to find the optimal alpha and sigma here 
        solution = optimize.minimize(objective_function, initial_guess, bounds=bounds, method='SLSQP')

        #creating variables to store the optimal alpha and sigma the solver found 
        sol.alphasol = solution.x[0]
        sol.sigmasol = solution.x[1]

        #printing the results in green color 
        print(f'\033[32mRunning the function was a success: {solution.success}\n\n' 
              + f'Results:\n\n'
              + f'The alpha fitted to the data: {sol.alphasol:4f}\n\n'
              + f'The sigma fitted to the data: {sol.sigmasol:4f}')
        

    def printing_estimate_alpha(self, sigma=None):
        '''estimate value of sigma while alpha is fixed to 0.5'''

        par = self.par
        sol = self.sol

        #setting alpha to 0.5 as specified 
        par.alpha = 0.5

        #Defining our objective function which we are minimizing 
        def objective_function(x):
            par.sigma = x
            self.solve_wF_vec()
            self.run_regression()
            return(par.beta0_target-sol.beta0)**2 + (par.beta1_target-sol.beta1)**2
        
        #creating bounds and initial guess
        bounds = [(0, 5)]
        initial_guess = [1]

        #call the solver to find the optimal sigma here 
        solution = optimize.minimize(objective_function, initial_guess, bounds=bounds, method='SLSQP')

        #creating variable to store the optimal sigma the solver found
        sol.sigma_when_alpha_fixed = solution.x[0]

        #printing the result in green color 
        print(f'\033[32mRunning the function was a success: {solution.success}\n\n' 
              + f'Result:\n\n'
              + f'The alpha was set to: {par.alpha:4f}\n\n'
              + f'The sigma fitted to the data is: {sol.sigma_when_alpha_fixed:4f}')

