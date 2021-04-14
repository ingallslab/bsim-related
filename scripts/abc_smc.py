# Approximate Bayesian Computation
import subprocess
import os
import math
import random
import pandas
from pathlib import Path

import numpy as np
from numpy import random as rnd
import scipy.stats as st

#from abcsysbio import EpsilonSchedule
#from abcsysbio import kernels
#from abcsysbio import statistics
#import inferParameters
#import plotter

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bsim_related.abc_smc import inferParameters, plotter, EpsilonSchedule, kernels, statistics

class abcsmc:

    def __init__(self,
                 cmds,
                 prior,
                 first_epsilon,
                 final_epsilon,
                 alpha,
                 n_par,
                 n_pop,
                 n_params,
                 initial_pop,
                 sim_time,
                 bsim_data,
                 cp_data,
                 bsim_jar,
                 jars,
                 driver_file,
                 bsim_export_time,
                 cp_export_time,
                 sim_dim):
        self.prior = prior
        self.first_epsilon = first_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = [self.first_epsilon]
        self.alpha = alpha
        self.alphas = [alpha]
        self.n_par = n_par
        self.n_pop = n_pop
        self.n_params = n_params
        self.initial_pop = initial_pop
        self.sim_time = sim_time
        self.bsim_data = bsim_data
        self.cp_data = cp_data
        self.bsim_jar = bsim_jar
        self.jars = jars
        self.driver_file = driver_file
        self.bsim_export_time = bsim_export_time
        self.cp_export_time = cp_export_time
        self.sim_dim = sim_dim
        
        self.kernel = []
        self.kernel_type = 2                                                        # component-wise normal kernel
        self.kernel_list = [[i for i in range(0, self.n_params)],[],[]]             # kernel_list[0] specifies the index of parameters used for abcsysbio

        self.prev_weights = []
        self.prev_distances = []
        self.prev_population = []

        self.parameters = []
        self.distances = []
        self.acceptance_rates = []
        self.total_runs = 0
        self.cmds = cmds                                                            # list of commands to change parameter values in BSim

    # Gets a value (uniform) between boundaries, rounded to 3 decimal places
    def getRandomValue(self, bounds):
        return round(random.uniform(bounds[0], bounds[1]), 3)

    # Convert the parameters into string to use in the command line
    def paramToString(self, curr_params):
        str_params = tuple(str(i) for i in curr_params)
        return str_params

    # Execute BSim simulation with given parameters
    def executeSimulation(self, curr_params):
        str_params = self.paramToString(curr_params)

        cmd = ["java", "-cp", self.bsim_jar + ";" + self.jars, self.driver_file, "-pop", str(self.initial_pop), "-simt", str(self.sim_time)]
        for i in range(0, len(str_params)):
            cmd.append(self.cmds[i])
            cmd.append(str_params[i])

        process = subprocess.Popen(cmd)
        process.wait()          # Waits until the simulation has been completed

    # Determine the component wise normal 
    def perturbParams(self, current_params):
        pert_params = []

        # index for the list of parameters
        ind = 0
        while ( ind < len(current_params) ):
            
            # Draws random sample from normal distribution rounded to 3 decimal places
            pert = round(rnd.normal(current_params[ind], np.sqrt(self.kernel[2][ind])), 3)

            # Check if the perturbed parameter is within the boundaries of the prior
            if ( pert >= self.prior[ind][0] and pert <= self.prior[ind][1] ):
                pert_params.append(pert)

                ind += 1
        return pert_params

    # Get the probability density of the current value from the pdf of the kernel
    def getPdfKernel(self, curr_params, prev_params, auxilliary):
        prob_density = 1.0
        for n in self.kernel[0]:
            mean = prev_params[n]
            stdv = np.sqrt(self.kernel[2][n])
            kern = statistics.getPdfGauss(mean, stdv, curr_params[n])
            kern = kern/auxilliary[n]
            prob_density *= kern
        return prob_density

    # Get auxillary
    def getAux(self, parameters):
        ret = []
        for n in self.kernel[0]:
            mean = parameters[n]
            stdv = np.sqrt(self.kernel[2][n])
            ret.append(st.norm.cdf(self.prior[n][1], mean, stdv) - st.norm.cdf(self.prior[n][0], mean, stdv))
        #print("ret: ", ret)
        return ret

    # Calculate the weight of a single particle
    def calcWeight(self, curr_params, t):
        if ( t == 0 ):
            return 1.0
        else:
            numer = 1.0
            for n in self.kernel[0]:     # Prior is uniform
                numer *= 1/(self.prior[n][1] - self.prior[n][0])
                 
            denom = 0.0
            aux = self.getAux(curr_params)
            for i in range (0, self.n_par):
                kernel_pdf = self.getPdfKernel(curr_params, self.prev_population[i], aux)
                denom += self.prev_weights[i] * kernel_pdf
            return numer/denom
        
    # Normalize the weights
    def normalizeWeights(self, weights, len_pop):
        total = sum( weights )
        for i in range(0, len_pop):
            weights[i] = weights[i]/float(total)

    # Compute the next epilson using previous distances in an automated epsilon schedule
    def computeEpsilon(self, curr_epsilon, pop):
        # Tolerance
        tol = 1e-4
        # Index of the total distance
        total_dist_ind = 0

        '''# Increase or decrease alpha depending on the acceptance rate
        if ( pop > 1): 
            if ( (100*((self.acceptance_rates[pop] - self.acceptance_rates[pop -1])/self.acceptance_rates[pop - 1]) < -35) and self.alpha < 0.80 ):
                self.alpha += 0.10
            elif ( (100*((self.acceptance_rates[pop] - self.acceptance_rates[pop -1])/self.acceptance_rates[pop - 1]) > 35)  and self.alpha > 0.20):
                self.alpha -= 0.10'''

        prev_dist = np.sort(self.prev_distances, axis = 0) # Sort on the vertical 
        ntar = int(self.alpha * self.n_par)
        
        # Get the new epsilon from the total distance
        new_epsilon = round(prev_dist[ntar, total_dist_ind], 3)

        # Attempt to reduce the epsilon if the new and previous epsilon are equal
        if (new_epsilon >= curr_epsilon):
            new_epsilon *= 0.95

        # Check if finished
        end = True
        if (new_epsilon < self.final_epsilon or abs(new_epsilon - self.final_epsilon) < tol):
            new_epsilon = self.final_epsilon
        else:
            end = False

        # Update epsilon schedule
        self.epsilon_schedule.append(new_epsilon)
        # Add current alpha value to array
        self.alphas.append(self.alpha)

        print("computeEpsilon new_epsilon:", new_epsilon)
        print("self.alpha: ", self.alpha)
        
        return end, new_epsilon

    # Runs using a fixed epsilon schedule
    def run_fixed(self, epsilon_type):

        tol_type = epsilon_type
        n_iter = self.n_pop
        self.epsilon_schedule = EpsilonSchedule.EpsilonSchedule(tol_type, self.final_epsilon, self.first_epsilon, n_iter).tol
        print("epsilon_schedule: ", self.epsilon_schedule)

        for pop in range(0, len(self.epsilon_schedule)):

            if ( pop == len(self.epsilon_schedule) - 1):
                curr_pop, curr_dist = self.iterate_population(self.epsilon_schedule[pop], pop, False, False)
            else:
                curr_pop, curr_dist = self.iterate_population(self.epsilon_schedule[pop], pop, False, False)

            # Add current population of accepted particles to total parameters
            self.parameters.append(curr_pop)
            # Add current population of accepted particle distances to total distances
            self.distances.append(curr_dist)

        return

    # Runs using an automated epsilon schedule
    def run_auto(self):
        running = True
        final = False
        epsilon = [self.first_epsilon]
        pop = 0

        while (running):
            if (final == True): running = False
            
            if ( epsilon == self.final_epsilon):
                curr_pop, curr_dist = self.iterate_population(epsilon, pop, False, False)
            else:
                curr_pop, curr_dist = self.iterate_population(epsilon, pop, False, False)

            # Add current population of accepted particles to total parameters
            self.parameters.append(curr_pop)
            # Add current population of accepted particle distances to total distances
            self.distances.append(curr_dist)

            if (running):
                # Compute the next epsilon
                final, epsilon = self.computeEpsilon(epsilon, pop)
                # Increase population
                pop += 1

    # Computes one population
    def iterate_population(self, epsilon, t, export_data, export_plots):
        # Number of accepted particles
        n = 0
        # Number of runs for population
        pop_runs = 0

        # The list of all accepted parameters and their distances
        parameters = []
        distances = []

        # The current and previous population of accepted particles
        curr_population = []
        # The weights of current and previous accepted particles in the population
        curr_weights = []
        # The current accepted distances in the population
        curr_distances = []
        # The current parameters to be tested
        current_params = []
        
        while ( n < self.n_par ):

            # Increase number of runs in population
            pop_runs += 1
            # Increase total number of runs
            self.total_runs += 1

            # Generate a parameter vector from specified bounds of prior distribution
            if ( t == 0 ):
                current_params = []
                for i in range(0, len(self.prior)):
                    current_params.append(self.getRandomValue(self.prior[i]))
                print("t==0; current params: ", current_params)
            else:
                current_params = random.choices(population = self.prev_population, weights = self.prev_weights, k = 1)[0]   # returns a list
                print("t!=0; before pert current params: ", current_params)
                print("prev_population: ", self.prev_population)
                
                # Perturb the particle
                current_params = self.perturbParams(current_params)
                print("t!=0; perturbed current params: ", current_params)

            # Wasserstein distances of the elongation and distance distributions between simulated and experimental data
            elongation_dist = -1
            division_dist = -1

            # In case of lost data in csv
            while (elongation_dist == -1 and division_dist == -1):
                # Run BSim simulation with given parameters
                self.executeSimulation(current_params)
                # Get the elongation and division distances
                elongation_dist, division_dist, local_anisotropy_dist, aspect_ratio_diff, density_parameter_diff = inferParameters.run(self.bsim_data,
                                                                                                                                       self.cp_data,
                                                                                                                                       self.paramToString(current_params),
                                                                                                                                       export_data,
                                                                                                                                       export_plots,
                                                                                                                                       self.bsim_export_time,
                                                                                                                                       self.cp_export_time,
                                                                                                                                       self.sim_dim)                
            # Get the total distance
            total_dist = elongation_dist + division_dist + local_anisotropy_dist + abs(aspect_ratio_diff) + abs(density_parameter_diff)
            
            print("elongation_dist: ", elongation_dist, "division_dist: ", division_dist)
            print("local_anisotropy_dist: ", local_anisotropy_dist, "aspect_ratio_diff: ", aspect_ratio_diff, "density_parameter_diff: ", density_parameter_diff)
            print("total_dist: ", total_dist)
            print("epsilon: ", epsilon)

            print("total_dist={} epsilon={} n={} n_par={}\ntotal_dist < epsilon: {}".format(total_dist, epsilon, n, self.n_par, (total_dist < epsilon)))
            # Accept the current parameters if the discrepancy between the simulated data and experimental data is less than the threshold
            if ( total_dist < epsilon ):
                
                curr_population.append(current_params)
                curr_weights.append(self.calcWeight(current_params, t))
                curr_distances.append((total_dist, elongation_dist, division_dist, local_anisotropy_dist, aspect_ratio_diff, density_parameter_diff))
                
                #print("curr_weights: ", curr_weights)
                print("current pop: ", curr_population)
                #print("curr_distances: ", curr_distances)

                # Increase the number of particles in the population
                n += 1

                print("n: ", n)
                print("t: ", t)
                print("total runs: ", self.total_runs)
                print("pop_runs: ", pop_runs)
                print("epsilon: ", epsilon)

        # Save the acceptance rate for the population rounded to 3 decimals
        acceptance_rate = round((self.n_par/pop_runs)*100, 3)
        self.acceptance_rates.append(acceptance_rate)
        print("acceptance_rates: ", self.acceptance_rates)
        
        # Update previous populations for the next population
        self.prev_distances = curr_distances
        self.prev_population = curr_population
        self.prev_weights = curr_weights

        # Normalize the weights
        self.normalizeWeights(self.prev_weights, len(self.prev_population))
        #print("normalize prev_weights: ", self.prev_weights)

        # Update kernel using previous population
        self.kernel = kernels.getKernel(self.kernel_type, self.kernel_list, np.array(self.prev_population), self.prev_weights)
        #print("kernel[2] (variance): ", self.kernel[2])

        return curr_population, curr_distances

    # Find the sum of squared distances between the median parameter of every population and the true value
    def get_error(self):
        n_pop = len(self.parameters)
        median_params = []
        for i in range(0, n_pop):
            sorted_params = np.sort(self.parameters[i], axis = 0)
            ind = round(len(sorted_params)/2)
            median_params.append(sorted_params[ind])

        print("median_params: ", median_params)

        true_values = [1.23, 0.277, 7.0, 0.1]
        #true_values = [1.23, 0.277, 7.0, 0.1, 50.0, 50.0, 10.0, 0.6, 100, 0.05]
        #true_values = [50.0, 50.0, 10.0, 0.6, 100, 0.05]
        errors = []
        for i in range(0, len(median_params)):
            error = 0
            for j in range(0, len(true_values)):
                error += ( median_params[i][j] - true_values[j] )**2
            errors.append(error)

        print("errors: ", errors)
            
        return errors, median_params

    # Save stats to csv
    def save_stats(self):
        n_pop = len(self.parameters)
        cols = ["Population", "Epsilon", "Acceptance_Rate", "Error", "Alpha", "Median_Parameters"]
        data = []
        errors, median_params = self.get_error()

        if ( len(self.alphas) < n_pop ):
            self.alphas = ["-" for i in range(0, n_pop)]

        for t in range(0, n_pop):
            row = [t + 1]
            row.append(self.epsilon_schedule[t])
            row.append(self.acceptance_rates[t])
            row.append(errors[t])
            row.append(self.alphas[t])
            row.append(median_params[t])

            # Add row
            data.append(row)

        # Export data to csv file
        param_data = pandas.DataFrame(data, columns = cols)
        print(param_data)

        # If folder doesn't exist, create new folder "Posterior_Data" to store the csv files
        data_folder = "Stats_" + str(self.parameters[0][0][0]) + "_" + str(self.parameters[0][0][1])
        stats_path = Path(__file__).parent.absolute()/'Posterior_Data'/data_folder
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)

        post_data_name = "posterior_stats" + "_" + str(self.parameters[0][0][0]) + "_" + str(self.parameters[0][0][1]) + ".csv"
        param_data.to_csv(stats_path/post_data_name)

    # Save posterior to csv
    def save_data(self, params):

        pop_length = len(self.parameters) 
            
        cols = ["Population", "Particles", "Epsilon", "Acceptance_Rate", "Median_Parameters", "Error",    
                "TotalDist", "ElongationDist", "DivisionDist", "LocalAnisotropyDist", "AspectRatioDiff", "DensityParamDiff"]
        cols += params
        
        data = []
        errors, median_params = self.get_error()

        for t in range(0, pop_length):
            for i in range(0, self.n_par):
                row = [t + 1]
                row.append(i)
                row.append(self.epsilon_schedule[t])
                row.append(self.acceptance_rates[t])
                row.append(median_params[t])
                row.append(errors[t])

                row.append(self.distances[t][i][0])
                row.append(self.distances[t][i][1])
                row.append(self.distances[t][i][2])
                row.append(self.distances[t][i][3])
                row.append(self.distances[t][i][4])
                row.append(self.distances[t][i][5])

                for k in range(0, len(params)):
                    row.append(self.parameters[t][i][k])

                # Add row
                data.append(row)

        data_folder = "Stats_" + str(self.parameters[0][0][0]) + "_" + str(self.parameters[0][0][1])
        param_data = pandas.DataFrame(data, columns = cols)
        print(param_data)

        # If folder doesn't exist, create new folder "Posterior_Data" to store the csv files
        stats_path = Path(__file__).parent.absolute()/'Posterior_Data'/data_folder
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)

        post_data_name = "posterior_data" + "_" + str(self.parameters[0][0][0]) + "_" + str(self.parameters[0][0][1]) + ".csv"
        
        # Export data to csv file
        param_data.to_csv(stats_path/post_data_name)

        # Graph the data

        # Kde plot
        plotter.plotKdeComp(param_data, stats_path, self.prior, params, 2, 2)
        
        # Pair plot
        plotter.pairPlot(param_data, stats_path, params)
        plotter.gridPlot(param_data, stats_path, params, self.prior)
        plotter.gridPlotVer3(param_data, stats_path, params, self.prior)

        # Plot statistics
        plotter.plotStats(param_data, stats_path, self.acceptance_rates, errors, self.epsilon_schedule)

        # Correlation matrix
        plotter.plotCovMatrixNorm(param_data, stats_path, params)

        # Plot true distirbution
        plotter.plotTrueElongDist(param_data, self.n_par, params, stats_path)

        # Plot the coefficients of variation
        plotter.plotCV(param_data, stats_path, params)

        # Plot the posterior kde for each parameter
        plotter.plotPostKde(param_data, stats_path, self.prior, params)

def main():
    # Define epsilon schedule
    first_epsilon = 25
    final_epsilon = 20

    # Define range for input
    el_mean_bounds = [0.7, 1.4]             # um/hr
    el_stdv_bounds = [0.05, 0.4]            # um/hr
    div_mean_bounds = [5, 9]                # um
    div_stdv_bounds = [0.01, 0.3]           # um
    
    k_int_bounds = [30, 80]                 # N/um
    k_cell_bounds = [30, 80]                # N/um
    k_stick_bounds = [1, 19]                # N/um
    rng_stick_bounds = [0.1, 1.1]           # um
    twist_bounds = [0.01, 0.2]              # N/um
    push_bounds = [0.01, 0.1]               # N/um

    prior = [el_mean_bounds, el_stdv_bounds, div_mean_bounds, div_stdv_bounds]
    '''prior = [el_mean_bounds, el_stdv_bounds, div_mean_bounds, div_stdv_bounds, k_int_bounds,
             k_cell_bounds, k_stick_bounds, rng_stick_bounds, twist_bounds, push_bounds]
    prior = [k_int_bounds, k_cell_bounds, k_stick_bounds, rng_stick_bounds, twist_bounds, push_bounds]'''

    # Number of populations
    n_pop = 2
    # Number of particles for each population
    n_par = 3
    # (Automated Epsilon) Specifies the quantile of the previous population distance distribution to choose as the next epsilon
    alpha = 0.6
    # (Fixed Epsilon) Specifies the epsilon schedule
    epsilon_type = "linear"#"exp"
    # Initial BSim population
    initial_pop = 1
    # BSim simulation time
    sim_time = 6.5
    # BSim simulation time step (hr)
    bsim_export_time = 0.5
    # Experiment time step (hr)
    cp_export_time = 0.5
    # Simulation dimensions
    sim_dim = (800, 600)  #(1870, 2208)
    
    # Simulation files
    bsim_data = "BSim_Simulation.csv"
    cp_data = 'BSim_Exp_Data.csv'#'BSim_Simulation_1.23_0.277_7.0_0.1-6.5.csv'

    # Paths and files required to run BSim
    bsim_jar = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\legacy\\jars\\bsim-2021.jar"
    jars = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\lib\\*"              # Gets all jars    
    driver_file = 'BSimPhageLogger.BSimPhageField'

    # Parameters
    params = ["ElongationMean", "ElongationStdv", "DivisionMean", "DivisionStdv"]
    #params = ["InternalForce", "CellCollisionForce", "XForce", "StickingRange", "Twist", "Push"]
    
    # Commands to change BSim parameter values (should correspond to params array)
    # Command names are located in BSim
    cmds = ["-el_mean", "-el_stdv", "-div_mean", "-div_stdv"]
    #cmds = ["-k_int", "-k_cell", "-k_stick", "-rng_stick", "-twist", "-push"]
    
    # Number of parameters in each particle
    n_params = len(params)
    
    abc = abcsmc(cmds, prior, first_epsilon, final_epsilon, alpha, n_par, n_pop, n_params, initial_pop, sim_time, bsim_data, cp_data,
                 bsim_jar, jars, driver_file, bsim_export_time, cp_export_time, sim_dim)
    abc.run_fixed(epsilon_type)
    #abc.run_auto()
    #abc.save_stats()
    abc.save_data(params)
    

# -------------------

main()






        
            
