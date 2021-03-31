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
                 cp_data):
        self.prior = prior
        self.first_epsilon = first_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = [self.first_epsilon]
        self.alpha = alpha
        self.n_par = n_par
        self.n_pop = n_pop
        self.n_params = n_params
        self.initial_pop = initial_pop
        self.sim_time = sim_time
        self.bsim_data = bsim_data
        self.cp_data = cp_data
        
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

    # Gets a value (uniform) between boundaries, rounded to 3 decimal places
    def getRandomValue(self, bounds):
        return round(random.uniform(bounds[0], bounds[1]), 3)

    # Parameter conversion
    def convertParams(self, params):
        pixel_to_um_ratio = 13.89

        # Apply conversion rate to 3 decimal places
        params = tuple( round(i / pixel_to_um_ratio, 3) for i in params)
        # Convert the parameters into string to use in the command line
        params = tuple(str(i) for i in params)
        
        return params

    # Execute BSim simulation with given parameters
    def executeSimulation(self, params):
        params = self.convertParams(params)
        
        bsim_jar = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\legacy\\jars\\bsim-2021.jar"
        jars = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\lib\\*"              # Gets all jars    
        program_path = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\run\\BSimPhageLogger\\BSimPhageField.java"
            
        cmd = ["java", "-cp", bsim_jar + ";" + jars, program_path, "-el_mean", params[0], "-el_stdv", params[1],
               "-div_mean", params[2], "-div_stdv", params[3], "-pop", str(self.initial_pop), "-simt", str(self.sim_time)]

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
                
            #print("numer/denom: ", numer/denom)
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

        # Increase or decrease alpha depending on the acceptance rate
        if ( pop > 1 and self.alpha < 0.90 and self.alpha > 0.1):
            if ( 100*((self.acceptance_rates[pop] - self.acceptance_rates[pop -1])/self.acceptance_rates[pop - 1]) < -40):
                self.alpha += 0.05
            elif ( 100*((self.acceptance_rates[pop] - self.acceptance_rates[pop -1])/self.acceptance_rates[pop - 1]) > 40):
                self.alpha -= 0.05
            print("self.alpha: ", self.alpha)

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
                curr_pop, curr_dist = self.iterate_population(epsilon, pop, True, True)
            else:
                curr_pop, curr_dist = self.iterate_population(epsilon, pop, False, False)

            # Add current population of accepted particles to total parameters
            self.parameters.append(curr_pop)
            # Add current population of accepted particle distances to total distances
            self.distances.append(curr_dist)

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
                current_params = [self.getRandomValue(self.prior[0]), self.getRandomValue(self.prior[1]), self.getRandomValue(self.prior[2]), self.getRandomValue(self.prior[3])]
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
                elongation_dist, division_dist, local_anisotropy_dist, aspect_ratio_diff, density_parameter_diff = inferParameters.run(self.bsim_data, self.cp_data, self.convertParams(current_params), export_data, export_plots)
            # Get the total distance
            total_dist = elongation_dist + division_dist + local_anisotropy_dist + abs(aspect_ratio_diff) + abs(density_parameter_diff)
            
            print("elongation_dist: ", elongation_dist, "division_dist: ", division_dist)
            print("local_anisotropy_dist: ", local_anisotropy_dist, "aspect_ratio_diff: ", aspect_ratio_diff, "density_parameter_diff: ", density_parameter_diff)
            print("total_dist: ", total_dist)

            print("total_dist={} epsilon={} n={} n_par={}\ntotal_dist < epsilon: {}".format(total_dist, epsilon, n, self.n_par, (total_dist < epsilon)))
            # Accept the current parameters if the discrepancy between the simulated data and experimental data is less than the threshold
            if ( total_dist < epsilon ):
                
                curr_population.append(current_params)
                curr_weights.append(self.calcWeight(current_params, t))
                curr_distances.append((total_dist, elongation_dist, division_dist, local_anisotropy_dist, aspect_ratio_diff, density_parameter_diff))
                
                print("curr_weights: ", curr_weights)
                print("current pop: ", curr_population)
                print("curr_distances: ", curr_distances)

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
        print("normalize prev_weights: ", self.prev_weights)

        # Update kernel using previous population
        self.kernel = kernels.getKernel(self.kernel_type, self.kernel_list, np.array(self.prev_population), self.prev_weights)
        print("kernel[2] (variance): ", self.kernel[2])

        return curr_population, curr_distances

    # Find the error between the median parameter of every population and the true value
    def get_error(self):
        n_pop = len(self.parameters)
        median_params = []
        for i in range(0, n_pop):
            sorted_params = np.sort(self.parameters[i], axis = 0)
            ind = round(len(sorted_params)/2)
            median_params.append(sorted_params[ind])

        print("median_params: ", median_params)

        c = 13.89
        true_values = [1.23*c, 0.277*c, 7.0*c, 0.1*c]
        errors = []
        for i in range(0, len(median_params)):
            error = 0
            for j in range(0, len(true_values)):
                error += ( median_params[i][j] - true_values[j] )**2
                print(( median_params[i][j] - true_values[j] )**2)
            errors.append(error)

        print("errors: ", errors)
            
        return errors, median_params

    # Save posterior to csv
    def save_data(self):

        pop_length = len(self.parameters) 
            
        cols = ["Population", "Particles", "Epsilon", "Acceptance_Rate", "Median_Parameters", "Errors", "ElongationMean", "ElongationStdv", "DivisionMean",    
                "DivisionStdv", "TotalDist", "ElongationDist", "DivisionDist", "LocalAnisotropyDist", "AspectRatioDiff", "DensityParamDiff"]
        data = []
        errors, median_params = self.get_error()

        
        #n_particles = 1
        for t in range(1, pop_length + 1):
            for i in range(1, self.n_par + 1):
                row = [t]
                row.append(i)
                print("t: ", t)
                row.append(self.epsilon_schedule[t - 1])
                row.append(self.acceptance_rates[t - 1])
                row.append(median_params[t - 1])
                row.append(errors[t - 1])
                
                row.append(self.parameters[t - 1][i - 1][0])
                row.append(self.parameters[t - 1][i - 1][1])
                row.append(self.parameters[t - 1][i - 1][2])
                row.append(self.parameters[t - 1][i - 1][3])

                row.append(self.distances[t - 1][i - 1][0])
                row.append(self.distances[t - 1][i - 1][1])
                row.append(self.distances[t - 1][i - 1][2])
                row.append(self.distances[t - 1][i - 1][3])
                row.append(self.distances[t - 1][i - 1][4])
                row.append(self.distances[t - 1][i - 1][5])

                #n_particles += 1

                # Add row
                data.append(row)

        # Export data to csv file
        param_data = pandas.DataFrame(data, columns = cols)
        print(param_data)

        # If folder doesn't exist, create new folder "Posterior_Data" to store the csv files
        comp_path = Path(__file__).parent.absolute()/'Posterior_Data' 
        if not os.path.exists(comp_path):
            os.makedirs(comp_path)

        post_data_name = "posterior_data" + "_" + str(self.parameters[0][0]) + ".csv"
        param_data.to_csv(comp_path/post_data_name)

        # Graph the data
        plot_folder = "Plots_" + str(self.parameters[0][0])
        params = ["ElongationMean", "ElongationStdv", "DivisionMean", "DivisionStdv"]

        # Kde plot
        plotter.plotKdeComp(param_data, plot_folder, self.prior, params)
        
        # Pair plot
        plotter.pairPlot(param_data, plot_folder, params, "kde", "Kde")

        # Correlation matrix
        plotter.plotCovMatrixNorm(param_data, plot_folder, params)

        # Plot true distirbution
        plotter.plotTrueDist(param_data, self.n_par, params, plot_folder)

        # Plot the coefficients of variation
        plotter.plotCV(param_data, plot_folder, params)

        # Plot the acceptance rate
        plotter.plotAcceptance(param_data, self.acceptance_rates, plot_folder)

        # Plot the posterior kde for each parameter
        plotter.plotPostKde(param_data, plot_folder, self.prior, params)

def main():
    # Define epsilon schedule
    first_epsilon = 40#14
    final_epsilon = 12#6#5.5

    # Define range for input
    el_mean_bounds = [10, 20]#[14, 20]           
    el_stdv_bounds = [1, 6] 
    div_mean_bounds = [90, 105]
    div_stdv_bounds = [0, 3]
    prior = [el_mean_bounds, el_stdv_bounds, div_mean_bounds, div_stdv_bounds]

    # Number of populations
    n_pop = 2#2
    # Number of particles for each population
    n_par = 6#10
    # Number of parameters in each particle
    n_params = 4
    # (Automated Epsilon) Specifies the quantile of the previous population distance distribution to choose as the next epsilon
    alpha = 0.5
    # (Fixed Epsilon) Specifies the epsilon schedule
    epsilon_type = "linear"
    # Initial BSim population
    initial_pop = 1
    # BSim simulation time
    sim_time = 6.5
    
    # Simulation files
    bsim_data = "BSim_Simulation.csv"
    cp_data = 'BSim_Simulation_1.23_0.277_7.0_0.1-6.5.csv'#'BSim_Simulation-0.5.csv'
    
    abc = abcsmc(prior, first_epsilon, final_epsilon, alpha, n_par, n_pop, n_params, initial_pop, sim_time, bsim_data, cp_data)
    abc.run_fixed(epsilon_type)
    #abc.run_auto()
    abc.save_data()

    #print("abc.parameters:\n", abc.parameters)
    #print("abc.distances:\n", abc.distances)

# -------------------

main()






        
            
