# Approximate Bayesian Computation

import inferParameters
from abcsysbio import EpsilonSchedule
from abcsysbio.KernelType import KernelType
from abcsysbio import kernels
from abcsysbio import statistics

import subprocess
import os
import math
import random
import pandas
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd
import scipy.stats as st
#from scipy.stats import norm

# Execute BSim simulation with given parameters
def executeSimulation(params, pop, sim_time):
    params = convertParams(params)
    
    bsim_jar = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\legacy\\jars\\bsim-2021.jar"
    jars = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\lib\\*"              # Gets all jars    
    program_path = "C:\\Users\\sheng\\eclipse_workspace_java\\bsim-ingallslab\\run\\BSimPhageLogger\\BSimPhageField.java"
        
    cmd = ["java", "-cp", bsim_jar + ";" + jars, program_path, "-el_mean", params[0], "-el_stdv", params[1],
           "-div_mean", params[2], "-div_stdv", params[3], "-pop", pop, "-simt", sim_time]

    process = subprocess.Popen(cmd)
    process.wait()          # Waits until the simulation has been completed

# Parameter conversion
def convertParams(params):
    pixel_to_um_ratio = 13.89

    # Apply conversion rate to 3 decimal places
    params = tuple( round(i / pixel_to_um_ratio, 3) for i in params)
    # Convert the parameters into string to use in the command line
    params = tuple(str(i) for i in params)
    
    return params

# Gets a value (uniform) between boundaries, rounded to 3 decimal places
def getRandomValue(bounds):
    return round(random.uniform(bounds[0], bounds[1]), 3)

# Determine the component wise normal 
def component_wise_normal_kernel(current_params, kernel):
    pert_params = []
    #print("kernel[2]: ", kernel[2])
    for n in kernel[0]:
        # Draws random sample from normal distribution rounded to 3 decimal places
        pert_params.append(round(rnd.normal(current_params[n], np.sqrt(kernel[2][n])), 3))
    return pert_params

# Calculate the weight of a single particle
def calcWeight(prior, prev_weights, curr_params, prev_pop, kernel, n_par, t):
    if ( t == 0 ):
        return 1.0
    else:
        numer = 1.0
        for n in kernel[0]:     # Prior is uniform
            numer *= 1/(prior[n][1] - prior[n][0])
             
        denom = 0.0
        for i in range (0, n_par):
            kernel_pdf = getPdfKernel(kernel, curr_params, prev_pop[i])
            denom += prev_weights[i] * kernel_pdf
            
        print("numer/denom: ", numer/denom)
        return numer/denom

# Get the probability density of the current value from the pdf of the kernel
def getPdfKernel(kernel, curr_params, prev_params):
    prob_density = 1.0
    for n in kernel[0]:
        mean = prev_params[n]
        stdv = np.sqrt(kernel[2][n])
        kern = st.norm.pdf(curr_params[n], mean, stdv)
        prob_density *= kern
    #print("prob_density: ", prob_density)
    return prob_density
    
# Normalize the weights
def normalizeWeights(weights, len_pop):
    total = sum( weights )
    for i in range(0, len_pop):
        weights[i] = weights[i]/float(total)
    
# ABC-SMC
def abc_smc(prior, first_epsilon, final_epsilon, n_par):

    # Construct epsilon schedule
    tol_type = "linear"
    n_iter = 2
    epsilonSchedule = EpsilonSchedule.EpsilonSchedule(tol_type, final_epsilon, first_epsilon, n_iter).tol
    print("epsilonSchedule: ", epsilonSchedule)
    
    # Current epsilon value for distance threshold
    epsilon = epsilonSchedule[0]

    # The list of all accepted parameters and their distances
    parameters = []
    distances = []
    print("parameters empty: ", parameters)

    # The current and previous population of accepted particles
    curr_population = []
    prev_population = []
    # The weights of current and previous accepted particles in the population
    curr_weights = []
    prev_weights = []
    # The current accepted distances in the population
    curr_distances = []

    # File names
    bsim_data_name = "Infection_Simulation.csv"
    cp_data_name = 'Infection_Simulation_params_1.23_0.277_7.0_0.1.csv'

    # Initial Bacteria Population
    initial_pop = str(1)
    # Simulation Time
    sim_time = str(6.5)

    # Define kernel type and index of parameters
    kernel_type = 2                         # component-wise normal kernel
    kernel_list = [[0, 1, 2, 3],[],[]]      # kernel_list[0] specifies the index of parameters used for abcsysbio
    print("kernel_type: ", kernel_type)
    kernel = kernels.getKernel(kernel_type, kernel_list, np.array([[]]), prev_weights)
    print("kernel[2]: ", kernel[2])

    # Population indicator
    t = 0
    # Particle indicator
    n = 0

    # Generate a parameter vector from specified bounds of prior distribution
    current_params = [getRandomValue(prior[0]), getRandomValue(prior[1]), getRandomValue(prior[2]), getRandomValue(prior[3])]
    
    # Search for acceptable parameters for all epsilon
    while ( epsilon >= final_epsilon ):

        print("current parameters: ", current_params)

        # Run BSim simulation with given parameters
        executeSimulation(current_params, initial_pop, sim_time)
        # Get the elongation and division distances
        elongation_dist, division_dist = inferParameters.run(bsim_data_name, cp_data_name, convertParams(current_params), export_data = False, export_plots = False)
        # Get the weighted total distance
        total_dist = elongation_dist*0.5 + division_dist*0.5
    
        print("elongation_dist: ", elongation_dist, "division_dist: ", division_dist, "total_dist: ", total_dist)

        # Accept the current parameters if the discrepancy between the simulated data and experimental data is less than the threshold
        if ( total_dist < epsilon ):
            
            curr_population.append(current_params)
            curr_weights.append(calcWeight(prior, prev_weights, current_params, prev_population, kernel, n_par, t))
            curr_distances.append((elongation_dist, division_dist, total_dist))
            
            print("curr_weights: ", curr_weights)
            print("current pop: ", curr_population)
            print("curr_distances: ", curr_distances)

            # Increase the number of particles in the population
            n = n + 1

            # If number of particles in population has reached the max 
            if ( n == n_par ):

                # Increase number of populations
                t = t + 1

                # Determine the next threshold
                if ( t < len(epsilonSchedule) ):
                    epsilon = epsilonSchedule[t]
                else:
                    epsilon = -1

                # Add current population of accepted particles to total parameters
                parameters.append(curr_population)
                # Add current populatino of accepted particle distances to total distances
                distances.append(curr_distances)
                curr_distances = []
                
                # Reset population arrays
                prev_population = curr_population
                curr_population = []

                # Reset weight arrays
                prev_weights = curr_weights
                curr_weights = []

                # Normalize the weights
                normalizeWeights(prev_weights, len(prev_population))
                print("normalize prev_weights: ", prev_weights)

                # Update kernel using previous population
                kernel = kernels.getKernel(kernel_type, kernel_list, np.array(prev_population), prev_weights)
                print("kernel[2] (variance): ", kernel[2])

                # Reset population counter
                n = 0

        # Continue to sample from prior
        if ( t == 0 ):
            current_params = [getRandomValue(prior[0]), getRandomValue(prior[1]), getRandomValue(prior[2]), getRandomValue(prior[3])]
            print("t==0; current params: ", current_params)

        # Sample from the previous population with associated weights
        else:
            current_params = random.choices(population = prev_population, weights = prev_weights, k = 1)[0]   # returns a list
            print("t!=0; before pert current params: ", current_params)
            
            # Perturb the particle
            current_params = component_wise_normal_kernel(current_params, kernel)
            print("t!=0; perturbed current params: ", current_params)
        
        print("n: ", n)
        print("t: ", t)
        print("epsilon: ", epsilon)
            
    return parameters, distances

# Save posterior to csv
def saveData(parameters, distances, n_par):

    pop_length = len(parameters) 
    print(pop_length)
        
    cols = ["Population", "Particles", "ElongationMean", "ElongationStdv", "DivisionMean", "DivisionStdv", "ElongationDist", "DivisionDist",
            "TotalDist"]
    data = []

    n_particles = 1
    for t in range(1, pop_length + 1):
        for i in range(1, n_par + 1):
            row = [t]
            row.append(n_particles)
            row.append(parameters[t - 1][i - 1][0])
            row.append(parameters[t - 1][i - 1][1])
            row.append(parameters[t - 1][i - 1][2])
            row.append(parameters[t - 1][i - 1][3])

            row.append(distances[t - 1][i - 1][0])
            row.append(distances[t - 1][i - 1][1])
            row.append(distances[t - 1][i - 1][2])

            n_particles += 1

            # Add row
            data.append(row)

    # Export data to csv file
    posterior_data = pandas.DataFrame(data, columns = cols)
    print(posterior_data)

    # If folder doesn't exist, create new folder "Posterior_Data" to store the csv files
    comp_path = Path(__file__).parent.absolute()/'Posterior_Data' 
    if not os.path.exists(comp_path):
        os.makedirs(comp_path)

    post_data_name = "posterior_data" + "_" + str(parameters[0][0]) + ".csv"
    posterior_data.to_csv(comp_path/post_data_name)

    plot_folder = "Plots_" + str(parameters[0][0])
    # Pair plot
    pairPlot(posterior_data, plot_folder)

    # Correlation matrix
    plotCorrMatrix(posterior_data, plot_folder)
    plotCovmatrix(posterior_data, plot_folder)

    # Distplot
    plotPriorComparison(posterior_data, plot_folder)

# Plot using pairplot
def pairPlot(df_posterior, plot_folder):
    
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Plot
    sns.pairplot(df_posterior, diag_kind = 'hist')

    # Save plot
    title = "default" + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()

def plotCorrMatrix(df_posterior, plot_folder):

    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
        
    #plt.matshow(df_posterior.corr())

    f = plt.figure(figsize = (19, 15))
    plt.matshow(df_posterior.corr(), fignum = f.number)
    plt.xticks(range(df_posterior.select_dtypes(['number']).shape[1]), df_posterior.select_dtypes(['number']).columns, fontsize = 14, rotation = 45)
    plt.yticks(range(df_posterior.select_dtypes(['number']).shape[1]), df_posterior.select_dtypes(['number']).columns, fontsize = 14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 14)
    plt.title('Correlation Matrix', fontsize = 16);

    # Save plot
    title = "corr1" + ".png"
    plt.savefig(plot_path/title)
    
    plt.show()
    plt.close()

def plotCovmatrix(df_posterior, plot_folder):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    f = plt.figure(figsize = (19, 15))
    plt.matshow(df_posterior.cov(), fignum = f.number)
    plt.xticks(range(df_posterior.select_dtypes(['number']).shape[1]), df_posterior.select_dtypes(['number']).columns, fontsize = 14, rotation = 45)
    plt.yticks(range(df_posterior.select_dtypes(['number']).shape[1]), df_posterior.select_dtypes(['number']).columns, fontsize = 14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 14)
    plt.title('Covariance Matrix', fontsize = 16);

    # Save plot
    title = "cov1" + ".png"
    plt.savefig(plot_path/title)
    
    plt.show()
    plt.close()

# Compare parameters with prior population
def plotPriorComparison(df_posterior, plot_folder):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    # Number of populations
    n_pop = df_posterior.at[df_posterior.shape[0] - 1, "Population"]

    # Data for prior
    df_prior_el_mean = df_posterior[df_posterior["Population"] == 1].ElongationMean
    df_prior_el_stdv = df_posterior[df_posterior["Population"] == 1].ElongationStdv
    df_prior_div_mean = df_posterior[df_posterior["Population"] == 1].DivisionMean
    df_prior_div_stdv = df_posterior[df_posterior["Population"] == 1].DivisionStdv

    fig = plt.figure()

    # Create a graph for each population
    for n in range(2, n_pop + 1):
        df_pop = df_posterior[df_posterior["Population"] == n]

        ax1 = fig.add_subplot(221)
        sns.histplot(data = df_pop.ElongationMean, ax = ax1)
        sns.kdeplot(data = df_prior_el_mean, ax = ax1)
        plt.legend(['Prior', 'Population ' + str(n)])

        ax2 = fig.add_subplot(222)
        sns.histplot(data = df_pop.ElongationStdv, ax = ax2)
        sns.kdeplot(data = df_prior_el_stdv, ax = ax2)
        plt.legend(['Prior', 'Population ' + str(n)])

        ax3 = fig.add_subplot(223)
        sns.histplot(data = df_pop.DivisionMean, ax = ax3)
        sns.kdeplot(data = df_prior_div_mean, ax = ax3)
        plt.legend(['Prior', 'Population ' + str(n)])

        ax4 = fig.add_subplot(224)
        sns.histplot(data = df_pop.DivisionStdv, ax = ax4)
        sns.kdeplot(data = df_prior_div_stdv, ax = ax4)
        plt.legend(['Prior', 'Population ' + str(n)])

        # Save plot
        title = "Population " + str(n) + ".png"
        plt.savefig(plot_path/title)
        
        plt.show()
        plt.close()
        

# Main function
def main():

    # Define epsilon schedule
    first_epsilon = 8
    final_epsilon = 5

    # Define range for input
    el_mean_bounds = [15, 18]               
    el_stdv_bounds = [3, 4]

    div_mean_bounds = [95, 98]
    div_stdv_bounds = [0, 1.5]

    prior = [el_mean_bounds, el_stdv_bounds, div_mean_bounds, div_stdv_bounds]

    # The list of acceptable parameters and their distances
    parameters = []
    distances = []

    # Number of particles for each population
    n_par = 5          

    # Run the ABC method
    parameters, distances = abc_smc(prior, first_epsilon, final_epsilon, n_par)
    print("parameters: ", parameters)
    print("distances: ", distances)

    # Save to csv
    saveData(parameters, distances, n_par)


# ----------------------------


main()






