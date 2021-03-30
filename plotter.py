# abc-smc plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy.stats import uniform
from scipy import stats
import numpy as np
import math
#import pandas

# Plot using pairplot
def pairPlot(df, plot_folder, params, kind, graph_title):
    
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    # Plot
    sns.pairplot(df_params, kind = kind)
    #sns.set(xlim = (0,100), ylim = (0,100))
        
    # Save plot
    title = graph_title + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()

# Graph the covariance matrix using normalized values
def plotCovMatrixNorm(df, plot_folder, params):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    # Standardize the posterior
    norm_params = []
    for p in params:
        norm_params.append( (df_params[p] - df_params[p].min())/(df_params[p].max() - df_params[p].min()) )
    #print("norm_params: ", norm_params)

    # Compute the covariance matrix
    f = plt.figure(figsize = (19, 15))
    plt.matshow(np.cov(norm_params), fignum = f.number)
    plt.xticks(range(df_params.select_dtypes(['number']).shape[1]), df_params.select_dtypes(['number']).columns, fontsize = 14, rotation = 45)
    plt.yticks(range(df_params.select_dtypes(['number']).shape[1]), df_params.select_dtypes(['number']).columns, fontsize = 14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 14)
    plt.title('Covariance Matrix', fontsize = 16);

    # Save plot
    title = "Cov_Norm" + ".png"
    plt.savefig(plot_path/title)
    
    plt.show()
    plt.close()

    #print("np.cov(norm_params): ", np.cov(norm_params))

# Plot the coefficient of variations
def plotCV(df, plot_folder, params):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    # Calculate the coefficient of variation
    fig = plt.figure(figsize = (7, 5))
    plt.bar(params, stats.variation(df_params), alpha = 0.6)
    plt.title("Coefficients of Variation")
    #print("stats.variation(df_params): ", stats.variation(df_params))

    # Save plot
    title = "CV" + ".png"
    plt.savefig(plot_path/title)
            
    plt.show()
    plt.close()

# Plot acceptance rates
def plotAcceptance(df, acceptance_rates, plot_folder):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Number of population
    n_pop = df.at[df.shape[0] - 1, "Population"]
    pops = [i for i in range(1, n_pop + 1)]

    fig = plt.figure(figsize = (7, 5))
    plt.bar(pops, acceptance_rates, alpha = 0.6)
    plt.title("Acceptance Rates")

    # Save plot
    title = "Acceptance_Rates" + ".png"
    plt.savefig(plot_path/title)
            
    plt.show()
    plt.close()
    

# Compare parameters with prior population
def plotKdeComp(df_posterior, plot_folder, prior, params):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    # Number of populations
    n_pop = df_posterior.at[df_posterior.shape[0] - 1, "Population"]

    n_params = len(prior)
    size = 1000
    # Data for uniform prior
    linspace = []
    uniform_dist = []
    for i in range(0, n_params):
        bounds = prior[i]
        uniform_dist.append(uniform(loc = bounds[0], scale = abs(bounds[1]-bounds[0])))
        linspace.append(np.linspace(uniform_dist[i].ppf(0), uniform_dist[i].ppf(1), size))

    # Create a page of parameter plots for every 10 populations
    n_ten_pop = math.ceil(n_pop / 10)
    n_tmp = n_pop
    current_pop = 1
    for j in range(0, n_ten_pop):
    
        if ( n_tmp < 10 ):
            x = n_tmp
        else:
            x = 10
        
        fig = plt.figure(figsize = (20,10))
        ax = [i for i in range(0, n_params)]
        
        for i in range(0, n_params):           
            ax[i] = (fig.add_subplot(2, 2, i + 1))
            # Plot the prior
            ax[i].plot(linspace[i], uniform_dist[i].pdf(linspace[i]), 'r-', lw = 3, alpha = 0.5, label = 'Prior')
            for n in range(1, x + 1):
                df_pop = df_posterior[df_posterior["Population"] == current_pop]
                sns.kdeplot(data = df_pop[params[i]], ax = ax[i], label = ("Population " + str(current_pop)))

                current_pop += 1
                
            plt.legend(loc = "upper right")
            current_pop -= x

        current_pop += x
        n_tmp -= x
        
        fig.tight_layout()
                
        # Save plot
        title = "Parameter_Kde_" + str(j) + ".png"
        plt.savefig(plot_path/title)
                
        plt.show()
        plt.close()

# Plots the original distribution of parameters
def plotTrueDist(df, n_par, params, plot_folder):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]
        
    #n_params = len(params)
    fig = plt.figure(figsize = (20,10))

    c = 13.89
    mean = 1.23*c
    stdv = 0.277*c
    x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
    ax1 = (fig.add_subplot(1, 2, 1))
    ax1.plot(x, stats.norm.pdf(x, mean, stdv), 'k-', alpha = 0.5, lw = 3, label = "True Distribution")
    for i in range(0, n_par):
        post_mean = df_params["ElongationMean"].iat[i]
        post_stdv = df_params["ElongationStdv"].iat[i]
        ax1.plot(x, stats.norm.pdf(x, post_mean, post_stdv), alpha = 0.5, label = ("Posterior " + str(i + 1)))
    plt.legend(loc = "upper right")
    plt.title('Elongation Rate');

    mean = 7.0*c
    stdv = 0.1*c
    x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
    ax2 = (fig.add_subplot(1, 2, 2))
    ax2.plot(x, stats.norm.pdf(x, mean, stdv), 'k-', alpha = 0.5, lw = 3, label = "True Distribution")
    for i in range(0, n_par):
        post_mean = df_params["DivisionMean"].iat[i]
        post_stdv = df_params["DivisionStdv"].iat[i]
        ax2.plot(x, stats.norm.pdf(x, post_mean, post_stdv), alpha = 0.5, label = ("Posterior " + str(i + 1)))
    plt.legend(loc = "upper right")
    plt.title('Division Threshold')
            
    # Save plot
    title = "True_Distribution " + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()


















