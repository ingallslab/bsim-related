# abc-smc plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy.stats import uniform
from scipy import stats
import numpy as np

# Plot using pairplot
def pairPlot(df_posterior, plot_folder, params):
    
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    df_params = df_posterior[params]

    # Plot
    sns.pairplot(df_params, diag_kind = 'kde')

    # Save plot
    title = "pair_plot" + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()

# Graph the correlation matrix
def plotCorrMatrix(df_posterior, plot_folder, params):

    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    df_params = df_posterior[params]

    f = plt.figure(figsize = (19, 15))
    plt.matshow(df_params.corr(), fignum = f.number)
    plt.xticks(range(df_params.select_dtypes(['number']).shape[1]), df_params.select_dtypes(['number']).columns, fontsize = 14, rotation = 45)
    plt.yticks(range(df_params.select_dtypes(['number']).shape[1]), df_params.select_dtypes(['number']).columns, fontsize = 14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 14)
    plt.title('Correlation Matrix', fontsize = 16);

    # Save plot
    title = "Correlation" + ".png"
    plt.savefig(plot_path/title)
    
    plt.show()
    plt.close()

# Graph the covariance matrix
def plotCovmatrix(df_posterior, plot_folder, params):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    df_params = df_posterior[params]

    f = plt.figure(figsize = (19, 15))
    plt.matshow(df_params.cov(), fignum = f.number)
    plt.xticks(range(df_params.select_dtypes(['number']).shape[1]), df_params.select_dtypes(['number']).columns, fontsize = 14, rotation = 45)
    plt.yticks(range(df_params.select_dtypes(['number']).shape[1]), df_params.select_dtypes(['number']).columns, fontsize = 14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize = 14)
    plt.title('Covariance Matrix', fontsize = 16);

    # Save plot
    title = "cov1" + ".png"
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

    # Create a graph for each parameter
    fig = plt.figure(figsize = (20,10))
    ax = [i for i in range(0, n_params)]
    
    for i in range(0, n_params):
        ax[i] = (fig.add_subplot(3, 2, i + 1))
        # Plot the prior
        ax[i].plot(linspace[i], uniform_dist[i].pdf(linspace[i]), 'r-', lw = 3, alpha = 0.5, label = 'Prior')
        for n in range(1, n_pop + 1):
            df_pop = df_posterior[df_posterior["Population"] == n]
            sns.kdeplot(data = df_pop[params[i]], ax = ax[i], label = ("Population " + str(n)))

        plt.legend(loc = "upper right")
        #plt.tight_layout()

    # Plots the 'true' distribution
    c = 13.89
    mean = 1.23*c
    stdv = 0.277*c
    x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
    ax1 = (fig.add_subplot(3, 2, 5))
    ax1.plot(x, stats.norm.pdf(x, mean, stdv), 'k-', alpha = 0.5, label = "Elongation Rate")
    plt.title('Elongation Rate');

    mean = 7.0*c
    stdv = 1.0*c
    x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
    ax2 = (fig.add_subplot(3, 2, 6))
    ax2.plot(x, stats.norm.pdf(x, mean, stdv), 'k-', alpha = 0.5, label = "Division Threshold")
    plt.title('Division Threshold');

    fig.tight_layout()
            
    # Save plot
    title = "Parameter_Kde1" + ".png"
    plt.savefig(plot_path/title)
            
    plt.show()
    plt.close()

# Plots the original distributiond of parameters
def plotTrueDist(param_titles, params, plot_folder):
    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
        
    n_params = len(param_titles)
    fig = plt.figure(figsize = (20,10))
    ax = [i for i in range(0, n_params)]
    
    for i in range(0, n_params):
        ax[i] = (fig.add_subplot(2, 2, i + 1))
        mean = params[i][0]
        stdv = params[i][1]
        x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
        ax[i].plot(x, stats.norm.pdf(x, mean, stdv), label = param_titles[i])
        plt.title(param_titles[i])

    # Save plot
    title = "True Distribution " + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()

# Plot the parameters
def plotParameters(df_posterior, plot_folder):

    # If folder doesn't exist, create new folder "Posterior_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Posterior_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fig = plt.figure(figsize = (20,10))
    ax = [i for i in range(0, 2)]

    x = ["ElongationMean", "DivisionMean"]
    y = ["ElongationStdv", "DivisionStdv"]

    # Number of populations
    n_pop = df_posterior.at[df_posterior.shape[0] - 1, "Population"]
    
    for i in range(0, 2):
        ax[i] = (fig.add_subplot(1, 2, i + 1))
        for n in range(1, n_pop + 1):
            df_pop = df_posterior[df_posterior["Population"] == n]
            sns.scatterplot(data=df_pop, x=x[i], y=y[i], hue="Population")

    fig.tight_layout()

    # Save plot
    title = "Parameters" + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()
    



















