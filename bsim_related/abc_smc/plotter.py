# abc-smc plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
from pathlib import Path
import os
from scipy.stats import uniform
from scipy import stats
import numpy as np
import math
from itertools import combinations
#import pandas

# Plot using pairplot
def pairPlot(df, plot_path, params):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    # Plot
    sns.pairplot(df_params, kind = 'kde')
        
    # Save plot
    title = "PairPlot" + ".png"
    plt.savefig(plot_path/title)

    #plt.show()
    plt.close()

'''# Plot the offdiagonal kdes in seperate graphs
def plotOffDiagSeperate(df, plot_path, params):
    
    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    n_params = len(params)
    
    # Scale the values
    for i in range(0, n_params):
        df_params[params[i]] /= df_params[params[i]].mean()
    print("df_params:\n", df_params)

    fig, ax = plt.subplots(nrows = 3, ncols = 2, sharex = True, sharey = True)
    ax = ax.flatten()
    n_off_diag = (n_params * (n_params - 1))//2

    # Plot Kdes
    x = 0   
    for combo in combinations(params, 2):
        g = sns.kdeplot(data = df_params, x = combo[0], y = combo[1], ax = ax[x], label = (combo[0] + " vs " + combo[1]))
        g.set(xlabel = None)
        g.set(ylabel = None)
        ax[x].legend(loc = "upper right", prop={'size': 6})
        # Increase axes index
        x += 1
    fig.tight_layout()
    
    # Save plot
    title = "offDiag_kde_sep_shareAx" + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()'''

# Plot scaled kdes. All plots share an axis
def gridPlot(df, plot_path, params, prior):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    n_params = len(params)
    
    # Scale the values
    for i in range(0, n_params):
        df_params[params[i]] /= df_params[params[i]].mean()
    #print("df_params:\n", df_params)

    size = 1000
    # Data for uniform prior
    linspace = []
    uniform_dist = []
    for i in range(0, n_params):
        bounds = prior[i]/df_posterior[params[i]].mean()#scaled
        uniform_dist.append(uniform(loc = bounds[0], scale = abs(bounds[1]-bounds[0])))
        linspace.append(np.linspace(uniform_dist[i].ppf(0), uniform_dist[i].ppf(1), size))

    fig, ax = plt.subplots(nrows = n_params, ncols = n_params, sharex = True, sharey = True, figsize = (12,10))
    ax = ax.flatten()

    # Plot Kdes
    x = 0 
    for i in range(0, n_params):
        for j in range(0, n_params):
            if ( i == j ):
                g = sns.kdeplot(data = df_params, x = params[i], ax = ax[x], label = params[i])
                ax[x].plot(linspace[i], uniform_dist[i].pdf(linspace[i]), lw = 2, alpha = 0.7, label = (params[i] + " Prior"))
            else:
                g = sns.kdeplot(data = df_params, x = params[j], y = params[i], ax = ax[x], label = (params[j] + " vs " + params[i]))
            '''g.set(xlabel = None)
            g.set(ylabel = None)
            ax[x].legend(loc = "upper right", prop={'size': 8})'''
            # Increase axes index
            x += 1
    fig.tight_layout()
    
    # Save plot
    title = "grid_plot" + ".png"
    plt.savefig(plot_path/title)

    #plt.show()
    plt.close()

'''# Plot scaled kdes. Each plot has its own axis
def gridPlotVer2(df, plot_path, params, prior):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    n_params = len(params)
    
    # Scale the values
    for i in range(0, n_params):
        df_params[params[i]] /= df_params[params[i]].mean()
    print("df_params:\n", df_params)

    size = 1000
    # Data for uniform prior
    linspace = []
    uniform_dist = []
    for i in range(0, n_params):
        bounds = prior[i]/df_posterior[params[i]].mean()#scaled
        uniform_dist.append(uniform(loc = bounds[0], scale = abs(bounds[1]-bounds[0])))
        linspace.append(np.linspace(uniform_dist[i].ppf(0), uniform_dist[i].ppf(1), size))

    fig = plt.figure(figsize = (12,10))
    gs1 = gs.GridSpec(nrows = n_params, ncols = n_params)
    ax = []

    # Plot Kdes
    x = 0 
    for i in range(0, n_params):
        for j in range(0, n_params):
            ax.append(plt.subplot(gs1[i, j]))
            if ( i == j ):
                g = sns.kdeplot(data = df_params, x = params[i], ax = ax[x], label = params[i])
                ax[x].plot(linspace[i], uniform_dist[i].pdf(linspace[i]), lw = 2, alpha = 0.7, label = (params[i] + " Prior"))
            else:
                g = sns.kdeplot(data = df_params, x = params[j], y = params[i], ax = ax[x], label = (params[j] + " vs " + params[i]))
            g.set(xlabel = None)
            g.set(ylabel = None)
            ax[x].legend(loc = "upper right", prop={'size': 8})
            # Increase axes index
            x += 1
    fig.tight_layout()
    
    # Save plot
    title = "grid_plotv2" + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()'''

# Plot scaled kdes. Disgonals and offdiagonals share seperate axes
def gridPlotVer3(df, plot_path, params, prior):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    n_params = len(params)
    
    # Scale the values
    for i in range(0, n_params):
        df_params[params[i]] /= df_params[params[i]].mean()
    #print("df_params:\n", df_params)

    size = 1000
    # Data for uniform prior
    linspace = []
    uniform_dist = []
    for i in range(0, n_params):
        bounds = prior[i]/df_posterior[params[i]].mean()#scaled
        uniform_dist.append(uniform(loc = bounds[0], scale = abs(bounds[1]-bounds[0])))
        linspace.append(np.linspace(uniform_dist[i].ppf(0), uniform_dist[i].ppf(1), size))

    fig = plt.figure(figsize = (12,10))
    gs1 = gs.GridSpec(nrows = n_params, ncols = n_params)

    ax = []
    for i in range(0, n_params):
        for j in range(0, n_params):
            if ( i == j and len(ax) != 0 ):
                ax.append(plt.subplot(gs1[i, j], sharex = ax[0], sharey = ax[0]))
            elif ( len(ax) > 1 ):
                ax.append(plt.subplot(gs1[i, j], sharex = ax[1], sharey = ax[1]))
            else:
                ax.append(plt.subplot(gs1[i, j]))

    # Plot Kdes
    x = 0 
    for i in range(0, n_params):
        for j in range(0, n_params):
            if ( i == j ):
                g = sns.kdeplot(data = df_params, x = params[i], ax = ax[x], label = params[i])
                ax[x].plot(linspace[i], uniform_dist[i].pdf(linspace[i]), lw = 2, alpha = 0.7, label = (params[i] + " Prior"))
            else:
                g = sns.kdeplot(data = df_params, x = params[j], y = params[i], ax = ax[x], label = (params[j] + " vs " + params[i]))
            g.set(xlabel = None)
            g.set(ylabel = None)
            ax[x].legend(loc = "upper right", prop={'size': 8})
            # Increase axes index
            x += 1
    fig.tight_layout()
    
    # Save plot
    title = "grid_plotv3" + ".png"
    plt.savefig(plot_path/title)

    #plt.show()
    plt.close()


# Plot the acceptance rate, error, and epsilon against the population
def plotStats(df, plot_path, acceptance_rate, errors, epsilons):

    n_pop = df.at[df.shape[0] - 1, "Population"]
    population = [(i + 1) for i in range(0, n_pop)]

    print("epsilons: ", epsilons)
    print("population: ", population)
    #fig = plt.figure(figsize = (15, 8))
    plt.plot(population, acceptance_rate, '-o', color = 'g', label = "Acceptance Rate")
    plt.plot(population, errors, '-o', color = 'r', label = "Error")
    plt.plot(population, epsilons, '-o', color = 'b', label = "Epsilon")
    plt.legend(loc = "upper right")

    # Save plot
    title = "Stats" + ".png"
    plt.savefig(plot_path/title)
                
    plt.show()
    plt.close()

# Plots the kde of the posterior for each parameter
def plotPostKde(df, plot_path, prior, params):
    
    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    #fig, ax = plt.subplots(figsize = (15,10))
    fig = plt.figure(figsize = (15,10))
    ax = fig.add_subplot(111)

    size = 1000
    n_params = len(params)
    # Data for uniform prior
    linspace = []
    uniform_dist = []
    for i in range(0, n_params):
        bounds = prior[i]/df_posterior[params[i]].mean()#scaled
        uniform_dist.append(uniform(loc = bounds[0], scale = abs(bounds[1]-bounds[0])))
        linspace.append(np.linspace(uniform_dist[i].ppf(0), uniform_dist[i].ppf(1), size))

    #cm = plt.get_cmap('gist_rainbow')
    cm = plt.get_cmap('Set3')
    for i in range(0, len(params)):
        # Plot the priors
        ax.plot(linspace[i], uniform_dist[i].pdf(linspace[i]), lw = 2, alpha = 0.7, color = cm(i), label = (params[i] + " Prior"))
        # Plot the kdes
        scaled_data = df_posterior[params[i]]/df_posterior[params[i]].mean()
        g = sns.kdeplot(data = scaled_data, color = cm(i), ax = ax, label = params[i])
    g.set(xlabel = None)
                
    ax.legend(loc = "upper right")
    plt.title("Posterior Parameter Kde")
                
    # Save plot
    title = "Posterior_Kde" + ".png"
    plt.savefig(plot_path/title)
                
    plt.show()
    plt.close()

# Graph the covariance matrix using normalized values
def plotCovMatrixNorm(df, plot_path, params):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    # Standardize the posterior
    norm_params = []
    for p in params:
        norm_params.append( (df_params[p] - df_params[p].min())/(df_params[p].max() - df_params[p].min()) )

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

# Plot the coefficient of variations
def plotCV(df, plot_path, params):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]

    # Calculate the coefficient of variation
    fig = plt.figure(figsize = (7, 5))
    plt.bar(params, stats.variation(df_params), alpha = 0.6)
    plt.title("Coefficients of Variation")
    plt.xticks(fontsize = 5)#, rotation = 90)

    # Save plot
    title = "CV" + ".png"
    plt.savefig(plot_path/title)
            
    #plt.show()
    plt.close()

'''# Plot acceptance rates
def plotAcceptance(df, acceptance_rates, plot_path):

    # Number of population
    n_pop = df.at[df.shape[0] - 1, "Population"]
    pops = [i for i in range(1, n_pop + 1)]

    fig = plt.figure(figsize = (7, 5))
    plt.bar(pops, acceptance_rates, alpha = 0.6)
    plt.title("Acceptance Rates")

    # Save plot
    title = "Acceptance_Rates" + ".png"
    plt.savefig(plot_path/title)
            
    #plt.show()
    plt.close()'''
    
# Compare parameters with prior population
def plotKdeComp(df_posterior, plot_path, prior, params, rows, cols ):
    
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
            ax[i] = (fig.add_subplot(rows, cols, i + 1))
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

# Plots the original distribution of elongation rate and division threshold
def plotTrueElongDist(df, n_par, params, plot_path):

    # Obtain the posterior
    n_pop = df.at[df.shape[0] - 1, "Population"]
    df_posterior = df[df["Population"] == n_pop]
    df_params = df_posterior[params]
        
    fig = plt.figure(figsize = (20,10))

    mean = 1.23
    stdv = 0.277
    x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
    ax1 = (fig.add_subplot(1, 2, 1))
    ax1.plot(x, stats.norm.pdf(x, mean, stdv), 'k-', alpha = 0.5, lw = 3, label = "True Distribution")
    for i in range(0, n_par):
        post_mean = df_params["ElongationMean"].iat[i]
        post_stdv = df_params["ElongationStdv"].iat[i]
        ax1.plot(x, stats.norm.pdf(x, post_mean, post_stdv), alpha = 0.5, label = ("Particle " + str(i + 1)))
    plt.legend(loc = "upper right")
    plt.title('Posterior Elongation Rates');

    mean = 7.0
    stdv = 0.1
    x = np.linspace(mean - 3*stdv, mean + 3*stdv, 100)
    ax2 = (fig.add_subplot(1, 2, 2))
    ax2.plot(x, stats.norm.pdf(x, mean, stdv), 'k-', alpha = 0.5, lw = 3, label = "True Distribution")
    for i in range(0, n_par):
        post_mean = df_params["DivisionMean"].iat[i]
        post_stdv = df_params["DivisionStdv"].iat[i]
        ax2.plot(x, stats.norm.pdf(x, post_mean, post_stdv), alpha = 0.5, label = ("Particle " + str(i + 1)))
    plt.legend(loc = "upper right")
    plt.title('Posterior Division Thresholds')
            
    # Save plot
    title = "True_Distribution " + ".png"
    plt.savefig(plot_path/title)

    plt.show()
    plt.close()


















