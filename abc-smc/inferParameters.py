# Infer elongation rate and division threshold parameters from csv file.

import math
import pandas
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from pathlib import Path
import os

# ---------------------------- Elongation Rate ----------------------------------
# Find the change between the lengths of non-infected bacteria at each timestep

def getElongationRate(LengthColTitle, TimeColTitle, data):
    # Get the total number of objects in the simulation
    obj_count = data.at[data.shape[0] - 1, "ObjectNumber"]

    # Store the average elongation rate for each bacteria
    avg_elongation_rates = []

    # Loop through the number of objects
    for obj_number in range(1, obj_count + 1):
        elongation_rates = []
        
        df_bsim = data[data["ObjectNumber"] == obj_number]
        prev_length = df_bsim["Length"].iat[0]
        prev_time = df_bsim["SimulationTime"].iat[0]

        for row in df_bsim.itertuples():

            # To account only for elongation events ( should be positive; disregard changes in length during division and infection )
            change_in_length = row.Length - prev_length
            change_in_time = row.SimulationTime - prev_time
            if ( change_in_length >= 0 and change_in_time > 0) :
                elongation_rates.append(change_in_length / change_in_time)

            # Update previous values
            prev_length = row.Length
            prev_time = row.SimulationTime

        # Find the average elongation rate for a bacterium
        avg_elongation = 0.0
        if ( len(elongation_rates) > 0 ) :
            for i in range (0, len(elongation_rates)):
                avg_elongation = avg_elongation + elongation_rates[i]
            avg_elongation = avg_elongation / len(elongation_rates)

            avg_elongation_rates.append(avg_elongation)
        else:
            avg_elongation_rates.append("-")

    return avg_elongation_rates

# ---------------------------- Division Threshold -------------------------------
# Find the length of the bacteria before the population increases and the length of the bacteria decreases by 30%

def getDivisionThreshold(LengthColTitle, PopColTitle, data):
    obj_count = data.at[data.shape[0] - 1, "ObjectNumber"]
    
    avg_division_lengths = []

    for obj_number in range(1, obj_count + 1):
        df_bsim = data[data["ObjectNumber"] == obj_number]
        #print(df_bsim)
        prev_length = df_bsim["Length"].iat[0]
        prev_pop = df_bsim["Population"].iat[0]

        division_lengths = []
        
        for row in df_bsim.itertuples():
            percent_change = ( row.Length - prev_length ) / prev_length
            if ( row.Population > prev_pop and percent_change <= -0.3 ):
                division_lengths.append(prev_length)
                #print("prev_length")
                #print(prev_length)

            #Update previous values
            prev_length = row.Length
            prev_pop = row.Population

        # Find the average division threshold for a bacterium
        avg_division_length = 0.0
        if ( len(division_lengths) > 0 ):
            for i in range (0, len(division_lengths)):
                avg_division_length = avg_division_length + division_lengths[i]
            avg_division_length = avg_division_length / len(division_lengths)

            avg_division_lengths.append(avg_division_length)
        else:
            avg_division_lengths.append("-")

    return avg_division_lengths

# Create a plot from a distribution with bars side-by-side
def plotDistributions( data1, data2, label1, label2, maxRate, title ) :
    bins = np.linspace(0, maxRate, maxRate)

    plt.hist([data1, data2], bins, alpha = 0.5, label=[label1, label2])
    plt.legend(loc = 'upper right')
    plt.title(title)

    # If folder doesn't exist, create new folder "Comparison_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Comparison_Plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # Save plot
    title = title + ".png"
    plt.savefig(plot_path/title)
    
    #plt.show()
    plt.close()

# Create a plot from a distribution with overlapping bars
def plotDistributionsOverlay( data1, data2, label1, label2, maxRate, title ) :
    bins = np.linspace(0, maxRate, maxRate)
    plt.style.use('seaborn-deep')
    
    plt.hist(data1, bins, alpha=0.5, label = label1, edgecolor = 'black')
    plt.hist(data2, bins, alpha=0.5, label = label2, edgecolor = 'black')
    plt.legend(loc = 'upper right')
    plt.title(title)
    
    # If folder doesn't exist, create new folder "Comparison_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Comparison_Plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # Save plot
    title = title + ".png"
    plt.savefig(plot_path/title)
    
    #plt.show()
    plt.close()

# Main Function
def run(bsim_file, cp_file, current_params, export_data, export_plots):    
    bsim_folder_name = "params_" + current_params[0] + "_" + current_params[1] + "_" + current_params[2] + "_" + current_params[3]
    
    # Get data from the csv file
    bsim_path = Path(__file__).parent.absolute()/'PhageFieldSims'/bsim_folder_name/bsim_file
    bsim_data = pandas.read_csv(bsim_path)

    cp_path = Path(__file__).parent.absolute()/'PhageFieldSims'/cp_file
    cp_data = pandas.read_csv(cp_path)
    
    print(bsim_data)
    print(cp_data)

    # Check if the dataframes are not empty
    if (not bsim_data.empty and not cp_data.empty):
        
        # Infer BSim Simulations
        avg_bsim_elongation_rates = getElongationRate("Length", "SimulationTime", bsim_data)
        avg_bsim_division_lengths = getDivisionThreshold("Length", "Population", bsim_data)

        # Infer Real Simulations
        avg_cp_elongation_rates = getElongationRate("Length", "SimulationTime", cp_data)
        avg_cp_division_lengths = getDivisionThreshold("Length", "Population", cp_data)

        # Remove zeros from plots and calculations
        non_zero_bsim_elongation = [i for i in avg_bsim_elongation_rates if i != "-"]
        non_zero_bsim_division = [i for i in avg_bsim_division_lengths if i != "-"]
        non_zero_cp_elongation = [i for i in avg_cp_elongation_rates if i != "-"]
        non_zero_cp_division = [i for i in avg_cp_division_lengths if i != "-"]

        # Finds the Wasserstein distance between two distributions
        ws_elongation = wasserstein_distance(non_zero_bsim_elongation, non_zero_cp_elongation)
        ws_division = wasserstein_distance(non_zero_bsim_division, non_zero_cp_division)

        # -------------------------------- Data to csv  ---------------------------------------
        if ( export_data ):
            obj_count = min(cp_data.at[cp_data.shape[0] - 1, "ObjectNumber"],
                                bsim_data.at[bsim_data.shape[0] - 1, "ObjectNumber"])
            #print(obj_count)
                
            cols = ["ObjectNumber", "BSimElongationRate", "CPElongationRate", "BSimDivisionThreshold", "CPDivisionThreshold", "WsElongation", "WsDivision"]
            data = []

            for obj_number in range(1, obj_count + 1):
                row = [obj_number]
                row.append(avg_bsim_elongation_rates[obj_number - 1])
                row.append(avg_cp_elongation_rates[obj_number - 1])
                row.append(avg_bsim_division_lengths[obj_number - 1])
                row.append(avg_cp_division_lengths[obj_number - 1])

                row.append(ws_elongation)
                row.append(ws_division)

                # Add row
                data.append(row)

            # Export data to csv file
            comparison_data = pandas.DataFrame(data, columns = cols)
            print(comparison_data)

            # If folder doesn't exist, create new folder "Comparison_Data" to store the csv files
            comp_path = Path(__file__).parent.absolute()/'Comparison_Data' 
            if not os.path.exists(comp_path):
                os.makedirs(comp_path)

            comp_data_name = "comparison_data" + "_" + str(current_params) + ".csv"
            comparison_data.to_csv(comp_path/comp_data_name)#'comparison_data-26hr-ver2.csv'

        # -------------------------- Plot the distributions -------------------------------------
        if ( export_plots ):
            max_rate = max(max(non_zero_bsim_elongation), max(non_zero_cp_elongation))
            el_plot_title = "Elongation_Rate" + "_" + str(current_params)
            plotDistributionsOverlay(non_zero_bsim_elongation, non_zero_cp_elongation, "BSim", "CellProfiler", math.ceil(max_rate), el_plot_title)

            max_threshold = max(max(non_zero_bsim_division), max(non_zero_cp_division))
            div_plot_title = "Division_Threshold" + "_" + str(current_params)
            plotDistributionsOverlay(non_zero_bsim_division, non_zero_cp_division, "BSim", "CellProfiler", math.ceil(max_threshold), div_plot_title)

    else:
        print("Empty dataframe. bsim_data.empty: ", bsim_data.empty)
        return -1, -1

    # Return the wasserstein distances
    return ws_elongation, ws_division

#run('Infection_Simulation-26hr-ver2.csv', 'Infection_Simulation-26hr-ver2-cp.csv', (0,0,0,1))





















