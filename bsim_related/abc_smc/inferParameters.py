# Infer elongation rate and division threshold parameters from csv file.

import math
import pandas
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from pathlib import Path
import os

from ..data_processing.image_drawing import draw_image_bw
from ..data_processing.image_processing import image_envelope_props
from ..data_processing.cell_data_processing import get_local_anisotropies
#from image_drawing import draw_image_bw
#from image_processing import image_envelope_props
#from cell_data_processing import get_local_anisotropies


# Find the change between the lengths of non-infected bacteria at each timestep
def getElongationRate(data, time_step):
    # Get the total number of objects in the simulation
    obj_count = data.at[data.shape[0] - 1, "ObjectNumber"]
    
    # Store the average elongation rate for each bacteria
    avg_elongation_rates = []

    # Loop through the number of objects
    for obj_number in range(1, obj_count + 1):
        elongation_rates = []
        
        df = data[data["ObjectNumber"] == obj_number]

        if ( df["AreaShape_MajorAxisLength"].shape[0] == 0 ): 
            print("df_bsim[AreaShape_MajorAxisLength].shape: ", df["AreaShape_MajorAxisLength"].shape)
            print("IndexError: index 0 is out of bounds for axis 0 with size 0")
            return -1
        else:
            prev_length = df["AreaShape_MajorAxisLength"].iat[0]

        for row in df.itertuples():

            # To account only for elongation events ( should be positive; disregard changes in length during division and infection )
            change_in_length = row.AreaShape_MajorAxisLength - prev_length
            if ( change_in_length > 0 ):
                elongation_rates.append(change_in_length / time_step)

            # Update previous values
            prev_length = row.AreaShape_MajorAxisLength

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


# Find the length of the bacteria before the population increases and the length of the bacteria decreases by 5%
def getDivisionThreshold(data):
    obj_count = data.at[data.shape[0] - 1, "ObjectNumber"]
    
    # Get the population for each image
    n_images = data.at[data.shape[0] - 1, "ImageNumber"]
    pop = []

    for image_num in range(1, n_images + 1):
        df_image = data[data["ImageNumber"] == image_num]
        #print("df_image[].shape[0]: ", df_image.shape[0])
        if (df_image.shape[0] == 0):
            return -1
        else:
            pop.append(df_image["ObjectNumber"].iloc[-1])
    print("pop: ", pop)
    
    avg_division_lengths = []
    for obj_number in range(1, obj_count + 1):
        df = data[data["ObjectNumber"] == obj_number]

        if (df["AreaShape_MajorAxisLength"].shape[0] == 0):
            print("IndexError: index 0 is out of bounds for axis 0 with size 0")
            return -1
        else:
            prev_length = df["AreaShape_MajorAxisLength"].iat[0]
            prev_pop = pop[0]

        division_lengths = []
        
        for row in df.itertuples():
            percent_change = ( row.AreaShape_MajorAxisLength - prev_length ) / prev_length
            if ( pop[row.ImageNumber - 1] > prev_pop and percent_change <= -0.15 ):
                # Get the division length
                division_lengths.append(prev_length)
                #print("prev_length: ", prev_length)

            # Update previous values
            prev_length = row.AreaShape_MajorAxisLength
            prev_pop = pop[row.ImageNumber - 1]

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

# Create a plot from a distribution with overlapping bars
def plotDistributionsOverlay( data1, data2, label1, label2, objNum, title, plot_folder) :
    bins = objNum#math.ceil(np.sqrt(objNum))
    
    plt.style.use('seaborn-deep')
    plt.hist(data1, bins, alpha=0.5, label = label1, edgecolor = 'black')
    plt.hist(data2, bins, alpha=0.5, label = label2, edgecolor = 'black')
    plt.legend(loc = 'upper right')
    plt.title(title)
    
    # If folder doesn't exist, create new folder "Comparison_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Comparison_Plots'/plot_folder
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # Save plot
    title = title + ".png"
    plt.savefig(plot_path/title)
    
    #plt.show()
    plt.close()

# Plot the differences
def plotDifferences( data1, data2, label1, label2, objNum, title, plot_folder ):
    n = 1
    ind = np.arange(n)
    
    plt.bar(ind, data1 , 0.3, label = label1)
    plt.bar(ind + 0.3, data2, 0.3, label = label2)
    plt.legend(loc = 'upper right')
    plt.title(title)
    
    # If folder doesn't exist, create new folder "Comparison_Plots" to store the png files
    plot_path = Path(__file__).parent.absolute()/'Comparison_Plots'/plot_folder
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
    #bsim_path = Path(__file__).parent.absolute()/'PhageFieldSims'/bsim_file   #testing#
    bsim_data = pandas.read_csv(bsim_path, index_col = False) # force pandas to not use the first column as the index

    cp_path = Path(__file__).parent.absolute()/'PhageFieldSims'/cp_file
    cp_data = pandas.read_csv(cp_path, index_col = False)     # force pandas to not use the first column as the index
    
    print(bsim_data)
    print(cp_data)

    # Check if the dataframes are not empty
    if (not bsim_data.empty and not cp_data.empty):
        
        # Infer BSim Simulations
        bsim_time_step = 0.5#0.03#0.5    # in hours
        avg_bsim_elongation_rates = getElongationRate(bsim_data, bsim_time_step)
        avg_bsim_division_lengths = getDivisionThreshold(bsim_data)

        df_bsim = bsim_data[bsim_data["ImageNumber"] == bsim_data.at[bsim_data.shape[0] - 1, "ImageNumber"]]
        df_bsim.reset_index(drop = True, inplace = True)
        cell_centers_x_bsim = df_bsim["AreaShape_Center_X"]
        cell_centers_y_bsim = df_bsim["AreaShape_Center_Y"]
        # uses major and minor axis length for length and radius
        cell_lengths_bsim = df_bsim["AreaShape_MajorAxisLength"] - df_bsim["AreaShape_MinorAxisLength"]
        cell_radii_bsim = df_bsim["AreaShape_MinorAxisLength"] / 2
        cell_orientations_bsim = df_bsim["AreaShape_Orientation"]
        
        anisotropies_bsim = get_local_anisotropies(cell_centers_x_bsim, cell_centers_y_bsim, cell_orientations_bsim, radius = 60) # set range as 60 for now
        # store image_dimensions as a variable
        image_bsim = np.array( draw_image_bw((1000, 1000), cell_centers_x_bsim, cell_centers_y_bsim, cell_lengths_bsim, cell_radii_bsim, cell_orientations_bsim) )
        aspect_ratio_bsim, density_parameter_bsim = image_envelope_props(image_bsim)

        # Infer Real Simulations
        cp_time_step = 0.5#2/60#0.5     # in hours
        avg_cp_elongation_rates = getElongationRate(cp_data, cp_time_step)
        avg_cp_division_lengths = getDivisionThreshold(cp_data)

        df_cp = cp_data[cp_data["ImageNumber"] == cp_data.at[cp_data.shape[0] - 1, "ImageNumber"]]
        df_cp.reset_index(drop = True, inplace = True)
        cell_centers_x_cp = df_cp["AreaShape_Center_X"]
        cell_centers_y_cp = df_cp["AreaShape_Center_Y"]
        # uses major and minor axis length for length and radius
        cell_lengths_cp = df_cp["AreaShape_MajorAxisLength"] - df_bsim["AreaShape_MinorAxisLength"]
        cell_radii_cp = df_cp["AreaShape_MinorAxisLength"] / 2
        cell_orientations_cp = df_cp["AreaShape_Orientation"]

        anisotropies_cp = get_local_anisotropies(cell_centers_x_cp, cell_centers_y_cp, cell_orientations_cp, radius = 60) # set range as 60 for now
        # change to actual image when we have real data, store image_dimensions as a variable
        image_cp = np.array( draw_image_bw((1000, 1000), cell_centers_x_cp, cell_centers_y_cp, cell_lengths_cp, cell_radii_cp, cell_orientations_cp) )
        aspect_ratio_cp, density_parameter_cp = image_envelope_props(image_cp)

        # Check if length dataframe was valid
        if (avg_bsim_elongation_rates == -1 or avg_bsim_division_lengths == -1 or avg_cp_elongation_rates == -1 or avg_cp_division_lengths == -1
            or anisotropies_bsim == -1 or anisotropies_cp == -1):
            print("Invalid dataframe for length")
            return -1, -1, -1, -1, -1

        # Remove zeros from plots and calculations
        non_zero_bsim_elongation = [i for i in avg_bsim_elongation_rates if i != "-"]
        non_zero_bsim_division = [i for i in avg_bsim_division_lengths if i != "-"]
        non_zero_bsim_aniso = [i for i in anisotropies_bsim if i != "-"]
        non_zero_cp_elongation = [i for i in avg_cp_elongation_rates if i != "-"]
        non_zero_cp_division = [i for i in avg_cp_division_lengths if i != "-"]
        non_zero_cp_aniso = [i for i in anisotropies_cp if i != "-"]
        

        # Finds the Wasserstein distance between two distributions
        ws_elongation = wasserstein_distance(non_zero_bsim_elongation, non_zero_cp_elongation)
        ws_division = wasserstein_distance(non_zero_bsim_division, non_zero_cp_division)
        ws_local_anisotropies = wasserstein_distance(non_zero_bsim_aniso, non_zero_cp_aniso)
        aspect_ratio_diff = aspect_ratio_bsim - aspect_ratio_cp
        density_parameter_diff = density_parameter_bsim - density_parameter_cp

        # -------------------------------- Data to csv  ---------------------------------------
        if ( export_data ):
            obj_count = min(cp_data.at[cp_data.shape[0] - 1, "ObjectNumber"],
                                bsim_data.at[bsim_data.shape[0] - 1, "ObjectNumber"])
            #print(obj_count)
                
            cols = ["ObjectNumber", "BSimElongationRate", "CPElongationRate", "BSimDivisionThreshold", "CPDivisionThreshold",
                    "BSimAnisotropies", "CPAnisotropies", "BSimAspectRatio", "CPAspectRatio", "BSimDensityParam", "CPDensityParam",
                    "WsElongation", "WsDivision", "WsLocalAnisotropies", "AspectRatioDiff", "DensityParamDiff"]
            data = []

            #print("len(anisotropies_bsim): ", len(anisotropies_bsim))
            #print("obj_count + 1: ", obj_count + 1)
            for obj_number in range(1, obj_count + 1):
                row = [obj_number]
                row.append(avg_bsim_elongation_rates[obj_number - 1])
                row.append(avg_cp_elongation_rates[obj_number - 1])
                row.append(avg_bsim_division_lengths[obj_number - 1])
                row.append(avg_cp_division_lengths[obj_number - 1])

                #print("obj_number - 1: ", obj_number - 1)
                row.append(anisotropies_bsim[obj_number - 1])
                row.append(anisotropies_cp[obj_number - 1])
                row.append(aspect_ratio_bsim)
                row.append(aspect_ratio_cp)
                row.append(density_parameter_bsim)
                row.append(density_parameter_cp)

                row.append(ws_elongation)
                row.append(ws_division)
                row.append(ws_local_anisotropies)

                row.append(aspect_ratio_diff)
                row.append(density_parameter_diff)

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
            obj_count = min(cp_data["ObjectNumber"].iat[-1],bsim_data["ObjectNumber"].iat[-1])
            #print("obj_count: ", obj_count)

            plot_folder = "Plots_" + str(current_params)
            
            el_plot_title = "Elongation_Rate" 
            plotDistributionsOverlay(non_zero_bsim_elongation, non_zero_cp_elongation, "BSim", "CellProfiler", obj_count, el_plot_title, plot_folder)

            div_plot_title = "Division_Threshold" 
            plotDistributionsOverlay(non_zero_bsim_division, non_zero_cp_division, "BSim", "CellProfiler", obj_count, div_plot_title, plot_folder)

            ani_plot_title = "Local_Anisotropies" 
            plotDistributionsOverlay(anisotropies_bsim, anisotropies_cp, "BSim", "CellProfiler", obj_count, ani_plot_title, plot_folder)

            aspect_plot_title = "Aspect_Ratio" 
            plotDifferences( aspect_ratio_bsim, aspect_ratio_cp, "BSim", "CellProfiler", obj_count, aspect_plot_title, plot_folder)

            density_plot_title = "Density" 
            plotDifferences( density_parameter_bsim, density_parameter_cp, "BSim", "CellProfiler", obj_count, density_plot_title, plot_folder)

    else:
        print("Empty dataframe. bsim_data.empty: ", bsim_data.empty)
        return -1, -1, -1, -1, -1

    # Return the wasserstein distances
    return ws_elongation, ws_division, ws_local_anisotropies, aspect_ratio_diff, density_parameter_diff

#run('BSim_Simulation.csv', 'BSim_Simulation_1.23_0.277_7.0_0.1-6.5.csv', 'real_data_01', True, True)
#run('BSim_Simulation-0.03-1.csv', 'MyExpt_filtered_objects_2.csv', 'real_data_01', True, True)





















