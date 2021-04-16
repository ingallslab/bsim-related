# Infer elongation rate and division threshold parameters from csv file.

import math
import pandas
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from pathlib import Path
import os
from statistics import mean
#import time

from ..data_processing.image_drawing import draw_image_bw
from ..data_processing.image_processing import image_envelope_props
from ..data_processing.cell_data_processing import get_local_anisotropies, get_growth_info
#from image_drawing import draw_image_bw
#from image_processing import image_envelope_props
#from cell_data_processing import get_local_anisotropies

# Find the change between the lengths of non-infected bacteria at each timestep
def getElongationRate(data, time_step):
    # Get the total number of objects in the simulation
    obj_count = data["ObjectNumber"].iloc[-1]
    
    # Store the average elongation rate for each bacteria
    avg_elongation_rates = []

    # Loop through the number of objects
    for obj_number in range(1, obj_count + 1):
        elongation_rates = []  # Elongation rates from birth to division for a cell
        avg_life_rates = []    # Average elongation rate from birth to division for a cell
        df = data[data["ObjectNumber"] == obj_number]

        # Get the first length and population
        if ( df["AreaShape_MajorAxisLength"].shape[0] == 0 ): 
            print("IndexError: index 0 is out of bounds for axis 0 with size 0")
            return -1
        else:
            prev_length = df["AreaShape_MajorAxisLength"].iat[0]
            first_image = data["ImageNumber"].iloc[0]
            prev_pop = data[data["ImageNumber"] == first_image]["ObjectNumber"].iloc[-1]

        for row in df.itertuples():
            change_in_length = row.AreaShape_MajorAxisLength - prev_length
            curr_pop = data[data["ImageNumber"] == row.ImageNumber]["ObjectNumber"].iloc[-1]

            # Check for division
            if ( change_in_length < 0 and curr_pop > prev_pop ):
                if (len(elongation_rates) > 0):
                    avg_life_rates.append( sum(elongation_rates)/len(elongation_rates) )
                    elongation_rates = []

            # Elongation events should be positive; disregard changes in length during division and infection 
            elif ( change_in_length > 0 ):
                elongation_rates.append(change_in_length / time_step)

            # Update previous values
            prev_length = row.AreaShape_MajorAxisLength
            prev_pop = curr_pop

        # Find the average elongation rate for a bacterium
        if ( len(avg_life_rates) > 0 ) :
            avg_elongation_rates.append( sum(avg_life_rates)/len(avg_life_rates) )
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
            if ( pop[row.ImageNumber - 1] > prev_pop and row.AreaShape_MajorAxisLength - prev_length < 0 ):
                # Get the division length
                division_lengths.append(prev_length)

            # Update previous values
            prev_length = row.AreaShape_MajorAxisLength
            prev_pop = pop[row.ImageNumber - 1]

        # Find the average division threshold for a bacterium
        if ( len(division_lengths) > 0 ):
            avg_division_lengths.append( sum(division_lengths) / len(division_lengths) )
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
def run(bsim_file, cp_file, current_params, export_data, export_plots, bsim_export_time, cp_export_time, sim_dim):
    # Find the newest folder created
    folder_path = Path(__file__).parent.absolute()/'..'/'..'/'scripts'/'PhageFieldSims'
    folders = [os.path.join(folder_path, x) for x in os.listdir(folder_path)]
    newest = max(folders, key = os.path.getctime)
    #print("Newest modified", newest)
    
    # Get data from the csv file
    bsim_path = Path(__file__).parent.parent.parent.absolute()/'scripts'/'PhageFieldSims'/newest/bsim_file
    bsim_data = pandas.read_csv(bsim_path, index_col = False) # force pandas to not use the first column as the index

    cp_path = Path(__file__).parent.parent.parent.absolute()/cp_file
    cp_data = pandas.read_csv(cp_path, index_col = False)     # force pandas to not use the first column as the index
    
    print(bsim_data)
    print(cp_data)

    # Check if the dataframes are not empty
    if (not bsim_data.empty and not cp_data.empty):
        
        # Infer BSim Simulations
        #avg_bsim_elongation_rates = getElongationRate(bsim_data, bsim_export_time)
        #avg_bsim_division_lengths = getDivisionThreshold(bsim_data)
        avg_bsim_elongation_rates, avg_bsim_division_lengths = get_growth_info(bsim_data, bsim_export_time) ######

        # should be the same
        image_count = min(cp_data.at[cp_data.shape[0] - 1, "ImageNumber"],
                    bsim_data.at[bsim_data.shape[0] - 1, "ImageNumber"])
        ws_local_anisotropies = []
        aspect_ratio_diff = []
        density_parameter_diff = []
        for image_number in range(1, image_count + 1):
            df_bsim = bsim_data[bsim_data["ImageNumber"] == image_number]
            df_bsim.reset_index(drop = True, inplace = True)
            df_cp = cp_data[cp_data["ImageNumber"] == image_number]
            df_cp.reset_index(drop = True, inplace = True)

            cell_centers_x_bsim = df_bsim["AreaShape_Center_X"]
            cell_centers_y_bsim = df_bsim["AreaShape_Center_Y"]
            # uses major and minor axis length for length and radius
            cell_lengths_bsim = df_bsim["AreaShape_MajorAxisLength"] - df_bsim["AreaShape_MinorAxisLength"]
            cell_radii_bsim = df_bsim["AreaShape_MinorAxisLength"] / 2
            cell_orientations_bsim = df_bsim["AreaShape_Orientation"]

            anisotropies_bsim = get_local_anisotropies(cell_centers_x_bsim, cell_centers_y_bsim, cell_orientations_bsim, radius = 60) # set range as 60 for now
            # store image_dimensions as a variable
            image_bsim = np.array( draw_image_bw(sim_dim, cell_centers_x_bsim, cell_centers_y_bsim, cell_lengths_bsim, cell_radii_bsim, cell_orientations_bsim) )
            aspect_ratio_bsim, density_parameter_bsim = image_envelope_props(image_bsim)

            cell_centers_x_cp = df_cp["AreaShape_Center_X"]
            cell_centers_y_cp = df_cp["AreaShape_Center_Y"]
            # uses major and minor axis length for length and radius
            cell_lengths_cp = df_cp["AreaShape_MajorAxisLength"] - df_cp["AreaShape_MinorAxisLength"]
            cell_radii_cp = df_cp["AreaShape_MinorAxisLength"] / 2
            cell_orientations_cp = df_cp["AreaShape_Orientation"]

            anisotropies_cp = get_local_anisotropies(cell_centers_x_cp, cell_centers_y_cp, cell_orientations_cp, radius = 60) # set range as 60 for now
            # change to actual image when we have real data, store image_dimensions as a variable
            image_cp = np.array( draw_image_bw(sim_dim, cell_centers_x_cp, cell_centers_y_cp, cell_lengths_cp, cell_radii_cp, cell_orientations_cp) )
            aspect_ratio_cp, density_parameter_cp = image_envelope_props(image_cp)

            #non_zero_bsim_aniso = [i for i in anisotropies_bsim if i != "-"]
            #non_zero_cp_aniso = [i for i in anisotropies_cp if i != "-"]
            ws_local_anisotropies.append(wasserstein_distance(anisotropies_bsim, anisotropies_cp))
            aspect_ratio_diff.append(abs(aspect_ratio_bsim - aspect_ratio_cp))
            density_parameter_diff.append(abs(density_parameter_bsim - density_parameter_cp))

        '''print(ws_local_anisotropies)
        print(aspect_ratio_diff)
        print(density_parameter_diff)'''
        ws_local_anisotropies = mean(ws_local_anisotropies)
        aspect_ratio_diff = mean(aspect_ratio_diff)
        density_parameter_diff = mean(density_parameter_diff)
        
        '''
        df_bsim.reset_index(drop = True, inplace = True)
        cell_centers_x_bsim = df_bsim["AreaShape_Center_X"]
        cell_centers_y_bsim = df_bsim["AreaShape_Center_Y"]
        # uses major and minor axis length for length and radius
        cell_lengths_bsim = df_bsim["AreaShape_MajorAxisLength"] - df_bsim["AreaShape_MinorAxisLength"]
        cell_radii_bsim = df_bsim["AreaShape_MinorAxisLength"] / 2
        cell_orientations_bsim = df_bsim["AreaShape_Orientation"]
        
        anisotropies_bsim = get_local_anisotropies(cell_centers_x_bsim, cell_centers_y_bsim, cell_orientations_bsim, radius = 60) # set range as 60 for now
        # store image_dimensions as a variable
        image_bsim = np.array( draw_image_bw((1870, 2208), cell_centers_x_bsim, cell_centers_y_bsim, cell_lengths_bsim, cell_radii_bsim, cell_orientations_bsim) )
        aspect_ratio_bsim, density_parameter_bsim = image_envelope_props(image_bsim)
        '''

        # Infer Real Simulations
        #avg_cp_elongation_rates = getElongationRate(cp_data, cp_export_time)
        #avg_cp_division_lengths = getDivisionThreshold(cp_data)
        avg_cp_elongation_rates, avg_cp_division_lengths = get_growth_info(cp_data, cp_export_time)

        '''
        df_cp = cp_data[cp_data["ImageNumber"] == cp_data.at[cp_data.shape[0] - 1, "ImageNumber"]]
        df_cp.reset_index(drop = True, inplace = True)
        cell_centers_x_cp = df_cp["AreaShape_Center_X"]
        cell_centers_y_cp = df_cp["AreaShape_Center_Y"]
        # uses major and minor axis length for length and radius
        cell_lengths_cp = df_cp["AreaShape_MajorAxisLength"] - df_cp["AreaShape_MinorAxisLength"]
        cell_radii_cp = df_cp["AreaShape_MinorAxisLength"] / 2
        cell_orientations_cp = df_cp["AreaShape_Orientation"]
        anisotropies_cp = get_local_anisotropies(cell_centers_x_cp, cell_centers_y_cp, cell_orientations_cp, radius = 60) # set range as 60 for now
        # change to actual image when we have real data, store image_dimensions as a variable
        image_cp = np.array( draw_image_bw((1870, 2208), cell_centers_x_cp, cell_centers_y_cp, cell_lengths_cp, cell_radii_cp, cell_orientations_cp) )
        aspect_ratio_cp, density_parameter_cp = image_envelope_props(image_cp)
        '''

        # Check if length dataframe was valid
        if (avg_bsim_elongation_rates == -1 or avg_bsim_division_lengths == -1 or avg_cp_elongation_rates == -1 or avg_cp_division_lengths == -1):
            print("Invalid dataframe for length")
            return -1, -1, -1, -1, -1
        
        # Remove zeros from plots and calculations
        non_zero_bsim_elongation = [i for i in avg_bsim_elongation_rates if i != "-"]
        non_zero_bsim_division = [i for i in avg_bsim_division_lengths if i != "-"]
        non_zero_cp_elongation = [i for i in avg_cp_elongation_rates if i != "-"]
        non_zero_cp_division = [i for i in avg_cp_division_lengths if i != "-"]
        
        # Finds the Wasserstein distance between two distributions
        ws_elongation = wasserstein_distance(non_zero_bsim_elongation, non_zero_cp_elongation)
        ws_division = wasserstein_distance(non_zero_bsim_division, non_zero_cp_division)
        #ws_local_anisotropies = wasserstein_distance(anisotropies_bsim, anisotropies_cp)
        #aspect_ratio_diff = aspect_ratio_bsim - aspect_ratio_cp
        #density_parameter_diff = density_parameter_bsim - density_parameter_cp

        # -------------------------------- Data to csv  ---------------------------------------
        if ( export_data ):
            obj_count = min(cp_data["ObjectNumber"].iloc[-1], bsim_data["ObjectNumber"].iloc[-1])
            print("obj_count: ", obj_count)
                
            cols = ["ObjectNumber", "BSimElongationRate", "CPElongationRate", "BSimDivisionThreshold", "CPDivisionThreshold",
                    "BSimAnisotropies", "CPAnisotropies", "BSimAspectRatio", "CPAspectRatio", "BSimDensityParam", "CPDensityParam",
                    "WsElongation", "WsDivision", "WsLocalAnisotropies", "AspectRatioDiff", "DensityParamDiff"]
            data = []

            #print("len(anisotropies_bsim): ", len(anisotropies_bsim))
            for obj_number in range(0, obj_count):
                row = [obj_number + 1]
                row.append(avg_bsim_elongation_rates[obj_number])
                row.append(avg_cp_elongation_rates[obj_number])
                row.append(avg_bsim_division_lengths[obj_number])
                row.append(avg_cp_division_lengths[obj_number])

                #print("obj_number - 1: ", obj_number)
                row.append(anisotropies_bsim[obj_number])
                row.append(anisotropies_cp[obj_number])
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
