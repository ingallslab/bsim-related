import pandas
import numpy as np
from statistics import mean
import skimage.io, skimage.filters, skimage.util

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bsim_related.data_processing.image_drawing import draw_image_bw
from bsim_related.data_processing.image_processing import image_envelope_props
from bsim_related.data_processing.cell_data_processing import get_local_anisotropies
#from image_drawing import draw_image_bw
#from image_processing import image_envelope_props
#from cell_data_processing import get_local_anisotropies

# The radius used by get_local_anisotropies to decide if a neighbour is in range
radius = 60

image_dimensions = (700, 250)
comparison_data_fp = 'C:\\Users\\sohai\\IdeaProjects\\bsim\\examples\\PhysModBsim\\comparison_data.csv'
bsim_data_fp = 'C:\\Users\\sohai\\IdeaProjects\\bsim\\examples\\PhysModBsim\\MyExpt_EditedObjects8_simulation.csv'
cell_profiler_data_fp = 'C:\\Users\\sohai\\IdeaProjects\\bsim\\examples\\PhysModBsim\\MyExpt_EditedObjects8.csv'
image_cp_fp_stem = r"C:\Users\sohai\Downloads\untitled_folder_6\OrigBlue0"
image_extension = ".tiff"

#####################################################################################################################

# ignore, to remove
"""
with open('file1.csv') as cell_profiler_file, open('file2.csv') as simulation_file:
    cell_profiler_file_reader = csv.reader(cell_profiler_file)
    simulation_file_reader = csv.reader(simulation_file)
"""

# remove unnnecessary columns to make more efficient?
cell_profiler_data = pandas.read_csv(cell_profiler_data_fp)
bsim_data = pandas.read_csv(bsim_data_fp)
print(bsim_data)  # to remove
print(bsim_data.shape[0] - 1)  # to remove
print(cell_profiler_data)  # to remove
print(cell_profiler_data.shape[0] - 1)  # to remove
                
image_count = min(cell_profiler_data.at[cell_profiler_data.shape[0] - 1, "ImageNumber"],
                    bsim_data.at[bsim_data.shape[0] - 1, "ImageNumber"])
print(image_count) # to remove

cols = ["ImageNumber", "CellProfiler_Population", "Bsim_Population", "CellProfiler_MeanLength", "Bsim_MeanLength",
        "CellProfiler_VarianceLength", "Bsim_VarianceLength", "CellProfiler_MeanOrientation", "Bsim_MeanOrientation",
        "CellProfiler_VarianceOrientation", "Bsim_VarianceOrientation", "CellProfiler_AspectRatio", "Bsim_AspectRatio",
        "CellProfiler_DensityParameter", "Bsim_DensityParameter"]
data = []
for image_number in range(1, image_count + 1):
    df_bsim = bsim_data[bsim_data["ImageNumber"] == image_number]
    df_bsim.reset_index(drop = True, inplace = True)
    df_cp = cell_profiler_data[cell_profiler_data["ImageNumber"] == image_number]
    df_cp.reset_index(drop = True, inplace = True)
    print(df_bsim) # to remove
    print(df_cp) # to remove
    #############
    cell_count_bsim = df_bsim.shape[0]
    cell_centers_x_bsim = df_bsim["AreaShape_Center_X"]
    cell_centers_y_bsim = df_bsim["AreaShape_Center_Y"]
    # uses major and minor axis length for length and radius
    cell_lengths_bsim = df_bsim["AreaShape_MajorAxisLength"] - df_bsim["AreaShape_MinorAxisLength"]
    cell_radii_bsim = df_bsim["AreaShape_MinorAxisLength"] / 2
    cell_orientations_bsim = df_bsim["AreaShape_Orientation"]
    #############
    cell_count_cp = df_cp.shape[0]
    cell_centers_x_cp = df_cp["AreaShape_Center_X"]
    cell_centers_y_cp = df_cp["AreaShape_Center_Y"]
    # uses major and minor axis length for length and radius
    cell_lengths_cp = df_cp["AreaShape_MajorAxisLength"] - df_bsim["AreaShape_MinorAxisLength"]
    cell_radii_cp = df_cp["AreaShape_MinorAxisLength"] / 2
    cell_orientations_cp = df_cp["AreaShape_Orientation"]

    # make a row with all the data for this image_number
    row = [image_number]
    row.append(cell_count_cp)
    row.append(cell_count_bsim)
    row.append(cell_lengths_cp.mean())
    row.append(cell_lengths_bsim.mean())
    row.append(cell_lengths_cp.var(ddof = 0))
    row.append(cell_lengths_bsim.var(ddof = 0))
    row.append(cell_orientations_cp.mean())
    row.append(cell_orientations_bsim.mean())
    row.append(cell_orientations_cp.var(ddof = 0))
    row.append(cell_orientations_bsim.var(ddof = 0))

    # get image_cp
    image_cp = skimage.io.imread(fname = image_cp_fp_stem + str(image_number) + image_extension, as_gray = True)
    thresh = skimage.filters.threshold_otsu(image_cp)
    binary_image_cp = skimage.util.invert(image_cp > thresh) # inverting to make cells white, needed that way to pass into helper function
    image_cp = skimage.util.img_as_ubyte(binary_image_cp)
    aspect_ratio_cp, density_parameter_cp = image_envelope_props(binary_image_cp)
    # get image_bsim
    image_bsim = np.array( draw_image_bw(image_dimensions, cell_centers_x_bsim, cell_centers_y_bsim, cell_lengths_bsim, cell_radii_bsim, cell_orientations_bsim) )
    aspect_ratio_bsim, density_parameter_bsim = image_envelope_props(image_bsim)
    # aspect ratios
    row.append(aspect_ratio_cp)
    row.append(aspect_ratio_bsim)
    # density parameters
    row.append(density_parameter_cp)
    row.append(density_parameter_bsim)
    # division angle mean and variance for bsim
    row.append(df_bsim["Division_Angle"].mean())
    row.append(df_bsim["Division_Angle"].var(ddof = 0))
    # local order parameters: the local order parameter for each image is the average of the local anisotropies for that image
    # test to do: local order param. should be 1 for 1 cell
    anisotropies_bsim = get_local_anisotropies(cell_centers_x_bsim, cell_centers_y_bsim, cell_orientations_bsim, radius)
    row.append(mean(anisotropies_bsim))
    anisotropies_cp = get_local_anisotropies(cell_centers_x_cp, cell_centers_y_cp, cell_orientations_cp, radius)
    row.append(mean(anisotropies_cp))
    
    # add row
    data.append(row)
    #print(row) # to remove

comparison_data = pandas.DataFrame(data, columns = cols)
print(comparison_data) # to remove
comparison_data.to_csv(comparison_data_fp)