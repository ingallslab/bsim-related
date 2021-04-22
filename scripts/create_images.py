import pandas
from PIL import Image
import matplotlib
from pathlib import Path
import math

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bsim_related.data_processing.image_drawing import draw_image_bw, draw_image_colormap
#from image_drawing import draw_image_bw, draw_image_colormap
from bsim_related.data_processing.cell_data_processing import get_patch_numbers, get_patch_orientations, get_local_anisotropies
#from cell_data_processing import get_patch_numbers, get_patch_orientations, get_local_anisotropies

############################# arguments required to run script ####################################
# required for all image types
image_dimensions = (1870, 2208)
image_type = 'bw' # there are six types: 'bw', 'alpha_composite', 'colormap_orientations', 'colormap_local_anisotropies', 'colormap_microdomains', and 'colormap_microdomain_orientations'
ouput_image_stem = str(Path(__file__).parent.parent.absolute()/'data'/'image_')
ouput_image_extension = '.png'
data_fp = str(Path(__file__).parent.parent.absolute()/'data'/'MyExpt_filtered_objects_2.csv')
# required only for alpha_composite
base_image_stem = r"C:\Users\sohai\OneDrive\Desktop\images\image_" # the images should be numbered in three digits
base_image_extension = '.tif' 
# required for all colormap image types
my_dpi = 1000 #max(image_dimensions[0] / fig.get_size_inches()[0], image_dimensions[1] / fig.get_size_inches()[1])
# required only for colormap_local_anisotropies
neighbourhood_range = 60

#####################################################################################################################

# remove unnnecessary columns to make more efficient?
dataframe = pandas.read_csv(data_fp)
image_count = dataframe.at[dataframe.shape[0] - 1, "ImageNumber"]

for image_number in range(1, image_count + 1):
    # get data only for this image
    df_image = dataframe[dataframe["ImageNumber"] == image_number]
    df_image.reset_index(drop = True, inplace = True)
    cell_centers_x = df_image["AreaShape_Center_X"]
    cell_centers_y = df_image["AreaShape_Center_Y"]
    # uses major and minor axis length for length and radius
    cell_lengths = df_image["AreaShape_MajorAxisLength"] - df_image["AreaShape_MinorAxisLength"]
    cell_radii = df_image["AreaShape_MinorAxisLength"] / 2
    cell_orientations = df_image["AreaShape_Orientation"]
    print("creating image number " + str(image_number) + "...")
    # print(df_image) # to remove

    if (image_type == 'bw'):
        image = draw_image_bw(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        image.save(fp = ouput_image_stem + f'{image_number:03d}' + ouput_image_extension)
    elif (image_type == 'alpha_composite'):
        base_image_filepath = base_image_stem + f'{image_number:03d}' + base_image_extension
        image = draw_image_bw(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations, base_image_filepath)
        image.save(fp = ouput_image_stem + f'{image_number:03d}' + ouput_image_extension)
    elif (image_type == 'colormap_orientations'):
        cmap = matplotlib.cm.get_cmap('hsv') # cyclic colormap
        norm = matplotlib.colors.Normalize(vmin= -math.pi / 2, vmax= math.pi / 2)
        colormapping_parameters = cell_orientations
        colorbar_label = 'orientations'
        fig = draw_image_colormap(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations,
                                    colormapping_parameters, cmap, norm, colorbar_label)
        fig.savefig(ouput_image_stem + f'{image_number:03d}' + ouput_image_extension, dpi=my_dpi, bbox_inches='tight')
    elif (image_type == 'colormap_local_anisotropies'):
        cmap = matplotlib.cm.get_cmap('jet') # sequential colormap
        norm = matplotlib.colors.Normalize(vmin= 0.5, vmax= 1)
        # Get local anisotropy in orientations around each cell for colormapping
        colormapping_parameters = get_local_anisotropies(cell_centers_x, cell_centers_y, cell_orientations, neighbourhood_range)
        colorbar_label = 'local anisotropies'
        fig = draw_image_colormap(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations,
                                    colormapping_parameters, cmap, norm, colorbar_label)
        fig.savefig(ouput_image_stem + f'{image_number:03d}' + ouput_image_extension, dpi=my_dpi, bbox_inches='tight')
    elif(image_type == 'colormap_microdomains'):
        cmap = matplotlib.cm.get_cmap('jet')
        colormapping_parameters, patch_count = get_patch_numbers(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        norm = matplotlib.colors.Normalize(vmin= 1, vmax= patch_count)
        colorbar_label = 'microdomains'
        fig = draw_image_colormap(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations,
                                    colormapping_parameters, cmap, norm, colorbar_label)
        fig.savefig(ouput_image_stem + f'{image_number:03d}' + ouput_image_extension, dpi=my_dpi, bbox_inches='tight')
    elif (image_type == 'colormap_microdomain_orientations'):
        cmap = matplotlib.cm.get_cmap('hsv') # cyclic colormap
        norm = matplotlib.colors.Normalize(vmin= -math.pi / 2, vmax= math.pi / 2)
        colormapping_parameters = get_patch_orientations(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        colorbar_label = 'microdomain orientations'
        '''
        fig = draw_image_colormap(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        my_dpi= max(image_dimensions[0] / fig.get_size_inches()[0], image_dimensions[1] / fig.get_size_inches()[1])
        fig.savefig(ouput_image_stem + f'{image_number:03d}' + ouput_image_extension, dpi=500)
        '''
        fig = draw_image_colormap(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations,
                                    colormapping_parameters, cmap, norm, colorbar_label)
        fig.savefig(ouput_image_stem + f'{image_number:03d}' + ouput_image_extension, dpi=my_dpi, bbox_inches='tight')
    else:
        print('invalid image type')