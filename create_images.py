import pandas
from PIL import Image
import matplotlib

from image_drawing import draw_image_bw, draw_image_colormap

'''
# The radius used by get_local_anisotropies to decide if a neighbour is in range
radius = 60
'''

#my_dpi= max(image_dimensions[0] / fig.get_size_inches()[0], image_dimensions[1] / fig.get_size_inches()[1])
my_dpi = 1000 # only for colormap image types
image_dimensions = (3200, 3200)
image_type = 'colormap_microdomain_orientations'
folder_fp = r"C:\Users\sohai\OneDrive\Desktop\Work\images\image_"
data_fp = 'C:\\Users\\sohai\\IdeaProjects\\bsim\\examples\\PhysModBsim\\MyExpt_EditedObjects8_simulation_2000cells.csv'

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
    print("creating image number " + str(image_number) + "...") # to remove
    # print(df_image) # to remove

    if (image_type == 'bw'):
        image = draw_image_bw(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        image.save(fp = folder_fp + str(image_number) + ".png")
    elif (image_type == 'colormap_orientations'):
        fig = draw_image_colormap(image_dimensions, image_type, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        fig.savefig(folder_fp + str(image_number) + ".png", dpi=my_dpi, bbox_inches='tight')
        colormapping_parameters = cell_orientations
    elif (image_type == 'colormap_local_anisotropies'):
        fig = draw_image_colormap(image_dimensions, image_type, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        fig.savefig(folder_fp + str(image_number) + ".png", dpi=my_dpi, bbox_inches='tight')
    elif(image_type == 'colormap_microdomains'):
        fig = draw_image_colormap(image_dimensions, image_type, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        fig.savefig(folder_fp + str(image_number) + ".png", dpi=my_dpi, bbox_inches='tight')
    elif (image_type == 'colormap_microdomain_orientations'):
        '''
        fig = draw_image_colormap(image_dimensions, image_type, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        my_dpi= max(image_dimensions[0] / fig.get_size_inches()[0], image_dimensions[1] / fig.get_size_inches()[1])
        fig.savefig(folder_fp + str(image_number) + ".png", dpi=500)
        '''
        fig = draw_image_colormap(image_dimensions, image_type, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        fig.savefig(folder_fp + str(image_number) + ".png", dpi=my_dpi, bbox_inches='tight')
    else:
        print('invalid image type')