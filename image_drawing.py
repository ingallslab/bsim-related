#import pandas
import math
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt

from cell_data_processing import get_patch_numbers, get_patch_orientations, get_local_anisotropies

def draw_cell(draw, center_x, center_y, length, radius, orientation, fill, outline):
    # angle conversion needed so that 
    axis_orientation = -(orientation + (math.pi / 2)) # clockwise angle from 3 o clock of axis of cell
    perpendicular_orientation = -orientation # clockwise angle from 3 o clock of perpendicular to axis of cell

    half_length_along_axis_x = length / 2 * math.cos(axis_orientation)
    half_length_along_axis_y = length / 2 * math.sin(axis_orientation)
    radius_perpendicular_to_axis_x = radius * math.cos(perpendicular_orientation)
    radius_perpendicular_to_axis_y = radius * math.sin(perpendicular_orientation)

    p1 = (center_x + half_length_along_axis_x + radius_perpendicular_to_axis_x, center_y + half_length_along_axis_y + radius_perpendicular_to_axis_y)
    p2 = (center_x + half_length_along_axis_x - radius_perpendicular_to_axis_x, center_y + half_length_along_axis_y - radius_perpendicular_to_axis_y)
    p3 = (center_x - half_length_along_axis_x - radius_perpendicular_to_axis_x, center_y - half_length_along_axis_y - radius_perpendicular_to_axis_y)
    p4 = (center_x - half_length_along_axis_x + radius_perpendicular_to_axis_x, center_y - half_length_along_axis_y + radius_perpendicular_to_axis_y)
    draw.polygon([p1, p2, p3, p4], fill=fill, outline=outline)

    p5 = (center_x + half_length_along_axis_x - radius, center_y + half_length_along_axis_y - radius)
    p6 = (center_x + half_length_along_axis_x + radius, center_y + half_length_along_axis_y + radius)
    end_1 = perpendicular_orientation * 180 / math.pi
    start_1 = end_1 - 180
    draw.pieslice([p5, p6], start=start_1, end=end_1, fill=fill, outline=outline)

    p7 = (center_x - half_length_along_axis_x - radius, center_y - half_length_along_axis_y - radius)
    p8 = (center_x - half_length_along_axis_x + radius, center_y - half_length_along_axis_y + radius)
    draw.pieslice([p7, p8], start=end_1, end=start_1, fill=fill, outline=outline)

# draw a black and white image with the given dimensions and cell data
def draw_image_bw(image_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    #image = Image.new(mode = 'RGB', size = image_dimensions, color = (0, 0, 0))
    #image = Image.new(mode = '1', size = image_dimensions, color = 0) # boolean image
    image = Image.new(mode = 'L', size = image_dimensions, color = 0)
    draw = ImageDraw.Draw(image)

    cell_count = len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    # draw cells
    for i in range(cell_count):
        draw_cell(draw, cell_centers_x[i], cell_centers_y[i], cell_lengths[i], cell_radii[i], cell_orientations[i], 255, None)

    return image
    #image.save(fp = r"C:\Users\sohai\OneDrive\Desktop\Work\simulation_images\simulation_image_" + str(image_number) + ".png")

# draw a black and white image with the given dimensions and cell data
def draw_image_alpha_comp(image_dimensions, cell_profiler_image_filepath, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    image = Image.new(mode = 'RGBA', size = image_dimensions, color = (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    cell_count = len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    # draw cells
    for i in range(cell_count):
        draw_cell(draw, cell_centers_x[i], cell_centers_y[i], cell_lengths[i], cell_radii[i], cell_orientations[i], fill=(0, 0, 0, 30), outline=(0, 0, 0, 100))

    base = Image.open(fp = cell_profiler_image_filepath).convert('RGBA')
    out = Image.alpha_composite(base, image)
    return out

# The radius used by get_local_anisotropies to decide if a neighbour is in range
radius = 60

# draw a black and white image with the given dimensions and cell data
def draw_image_colormap(image_dimensions, colormap_type, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    image = Image.new(mode = 'RGBA', size = image_dimensions, color = (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
        
    # Make a colormap (for colored images)
    # colormapping_parameters could be orientations, local anisotropies, 
    # microdomains to which the cells belongs to, or orientations of the microdomains to which the cells belongs to
    # colormapping_parameters will be a panda series (in the case of 'colormap_orientations') or a list
    if (colormap_type == 'colormap_orientations'):
        cmap = matplotlib.cm.get_cmap('hsv') # cyclic colormap
        norm = matplotlib.colors.Normalize(vmin= -math.pi / 2, vmax= math.pi / 2)
        colormapping_parameters = cell_orientations
        colorbar_label = 'orientations'
    if (colormap_type == 'colormap_local_anisotropies'):
        cmap = matplotlib.cm.get_cmap('jet') # sequential colormap
        norm = matplotlib.colors.Normalize(vmin= 0.5, vmax= 1)
        # Get local anisotropy in orientations around each cell for colormapping
        colormapping_parameters = get_local_anisotropies(cell_centers_x, cell_centers_y, cell_orientations, radius)
        colorbar_label = 'local anisotropies'
    elif(colormap_type == 'colormap_microdomains'):
        cmap = matplotlib.cm.get_cmap('jet')
        colormapping_parameters, patch_count = get_patch_numbers(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        norm = matplotlib.colors.Normalize(vmin= 1, vmax= patch_count)
        colorbar_label = 'microdomains'
    elif (colormap_type == 'colormap_microdomain_orientations'):
        cmap = matplotlib.cm.get_cmap('hsv') # cyclic colormap
        norm = matplotlib.colors.Normalize(vmin= -math.pi / 2, vmax= math.pi / 2)
        colormapping_parameters = get_patch_orientations(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
        colorbar_label = 'microdomain orientations'

    cell_count = len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    # draw cells
    for i in range(cell_count):
        color = cmap(norm(colormapping_parameters[i]), bytes=True)
        draw_cell(draw, cell_centers_x[i], cell_centers_y[i], cell_lengths[i], cell_radii[i], cell_orientations[i], fill=color, outline=(64, 64, 64, 255))
    
    # Create figure and axes using matplotlib
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    #########ax.imshow(image, aspect='equal')
    # Add a Colorbar
    #matplotlib.colorbar.ColorbarBase(ax=ax, cmap=plt.get_cmap('hsv'), orientation="vertical")
    #fig.colorbar(im, cax=cax, orientation='vertical')
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=colorbar_label)
    ########plt.tight_layout(pad=1)
    # Save the plot
    #fig.savefig(r"C:\Users\sohai\OneDrive\Desktop\Work\color_map_images_microdomain_ors\color_map_plot_" + str(image_number) + ".png", bbox_inches='tight')

    return fig #, image # return axis?