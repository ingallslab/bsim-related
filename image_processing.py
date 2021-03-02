import pandas
import numpy as np
import skimage.morphology, skimage.measure
#import skimage.io, skimage.filters, skimage.util

r"""
# Helper function: Uses the binary image of the microcolony to find the properties of the contour based envelope of the microcolony
# Functionality to add: be able to take grayscale and binary images; and generate simulation images as binary (to remove unnecessary thresholding)
def binary_image_envelope_props(binary_image):
    binary_image_close = skimage.morphology.binary_closing(binary_image)
    # change to bw
    image_close = skimage.util.img_as_ubyte(binary_image_close)
    # get contours
    contours = skimage.measure.find_contours(binary_image_close)
    for contour in contours:
        contour_poly = skimage.draw.polygon(contour[:, 0], contour[:, 1])
        image_close[contour_poly] = 255

    #thresh = skimage.filters.threshold_otsu(image)
    #binary_image = skimage.util.invert(image > thresh) # inverting to make cells white, needed that way to get proper convex envelope
    #convex_hull_image = skimage.morphology.convex_hull_image(binary_image)
    #skimage.io.imsave(fname = r"C:\Users\sohai\OneDrive\Desktop\Work\simulation_images\simulation_hull_" + str(image_number) + ".png", arr=convex_hull_image)
    image_props = pandas.DataFrame(skimage.measure.regionprops_table(image_close, properties = ["major_axis_length", "minor_axis_length", "area"]))
    return [image_props["minor_axis_length"][0], image_props["major_axis_length"][0], image_props["area"][0]]

"""

# Helper function: Uses the image of the microcolony to find the properties of the contour based envelope of the microcolony,
#                  returns False if it fails
# Precondition: image should be a 2D numpy array with datatype as np.uint8 (i.e. in 8-bit greyscale format)
#               all pixels should be either fully black (for the background) or white (for cells) image 
#               (i.e. there should be no pixels that have values other than 0 and 255)
def image_envelope_props(image):
    # perform morphological closing to get rid of any gaps
    image_close = skimage.morphology.closing(image)
    # get contours
    contours = skimage.measure.find_contours(image_close)
    # fill contours so that the image is a filled envelope of the microcolony
    for contour in contours:
        contour_poly = skimage.draw.polygon(contour[:, 0], contour[:, 1])
        image_close[contour_poly] = 255
    # get properties of the envelope
    image_props = pandas.DataFrame(skimage.measure.regionprops_table(image_close, properties = ["major_axis_length", "minor_axis_length", "area"]))
    if (image_props.shape[0] == 1):
        aspect_ratio = image_props["minor_axis_length"][0] / image_props["major_axis_length"][0]
        density_parameter = np.sum(image == 255) / image_props["area"][0]
        return aspect_ratio, density_parameter
    else:
        return False

r"""
##### add documentation
# check what happens if image is not binary
def image_convex_envelope_props(binary_image):
    # get convex hull
    convex_hull_image = skimage.morphology.convex_hull_image(binary_image)
    #skimage.io.imsave(fname = r"C:\Users\sohai\OneDrive\Desktop\Work\simulation_images\simulation_hull_" + str(image_number) + ".png", arr=convex_hull_image)
    image_props = pandas.DataFrame(skimage.measure.regionprops_table(skimage.util.img_as_ubyte(convex_hull_image), properties = ["major_axis_length", "minor_axis_length", "area"]))
    return [image_props["minor_axis_length"][0], image_props["major_axis_length"][0], image_props["area"][0]]

"""