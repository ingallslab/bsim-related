#import pandas
import numpy as np
import math
import scipy.linalg as la

from cell_contact import determine_contact

# in radians; make this a function parameter
orientation_threshold = 0.07

# ; make this a function parameter
contact_range = 1.10

'''
def orientation_percentage_difference(orientation_1, orientation_2):
    orientation_difference = abs(orientation_1 - orientation_2)
    orientation_avg = (orientation_1 + orientation_2) / 2
    if (orientation_difference > math.pi / 2):
        orientation_difference = math.pi - orientation_difference
        if (orientation_avg >= 0):
            orientation_avg -= math.pi / 2
        else:
            orientation_avg += math.pi / 2
    
    return abs(orientation_difference / orientation_avg * 100)

# returns a numpy array
def get_patch_numbers(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    cell_count = len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    indices_of_cells_not_in_patch = list(range(cell_count))
    indices_of_cells_in_patches = []
    patch_count = 0

    patch_numbers_of_cells = [0] * cell_count

    # while indices_of_cells_not_in_patch is not empty
    while(indices_of_cells_not_in_patch):
        patch_count += 1
        indices_of_cells_in_patches.append([indices_of_cells_not_in_patch[0]])
        indices_of_cells_not_in_patch = indices_of_cells_not_in_patch[1:]
        for i in indices_of_cells_in_patches[patch_count - 1]:
            new_indices_of_cells_not_in_patch = []
            b_o = cell_orientations[i]

            for j in indices_of_cells_not_in_patch:
                n_o = cell_orientations[j]
                orientation_difference = abs(b_o - n_o)
                orientation_difference = min(orientation_difference, math.pi - orientation_difference)
                in_contact = determine_contact([cell_centers_x[i], cell_centers_y[i]], cell_lengths[i], cell_radii[i], b_o, 
                            [cell_centers_x[j], cell_centers_y[j]], cell_lengths[j], cell_radii[j], n_o, contact_range)

                if (orientation_difference < orientation_threshold and in_contact):
                    indices_of_cells_in_patches[patch_count - 1].append(j)
                    patch_numbers_of_cells[j] = patch_count
                    #print(orientation_difference)
                    #patch_numbers_of_cells.append(patch_numbers_of_cells[j])
                    #patch_orientations[patch_numbers_of_cells[j] - 1].append(b_o)
                    #break
                else:
                    new_indices_of_cells_not_in_patch.append(j)

            indices_of_cells_not_in_patch = new_indices_of_cells_not_in_patch

    return patch_numbers_of_cells, patch_count

def get_patch_orientations(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    cell_count = len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    indices_of_cells_not_in_patch = list(range(cell_count))
    indices_of_cells_in_patches = []
    patch_count = 0

    patch_numbers_of_cells = [0] * cell_count
    patch_orientations = []

    # while indices_of_cells_not_in_patch is not empty
    while(indices_of_cells_not_in_patch):
        patch_count += 1
        indices_of_cells_in_patches.append([indices_of_cells_not_in_patch[0]])
        orientations_in_patch = [cell_orientations[indices_of_cells_not_in_patch[0]]]
        indices_of_cells_not_in_patch = indices_of_cells_not_in_patch[1:]
        for i in indices_of_cells_in_patches[patch_count - 1]:
            new_indices_of_cells_not_in_patch = []
            b_o = cell_orientations[i]

            for j in indices_of_cells_not_in_patch:
                n_o = cell_orientations[j]
                orientation_difference = abs(b_o - n_o)
                orientation_difference = min(orientation_difference, math.pi - orientation_difference)
                in_contact = determine_contact([cell_centers_x[i], cell_centers_y[i]], cell_lengths[i], cell_radii[i], b_o, 
                            [cell_centers_x[j], cell_centers_y[j]], cell_lengths[j], cell_radii[j], n_o, contact_range)

                if (orientation_difference < orientation_threshold and in_contact):
                    indices_of_cells_in_patches[patch_count - 1].append(j)
                    patch_numbers_of_cells[j] = patch_count
                    orientations_in_patch.append(cell_orientations[j])
                    #print(orientation_difference)
                    #patch_numbers_of_cells.append(patch_numbers_of_cells[j])
                    #patch_orientations[patch_numbers_of_cells[j] - 1].append(b_o)
                    #break
                else:
                    new_indices_of_cells_not_in_patch.append(j)

            indices_of_cells_not_in_patch = new_indices_of_cells_not_in_patch

        patch_orientations.append(np.mean(orientations_in_patch))
        
    patch_orientations_of_cells = [patch_orientations[patch_number - 1] for patch_number in patch_numbers_of_cells]
    return patch_orientations_of_cells
'''

# returns a numpy array
def get_indices_of_cells_in_patches(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    cell_count = len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    indices_of_cells_not_in_patch = list(range(cell_count))
    indices_of_cells_in_patches = []
    patch_count = 0

    # while indices_of_cells_not_in_patch is not empty
    while(indices_of_cells_not_in_patch):
        patch_count += 1
        indices_of_cells_in_patches.append([indices_of_cells_not_in_patch[0]])
        indices_of_cells_not_in_patch = indices_of_cells_not_in_patch[1:]
        for i in indices_of_cells_in_patches[patch_count - 1]:
            new_indices_of_cells_not_in_patch = []
            b_o = cell_orientations[i]

            for j in indices_of_cells_not_in_patch:
                n_o = cell_orientations[j]
                orientation_difference = abs(b_o - n_o)
                orientation_difference = min(orientation_difference, math.pi - orientation_difference)
                in_contact = determine_contact([cell_centers_x[i], cell_centers_y[i]], cell_lengths[i], cell_radii[i], b_o, 
                            [cell_centers_x[j], cell_centers_y[j]], cell_lengths[j], cell_radii[j], n_o, contact_range)

                if (orientation_difference < orientation_threshold and in_contact):
                    indices_of_cells_in_patches[patch_count - 1].append(j)
                    # print(orientation_difference)
                else:
                    new_indices_of_cells_not_in_patch.append(j)

            indices_of_cells_not_in_patch = new_indices_of_cells_not_in_patch
    
    return indices_of_cells_in_patches

def get_patch_numbers(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    indices_of_cells_in_patches = get_indices_of_cells_in_patches(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
    patch_count = len(indices_of_cells_in_patches)
    patch_numbers_of_cells = [0] * len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken

    for i in range(patch_count):
        for j in indices_of_cells_in_patches[i]:
            patch_numbers_of_cells[j] = i + 1
    
    return patch_numbers_of_cells, patch_count

def get_patch_orientations(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    indices_of_cells_in_patches = get_indices_of_cells_in_patches(cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)
    patch_count = len(indices_of_cells_in_patches)
    patch_numbers_of_cells = [0] * len(cell_centers_x) # any other list of cell data (cell_centers_y, etc.) could also have been taken
    patch_orientations = []
    # turn cell_orientations into numpy array
    cell_orientations = np.array(cell_orientations)

    for i in range(patch_count):
        patch_orientations.append( cell_orientations[indices_of_cells_in_patches[i]].mean() )
        for j in indices_of_cells_in_patches[i]:
            patch_numbers_of_cells[j] = i + 1
    
    # patch_orientations = [np.mean([cell_orientations[i] for i in indices_in_patch]) for indices_in_patch in indices_of_cells_in_patches]
    patch_orientations_of_cells = [patch_orientations[patch_number - 1] for patch_number in patch_numbers_of_cells]
    return patch_orientations_of_cells


'''
# The radius used by get_local_anisotropies to decide if a neighbour is in range
radius = 60
'''

# Helper function: Get local anisotropy in orientations around each cell (from cell data)
#                  These anisotropies are averaged to get the local order parameter of the microcolony
# Postcondition: the local anisotropy for a cell at a certain index in the dataframe corresponds
#                to the anisotropy at that index in the list of anisotropies (i.e. the local 
#                anisotropies have a matching order)
def get_local_anisotropies(cell_centers_x, cell_centers_y, cell_orientations, radius):
    local_anisotropies = []
    cell_count = len(cell_centers_x) # cell_centers_y or cell_orientations could also have been taken
    for cell_index in range(cell_count):
        # Number of valid (or in range) neighbours for a cell 
        valid_neighbours = 0
        # Projection matrix
        projection_matrix = np.zeros( shape = (2, 2) )
        b_x = cell_centers_x[cell_index]
        b_y = cell_centers_y[cell_index]

        for neighbour_index in range(cell_count):
            n_x = cell_centers_x[neighbour_index]
            n_y = cell_centers_y[neighbour_index]

            # Calculate the distance between bacteria midpoints
            dist = math.hypot( b_x - n_x, b_y - n_y )

            # Check if the distance between the centers of the two bacteria is less than or equal to the radius
            if (dist <= radius):
                # Get the orientation to calculate projection matrix and increment valid neighbour count 
                n_angle = cell_orientations[neighbour_index]
                valid_neighbours += 1

                # Check for missing rows in csv file
                if (math.isnan(n_angle) == False):
                    # Compute the sum of the projection matrices on the orientation vectors of the neighbouring bacteria
                    projection_matrix = projection_matrix + np.matrix([[math.cos(n_angle)**2, math.cos(n_angle) * math.sin(n_angle)],
                                                                     [math.cos(n_angle) * math.sin(n_angle), math.sin(n_angle)**2]])
        
        # Compute the mean of the projection matrices on the orientation vectors of the neighbouring bacteria
        if (valid_neighbours > 0):
            projection_matrix = projection_matrix / valid_neighbours

        # Get the max real eigenvalues of the mean projection matrix; this is the local anisotropy
        local_anisotropies.append(max(la.eigvals(projection_matrix).real))

    return local_anisotropies