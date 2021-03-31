#import pandas
import math
import numpy as np

def bacterium_endpoints(center, length, cell_profiler_orientation):
    axis_orientation = -(cell_profiler_orientation + math.pi / 2)
    half_length_along_axis = length / 2 * np.array((math.cos(axis_orientation), math.sin(axis_orientation)))
    p1 = center + half_length_along_axis
    p2 = center - half_length_along_axis
    return p1, p2

'''
COMMENT FROM ORIGINAL AUTHOR
Vector returned (dP) is from the second bac, heading to the first

Geometric Tools for Computer Graphics book,
Also,
http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
Thank you!
'''
# determines whether two rod-shaped cells are in contact, taking the contact threshold into account
def determine_contact(b_center, b_length, b_radius, b_orientation, n_center, n_length, n_radius, n_orientation, contact_range):
    EPS = 1e-12
    b_end1, b_end2 = bacterium_endpoints(b_center, b_length, b_orientation)
    n_end1, n_end2 = bacterium_endpoints(n_center, n_length, n_orientation)

    # Calculates the distance between two cell centers (see link above)
    dist = np.subtract(b_center, n_center)
    rDist = (b_length + n_length) * 0.5 + (b_radius + n_radius)

    # define u as the direction vector of b (i.e. u = b_end2 - b_end1 = vector pointing from one endpoint to the other)
    u = np.subtract(b_end2, b_end1)
    # define v as a vector; v = x2 - x1 (neighbour cell)
    v = np.subtract(n_end2, n_end1)
    # define w as a vector; difference between the two x1 points					
    w = np.subtract(b_end1, n_end1)

    if (dist.dot(dist) < rDist * rDist):
        # always >= 0
        a = u.dot(u)         
        b = u.dot(v)
        # always >= 0
        c = v.dot(v)        
        d = u.dot(w)
        e = v.dot(w)
        # figure out what D is by visualizing what it is; always >= 0
        D = a * c - b * b 	
        sc = D
        sN = D
        # sc = sN / sD, default sD = D >= 0
        sD = D      		
        tc = D
        tN = D
        # tc = tN / tD, default tD = D >= 0
        tD = D      		

        # compute the line parameters of the two closest points
        if (D < EPS): 				# the lines are almost parallel
            sN = 0.0         		# force using point P0 on segment S1
            sD = 1.0       		# to prevent possible division by 0.0 later
            tN = e
            tD = c
        else:                 	# get the closest points on the infinite lines
            sN = (b * e - c * d)
            tN = (a * e - b * d)
            if (sN < 0.0):        	# sc < 0 => the s=0 edge is visible
                sN = 0.0
                tN = e
                tD = c
            elif (sN > sD):  	# sc > 1  => the s=1 edge is visible
                sN = sD
                tN = e + b
                tD = c

        if (tN < 0.0):            	# tc < 0 => the t=0 edge is visible
            tN = 0.0
            # recompute sc for this edge
            if (-d < 0.0):
                sN = 0.0
            elif (-d > a):
                sN = sD
            else:
                sN = -d
                sD = a

        elif (tN > tD):      # tc > 1  => the t=1 edge is visible
            tN = tD
            # recompute sc for this edge
            if ((-d + b) < 0.0):
                sN = 0
            elif ((-d + b) > a):
                sN = sD
            else:
                sN = (-d + b)
                sD = a

        # finally do the division to get sc and tc
        sc = 0.0 if (abs(sN) < EPS) else sN / sD
        tc = 0.0 if (abs(tN) < EPS) else tN / tD

        dP = w + (sc * u) - (tc * v)
        # get length of dP
        neighbourDist = math.sqrt(dP.dot(dP))
        return neighbourDist < contact_range * (b_radius + n_radius)
    else:
        return False