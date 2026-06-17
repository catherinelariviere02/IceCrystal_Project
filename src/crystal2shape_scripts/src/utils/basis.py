"""Basis module for identifying basis positions in the unit cell.

This module provides helper utilities for extracting unique basis positions
from a set of unit cell atomic positions.
"""

import numpy as np
from itertools import chain

def basis_func(box, positions, angle_tol=1.0, dist_cutoff=0.1):
    """Identifies the unique basis positions from unit cell particle coordinates.

    Groups periodic/translated coordinates and computes relative minimum distances
    to select the best representative basis positions.

    Args:
        box (freud.box.Box): The unit cell box object.
        positions (List[List[float]] or np.ndarray): Atom coordinates in the unit cell.
        angle_tol (float): Tolerance in degrees for aligned vector comparison. Defaults to 1.0.
        dist_cutoff (float): Distance tolerance offset around lattice lengths. Defaults to 0.1.

    Returns:
        List[List[float]]: Filtered unique basis positions.
    """
    box_vectors = np.transpose(box.to_matrix())
    V_unit = [vec/np.linalg.norm(vec) for vec in box_vectors]
    abc = [np.linalg.norm(box_vectors[0]), np.linalg.norm(box_vectors[1]), np.linalg.norm(box_vectors[2])]
    array = []
    for i in range(len(positions)):
        subarr = [positions[i]]
        for j in range(len(positions)):
            vec = np.array(positions[j]) - np.array(positions[i])
            if np.linalg.norm(vec) != 0:
                unit_vec = vec/np.linalg.norm(vec)
                d = np.linalg.norm(vec)
                angle0 = np.rad2deg(np.arccos(round(np.dot(V_unit[0], unit_vec), 3)))
                angle1 = np.rad2deg(np.arccos(round(np.dot(V_unit[1], unit_vec), 3)))
                angle2 = np.rad2deg(np.arccos(round(np.dot(V_unit[2], unit_vec), 3)))

                if angle0 <= angle_tol or angle0>= (180-angle_tol):
                    if (abc[0]-dist_cutoff) < d <= (abc[0]+dist_cutoff):
                        subarr.append(positions[j])
                elif angle1 <= angle_tol or angle1>= (180-angle_tol):
                    if (abc[1]-dist_cutoff) < d <= (abc[1]+dist_cutoff):
                        subarr.append(positions[j])
                elif angle2 <= angle_tol or angle2>= (180-angle_tol):
                    if (abc[2]-dist_cutoff) < d <= (abc[2]+dist_cutoff):
                        subarr.append(positions[j])

        cntr = 0
        for j in range(len(subarr)):
            if subarr[j] not in list(chain.from_iterable(array)):
                cntr += 1
        if cntr == len(subarr):
            array.append(subarr)  

        # Flatten the list
        flat_list = [item for sublist in array for item in sublist]
        if len(flat_list) >= len(positions):
            break     

    ref_position = positions[0]
    basis_positions = []
    for i in range(0, len(array)):
        dist_arr = [np.linalg.norm(np.array(array[i][j]) - np.array(ref_position)) for j in range(len(array[i]))]
        min_dist = min(dist_arr)
        min_index = dist_arr.index(min_dist)
        basis_positions.append(array[i][min_index])

    return basis_positions