import numpy as np
import rowan
import freud
from scipy.spatial import ConvexHull
import coxeter
from itertools import chain

from ..environment.rdf import rdf_class
from ..environment.bod import bod_class

class orientation_uc:
    """
    Class to compute the orientation of particles in a unit cell.
    
    Parameters
    ----------
    uc_box_list : list
        The box of the unit cell.

    uc_positions : np.ndarray
        Positions of atoms in the unit cell.

    uc_indices : np.ndarray
        Indices of unit cell particles in the replicated system. 

    box_arr_list : np.ndarray
        Box of the replicated system.

    positions : np.ndarray
        Positions of atoms in the replicated system.

    ref_particle_arr : list
        Reference particles for same Wyckoff sites.

    Returns
    -------
    uc_quats : np.ndarray
        Quaternions representing the orientation of particles in the unit cell.
    """

    def __init__(self):
        pass

    def orientation_uc_func(self, uc_box_list, box_arr_list, ref_particle_arr, cluster_centers, group_arr):
        box_arr = freud.box.Box.from_matrix(box_arr_list, dimensions=3)
        uc_box = freud.box.Box.from_matrix(uc_box_list, dimensions=3)

        ref_particle_unique_env = [ref_particle_arr[0]]
        track_arr = [group_arr[ref_particle_arr[0]]]
        for i in range(len(ref_particle_arr)):
            if group_arr[ref_particle_arr[i]] != group_arr[ref_particle_arr[0]] and group_arr[ref_particle_arr[i]] not in track_arr:
                ref_particle_unique_env.append(ref_particle_arr[i])
                track_arr.append(group_arr[ref_particle_arr[i]])


        # Get the vertices of the coord cell of each uc point
        coord_cell_vertices = []
        for i in ref_particle_unique_env:
            # print(group_arr[i])
            coord_verts_i = [cluster_centers[j] for j in group_arr[i]]
            coord_cell_vertices.append(coord_verts_i)

        uc_quats = []
        for i in range(0, len(coord_cell_vertices)):
            hull1 = ConvexHull(coord_cell_vertices[0])
            X = np.array([coord_cell_vertices[0][t] for t in hull1.vertices])
            hull2 = ConvexHull(coord_cell_vertices[i])
            Y = np.array([coord_cell_vertices[i][t] for t in hull2.vertices])

            # Recover the rotation and check
            # R, t, indices = rowan.mapping.icp(X, Y, tolerance=1e-2, return_indices=True)
            R, t = rowan.mapping.kabsch(X, Y)
            # q, t = rowan.mapping.davenport(X, Y)
            q = rowan.from_matrix(R)
            uc_quats.append(q)

        uc_quats = np.array(uc_quats)
        self.uc_quats = uc_quats
        self.ref_particle_unique_env = ref_particle_unique_env
        self.group_arr = group_arr