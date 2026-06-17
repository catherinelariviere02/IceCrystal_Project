import numpy as np
import coxeter
import freud
import rowan
from scipy.spatial import ConvexHull
import itertools

from . import detect_pg
from . import convex_intersection
from  . import pyvista_plot
from . import truncation

class orientation_apply_class:
    def __init__(self):
        pass

    def apply_orientation_func(self, counter, basis_type, sys_types, box_arr_list, 
                  uc_box_list, positions, type_arr, radii_arr, 
                  nearest_env, coord_env_similar_atoms, r, ref_particle, 
                  rcut, t_choice, uc_ids, crystal_type, _radius, particle_ids,
                  original_particle_ids, uc_indices, nearest_ids, original_coord_env_similar_atoms,
                  particles_outside_uc):
        
        """Get the shape of the particle based on its environment and the symmetry of the system.
    
        Parameters:
        ----------
        counter: int
                    The counter for the current atom type being processed.

        basis_type: list
                    The basis types for the particles in the system.

        sys_types: list
                    The system types for the particles in the system.

        box_arr_list: list
                    The list of box arrays for the system.

        uc_box_list: list
                    The list of unit cell box arrays for the system.

        positions: numpy.ndarray
                    The positions of the particles in the system.

        type_arr: list  
                    The type array for the particles in the system.

        radii_arr: list
                    The radii array for the particles in the system.

        nearest_env: numpy.ndarray
                    The nearest environment of the coordination environment.

        coord_env_similar_atoms: numpy.ndarray
                    The coordination environment of similar atoms.

        _radius: float (vander Waals radius/ionic radius/metallic radius)
                    The radius of the particle being processed.

        r: int
            The rounding factor for the coordinates.    

        ref_particle: int
                    The reference particle for the coordination environment.

        rcut: float
                    The cutoff distance for the coordination environment.   

        t_choice: str
                    The type choice for the particle being processed.

        uc_ids: list
                    The indices of the unit cell particles.

        crystal_type: str
                    The type of crystal structure being processed.

        nearest_ids: list
                    The indices of the nearest neighbors of the reference particle.

        uc_indices: list
                    The indices of the unit cell particles in the system.

        original_particle_ids: list
                    The original particle ids of the particles in the system.

        particle_ids: list
                    The particle ids of the particles in the system.

        original_coord_env_similar_atoms: numpy.ndarray
                    The original coordination environment of similar atoms.

        particles_outside_uc: list
                    The particle positions outside the unit cell.

        Returns:
        -------
        None

        """
        ref_point = positions[ref_particle]
        box_arr = freud.box.Box.from_matrix(box_arr_list, dimensions=3)
        uc_box = freud.box.Box.from_matrix(uc_box_list, dimensions=3)

        '''b1, b2, b3 = uc_box_list[:,0], uc_box_list[:,1], uc_box_list[:,2]
        # print(b1, b2, b3)

        # Box vertices
        boxvert1 = (-b1/2.0) + (-b2/2.0) + (-b3/2.0)
        boxvert2 = (-b1/2.0) + (b2/2.0) + (-b3/2.0)
        boxvert3 = (b1/2.0) + (-b2/2.0) + (-b3/2.0)
        boxvert4 = (b1/2.0) + (b2/2.0) + (-b3/2.0)
        boxvert5 = (-b1/2.0) + (-b2/2.0) + (b3/2.0)
        boxvert6 = (-b1/2.0) + (b2/2.0) + (b3/2.0)
        boxvert7 = (b1/2.0) + (-b2/2.0) + (b3/2.0)
        boxvert8 = (b1/2.0) + (b2/2.0) + (b3/2.0)

        box_points = np.array([boxvert1, boxvert3, boxvert2, boxvert5, boxvert7, boxvert6, boxvert4, boxvert8])
'''
        poly = coxeter.shapes.ConvexPolyhedron(coord_env_similar_atoms)
        num_edges, num_faces = len(list(poly.edges)), len(poly.faces)
        # Calculate the point group of the given vertices
        detect_pg_obj = detect_pg.detect_pg_class()
        detect_pg_obj.detect_pg_func(coord_env_similar_atoms, num_edges, num_faces)

        # Attributes of detect_pg_obj
        invariant_quaternions = detect_pg_obj.invariant_quaternions
        symmetry_dict = detect_pg_obj.symmetry_dict

        symm_axes = list(symmetry_dict.values())
        symm_axes = [g for t in symm_axes for g in t]
        # print("Symmetry axes:", symm_axes)

        tr_pt = 0
        for e in range(len(type_arr)):
            if basis_type[counter] == type_arr[e]:
                tr_pt = radii_arr[e] # Truncation point
                rad_pt = _radius[e]  # Radius of the particle being processed

        trn_pts = [np.array(symm_axes[j])*tr_pt for j in range(len(symm_axes))]
        normals = [t/np.linalg.norm(t) for t in trn_pts]

        truncation_obj = truncation.truncation_class()
        truncation_obj.truncation_func(ref_point, np.array(coord_env_similar_atoms), normals, trn_pts)
        truncated_verts = truncation_obj.updated_vertices
        coord_env_similar_atoms = np.array(truncated_verts)
        shape_poly = coord_env_similar_atoms.copy()

        posi_uc = [positions[j] for j in uc_ids]
        # Visualize shape
        poly_vertices = [coord_env_similar_atoms+positions[ref_particle]]
        pyvista_plot_obj = pyvista_plot.pyvista_plot_class(uc_box_list=uc_box_list, positions=posi_uc, lattice_sites=None, poly_vertices=poly_vertices, extra_positions=particles_outside_uc)
        pyvista_plot_obj.pyvista_plot_func()


        # Collect data as attributes
        self.shape_poly = shape_poly

        






