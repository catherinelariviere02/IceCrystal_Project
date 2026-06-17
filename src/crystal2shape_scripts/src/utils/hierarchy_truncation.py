import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from collections import Counter
from itertools import chain
from Geometry3D import *
import freud, coxeter, rowan
import itertools
from skspatial.objects import Plane

from ..operations.truncation import truncation_class
from ..visualization.pyvista_plot import pyvista_plot_class

class hierarchy_truncation_class():
    """
        This class performs hierarchy truncation based on the hierarchy array provided.
        It truncates the hierarchy according to the levels specified by `levelnum_neighbors`.
    """
    def __init__(self):
        pass

    def truncation_hierarchy(self, 
                             system, 
                             unitcell, 
                             group_arr, 
                             system_wyckoff, 
                             levelnum_neighbors, 
                             ref_particle, 
                             uc_box, 
                             positions, 
                             uc_ids,
                            particles_outside_uc, 
                            hierarchy_arr, 
                            hierarchy_coords, 
                            info_poly_coords, 
                            basis_types, 
                            atom_type_arr, 
                            tr_pt_same_type=None, 
                            tr_pt_dic=None, 
                            rtol=0.01, 
                            **kwargs):
        
        
        """

        Parameters:
        -----------
        system : Object
            The system object containing all particle information.

        unitcell : Object
            The unit cell object containing the unit cell information.

        group_arr : list
            A list of environmental information for the system.

        levelnum_neighbors : int
            The number of hierarchy levels to consider for truncation.

        ref_point : np.ndarray
            The reference point in the simulation box, typically the origin or a specific particle's position.

        ref_particle : int
            The index of the reference particle in the positions array.

        box_arr : list
            A Freud Box object defining the simulation box.

        uc_box : freud.box.Box
            The unit cell box defining the periodic boundary conditions.

        positions : np.ndarray
            An array of particle positions in the simulation box.

        uc_ids : list
            A list of indices representing the particles in the unit cell.

        particles_outside_uc : list
            A list of particles that are outside the unit cell.

        hierarchy_arr : list
            A list of lists representing the hierarchy of particles. Each sublist contains particle indices.

        hierarchy_coords : np.ndarray
            An array of coordinates for the particles in the hierarchy.

        info_poly_coords : np.ndarray
            An array of coordinates for the vertices used in constructing the information polyhedron.

        tr_pt_dic : Dict[str, float]
            Dictionary of truncation points for each pair of atom types, used for truncation.

        uc_particle_ids : list
            A list of indices representing the particles in the unit cell.

        rcut_wyckoff : float
            The cutoff distance for the Wyckoff ratio conservation.

        basis_types : list
            A list of basis types corresponding to the particles in the unit cell.

        atom_type : str
            The atom type of the reference particle.

        atom_type_arr : list
            A list of atom types corresponding to the particles in the info polyhedron.

        rtol : float (default=0.01)
            The distance tolerance parameter for numerical comparisons.

        atol : float (default=1.0)
            The angular tolerance parameter for numerical comparisons.

        Returns:
        --------
        stepwise_poly : list
            A list of numpy arrays representing the stepwise updated vertices of the polyhedron after each truncation step.

        Notes:
        -----
        The hierarchy truncation will be according to the array provided as input
        For example, if if the hierarchy arr is [[5, 2, 1], [6, 3, 1], [7, 4, 1], [9, 1], [8, 1]] (considering each integer as particle id)
        It considers the subarray of maximum length first, i.e. the subarray of length 3, [5, 2, 1], [6, 3, 1], [7, 4, 1]
        First it truncates (5, 2), (6, 3), (7, 4) pairs, then it adds the next level of hierarchy, i.e. (2, 1), (3, 1), (4, 1)
        to the rest of the subarray with same length i.e, [9, 1], [8, 1]
        It combines all the subarrays with length 2, and provides the final result.

        Before truncating each pair, it checks ----
        The vector which is getting to be truncated, the distance between the end particle (particle_2 where, vec = particle_2 - particle_1) 
        of that vector and the reference particle should be one of the distances existing in the unit cell length scales
        This need to be confirmed that we don't include any extra particles from the neighborhood unit cell
        Classic case of this phenomena is Hexagonal Diamond

        """
        # Dictionary with the extra parameters
        dic_data = {}
        for key, value in kwargs.items():
            dic_data[str(key)] = value

        uc_lattice_vectors_unit = [t/np.linalg.norm(t) for t in system.box.to_matrix().T]
        lattice_parameters = list(unitcell.box.to_box_lengths_and_angles())
        hierarchical_points_coords_temp = info_poly_coords.copy()
        uc_box_list = np.transpose(uc_box.to_matrix())
        raw_hierarchy_arr = hierarchy_arr[:]  # Storing the original hierarchy array
        lattice_vecs = np.transpose(uc_box.to_matrix())
        uc_lenscales = list(set([round(float(np.linalg.norm(lattice_vecs[i])), dic_data["rounding_factor"]) for i in range(len(lattice_vecs))]))
        
        uc_vecs = [t/np.linalg.norm(t) for t in lattice_vecs]
        # print(uc_lenscales) # Dummy truncation
        hierarchy_track, hierarchy_temp = [], []
        poly_stepwise = []

        # print(p, hierarchy_arr)  # Dummy print
        counter = 0
        for j in range(len(hierarchy_arr)):
            p = len(hierarchy_arr[j]) - 2
            while p >= (levelnum_neighbors[j] - 2):
                # print(p, hierarchy_arr[j][p], hierarchy_arr[j][p+1], len(hierarchy_arr[j]), levelnum_neighbors[j])

                # Making sure that no truncation occurs at any points connected to the ref particle with the lattice vectors
                # Considered particle (particle hierarchy_arr[j][p+1]) is not connected with the ref_particle via any of the lattice vectors
                dits_to_check_uc_satisfaction = round(float(np.linalg.norm((hierarchy_coords[hierarchy_arr[j][p+1]]))), dic_data["rounding_factor"])
                angles_uc_vecs = [round(float(np.rad2deg(np.arccos(np.clip(float(np.dot(uc_vecs[k], (hierarchy_coords[hierarchy_arr[j][p+1]]) / (np.linalg.norm((hierarchy_coords[hierarchy_arr[j][p+1]]))))), -1, 1)))), dic_data["rounding_factor"]) for k in range(len(uc_vecs)) if round(float((np.linalg.norm((hierarchy_coords[hierarchy_arr[j][p+1]])))), dic_data["rounding_factor"]) != 0.0 and len(hierarchy_arr[j]) > 2]
                angles_uc_vecs = [t for t in angles_uc_vecs if t == 0.0 or t == 180.0 or t == 60.0 or t == 120.0]

                if [hierarchy_arr[j][p+1], hierarchy_arr[j][p]] not in hierarchy_track: 
                    # system_positions_list = [[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)] for pos in system.positions]
                    # particle_coord_list = [(system_positions_list[ref_particle] + hierarchy_coords[hierarchy_arr[j][p]]).tolist(), (system_positions_list[ref_particle] + hierarchy_coords[hierarchy_arr[j][p+1]]).tolist()]
                    # particle_coord_list = [[round(float(coord), 3) for coord in particle_coord] for particle_coord in particle_coord_list]
                    # particle_index = [[k for k in range(len(system_positions_list)) if list(set(np.isclose(system_positions_list[k], particle_coord, atol=0.01))) == [True]][0] for particle_coord in particle_coord_list]

                    # if group_arr[ref_particle] not in [group_arr[p] for p in particle_index] or ref_particle in particle_index: # Particles not with identical local environment
                    # if len(angles_uc_vecs) == 0 or dits_to_check_uc_satisfaction not in uc_lenscales:
                    
                    # Treat hexagonal class separately as we don't consider all the "hexagonal" unit cell particles --> Think properly
                    if 168 <= dic_data["space_group_number"] <= 194:
                        if len(angles_uc_vecs) == 0 or dits_to_check_uc_satisfaction not in uc_lenscales:  
                            vec = (hierarchy_coords[hierarchy_arr[j][p+1]] - hierarchy_coords[hierarchy_arr[j][p]])
                        else:
                            vec = None
                    else:
                        vec = (hierarchy_coords[hierarchy_arr[j][p+1]] - hierarchy_coords[hierarchy_arr[j][p]])

                    if vec is not None: 
                        if atom_type_arr[hierarchy_arr[j][p+1]] == atom_type_arr[hierarchy_arr[j][p]]:
                            trunc_pt = tr_pt_same_type     
                        else:
                            trunc_pt = tr_pt_dic[f"{atom_type_arr[hierarchy_arr[j][(p)]]}_{atom_type_arr[hierarchy_arr[j][(p+1)]]}"]

                        trn_pts_local = [vec * trunc_pt]  # Truncation point with respect to the neighbor particle
                        normals = [trn_pts_local[n]/np.linalg.norm(trn_pts_local[n]) for n in range(len(trn_pts_local))]
                        if len(trn_pts_local) > 0:
                            trn_pts = [(trn_pts_local[0] + (hierarchy_coords[hierarchy_arr[j][p]]))]  # Coordinates of the truncation point with respect to the ref_particle
                            line_points = [[(hierarchy_coords[hierarchy_arr[j][p]]+positions[ref_particle]).tolist(), (trn_pts_local[0] + hierarchy_coords[hierarchy_arr[j][p]]+positions[ref_particle]).tolist()]]
                            truncation_obj = truncation_class()
                            posi_uc = [positions[j] for j in uc_ids]
                            type_uc = [system.typeids[j] for j in uc_ids]
                            wyckoff_uc = [system_wyckoff[j] for j in uc_ids]
                            truncation_obj.truncation_func(positions[ref_particle], hierarchical_points_coords_temp, normals, trn_pts, counter=counter,
                                                        positions=positions, uc_ids=uc_ids, uc_box=uc_box, particles_outside_uc=particles_outside_uc,
                                                        ref_particle=ref_particle, line_points=line_points, calc_pointgroup=False, show_poly=dic_data["show_truncation"], posi_uc=posi_uc, wyckoff_uc=wyckoff_uc,
                                                        type_uc=type_uc, extra_positions=particles_outside_uc, extra_particles_types=dic_data["extra_particles_types"], 
                                                        extra_particles_wyckoffs=dic_data["extra_particles_wyckoffs"], directory=dic_data["directory"], shape_id=dic_data["shape_id"])
                                
                            truncated_verts = truncation_obj.updated_vertices
                            # pointgroup = truncation_obj.pointgroup
                            hierarchical_points_coords_temp = np.array(truncated_verts)
                            hierarchy_track.append([hierarchy_arr[j][p+1], hierarchy_arr[j][p]])

                            # print(p, [atom_type_arr[hierarchy_arr[j][p]], atom_type_arr[hierarchy_arr[j][p+1]]], trunc_pt)
                            # print(p, hierarchy_arr[j][p], hierarchy_arr[j][p+1], len(hierarchy_arr[j]), levelnum_neighbors[j])

                            counter += 1
                p = p - 1

            poly_stepwise.append(hierarchical_points_coords_temp.copy())

        self.poly_stepwise = poly_stepwise

    def truncation_single_level(self, 
                                system, 
                                unitcell, 
                                group_arr, 
                                system_wyckoff, 
                                levelnum_neighbors, 
                                ref_particle, 
                                uc_box, 
                                positions, 
                                uc_ids,
                                particles_outside_uc, 
                                hierarchy_arr, 
                                hierarchy_coords, 
                                info_poly_coords, 
                                basis_types, 
                                atom_type_arr, 
                                tr_pt_same_type=None, 
                                tr_pt_dic=None, 
                                rtol=0.01, 
                                **kwargs):


        """

        Parameters:
        -----------
        system : Object
            The system object containing all particle information.

        levelnum_neighbors : int
            The number of hierarchy levels to consider for truncation.

        ref_point : np.ndarray
            The reference point in the simulation box, typically the origin or a specific particle's position.

        ref_particle : int
            The index of the reference particle in the positions array.

        box_arr : list
            A Freud Box object defining the simulation box.

        uc_box : freud.box.Box
            The unit cell box defining the periodic boundary conditions.

        positions : np.ndarray
            An array of particle positions in the simulation box.

        uc_ids : list
            A list of indices representing the particles in the unit cell.

        particles_outside_uc : list
            A list of particles that are outside the unit cell.

        hierarchy_arr : list
            A list of lists representing the hierarchy of particles. Each sublist contains particle indices.

        hierarchy_coords : np.ndarray
            An array of coordinates for the particles in the hierarchy.

        info_poly_coords : np.ndarray
            An array of coordinates for the vertices used in constructing the information polyhedron.

        tr_pt_dic : Dict[str, float]
            Dictionary of truncation points for each pair of atom types, used for truncation.

        uc_particle_ids : list
            A list of indices representing the particles in the unit cell.

        rcut_wyckoff : float
            The cutoff distance for the Wyckoff ratio conservation.
        
        basis_types : list
            A list of basis types corresponding to the particles in the unit cell.

        atom_type : str
            The atom type of the reference particle.
    
        atom_type_arr : list
            A list of atom types corresponding to the particles in the info polyhedron.

        rtol : float (default=0.01)
            The distance tolerance parameter for numerical comparisons. Default is 0.01.

        atol : float (default=1.0)
            The angular tolerance parameter for numerical comparisons. Default is 1.0.

        Returns:
        --------
        stepwise_poly : list
            A list of numpy arrays representing the stepwise updated vertices of the polyhedron after each truncation step.

        Notes:
        -----
        This class is responsible for truncating the hierarchy of particles at a particular level.
        This does not truncate in the hierarchical way.

        """

        # Dictionary with the extra parameters
        dic_data = {}
        for key, value in kwargs.items():
            dic_data[str(key)] = value

        hierarchical_points_coords_temp = info_poly_coords.copy()
        uc_box_list = np.transpose(uc_box.to_matrix())
        raw_hierarchy_arr = hierarchy_arr[:]  # Storing the original hierarchy array
        lattice_vecs = np.transpose(uc_box.to_matrix())
        uc_lenscales = list(set([round(float(np.linalg.norm(lattice_vecs[i])), dic_data["rounding_factor"]) for i in range(len(lattice_vecs))]))
        
        uc_vecs = [t/np.linalg.norm(t) for t in lattice_vecs]
        # print(uc_len_scales) # Dummy truncation
        hierarchy_track = []
        poly_stepwise = []
        #**************************************************************
        # It does not consider the entire info polyhedron for truncation
        # It only chooses the vertices from the last level which will be truncated
        # For example, if the truncation level is 5-th, it only considers the vertices from the 5-th level and truncate only once

        # print("hh", l, hierarchy_arr)  # Dummy print
        counter = 0
        for j in range(len(hierarchy_arr)):
            l_tr = levelnum_neighbors[j]
            # print(hierarchy_arr[j][(l_tr-2)], hierarchy_arr[j][(l_tr-1)], len(hierarchy_arr[j]), l_tr)  # Dummy print 
            
            dits_to_check_uc_satisfaction = round(float(np.linalg.norm((hierarchy_coords[hierarchy_arr[j][l_tr-1]]))), dic_data["rounding_factor"])
            angles_uc_vecs = [round(float(np.rad2deg(np.arccos(round(float(np.dot(uc_vecs[k], (hierarchy_coords[hierarchy_arr[j][l_tr-1]]) / (np.linalg.norm((hierarchy_coords[hierarchy_arr[j][l_tr-1]]))))), 1)))), dic_data["rounding_factor"]) for k in range(len(uc_vecs)) if round(float((np.linalg.norm((hierarchy_coords[hierarchy_arr[j][l_tr-1]])))), dic_data["rounding_factor"]) != 0.0 and len(hierarchy_arr[j]) > 2]
            angles_uc_vecs = [t for t in angles_uc_vecs if t == 0.0 or t == 180.0 or t == 60.0 or t == 120.0]

            if [hierarchy_arr[j][(l_tr-2)], hierarchy_arr[j][(l_tr-1)]] not in hierarchy_track :
                # system_positions_list = [[round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(pos[2]), 3)] for pos in system.positions]
                # particle_coord_list = [(system_positions_list[ref_particle] + hierarchy_coords[hierarchy_arr[j][l_tr-2]]).tolist(), (system_positions_list[ref_particle] + hierarchy_coords[hierarchy_arr[j][l_tr-1]]).tolist()]
                # particle_coord_list = [[round(float(coord), 3) for coord in particle_coord] for particle_coord in particle_coord_list]
                # particle_index = [[k for k in range(len(system_positions_list)) if list(set(np.isclose(system_positions_list[k], particle_coord, atol=0.01))) == [True]][0] for particle_coord in particle_coord_list]

                # Treat hexagonal class separately as we don't consider all the "hexagonal" unit cell particles --> Think properly
                if 168 <= dic_data["space_group_number"] <= 194:
                    if len(angles_uc_vecs) == 0 or dits_to_check_uc_satisfaction not in uc_lenscales:
                        vec = (hierarchy_coords[hierarchy_arr[j][(l_tr-1)]] - hierarchy_coords[hierarchy_arr[j][(l_tr-2)]])
                    else:
                        vec = None
                else:
                    vec = (hierarchy_coords[hierarchy_arr[j][(l_tr-1)]] - hierarchy_coords[hierarchy_arr[j][(l_tr-2)]])

                if vec is not None:
                    if atom_type_arr[hierarchy_arr[j][(l_tr-1)]] == atom_type_arr[hierarchy_arr[j][(l_tr-2)]]:
                        trunc_pt = tr_pt_same_type
                    else:
                        trunc_pt = tr_pt_dic[f"{atom_type_arr[hierarchy_arr[j][(l_tr-2)]]}_{atom_type_arr[hierarchy_arr[j][(l_tr-1)]]}"]

                    trn_pts_local = [vec * trunc_pt]  # Truncation point with respect to the neighbor particle

                    normals = [trn_pts_local[n]/np.linalg.norm(trn_pts_local[n]) for n in range(len(trn_pts_local))]
                    if len(trn_pts_local) > 0:
                        normals = np.array(normals)
                        trn_pts = [trn_pts_local[0] + (hierarchy_coords[hierarchy_arr[j][(l_tr-2)]])]  # Coordinates of the truncation point with respect to the ref_particle

                        line_points = [[(hierarchy_coords[hierarchy_arr[j][(l_tr-2)]] + system.positions[ref_particle]).tolist(), (trn_pts_local[0] + hierarchy_coords[hierarchy_arr[j][(l_tr-2)]] + system.positions[ref_particle]).tolist()]]
                        truncation_obj = truncation_class()
                        posi_uc = [positions[j] for j in uc_ids]
                        type_uc = [system.typeids[j] for j in uc_ids]
                        wyckoff_uc = [system_wyckoff[j] for j in uc_ids]
                        truncation_obj.truncation_func(positions[ref_particle], hierarchical_points_coords_temp, normals, trn_pts, counter=counter,
                                                    positions=positions, uc_ids=uc_ids, uc_box=uc_box, particles_outside_uc=particles_outside_uc,
                                                    ref_particle=ref_particle, line_points=line_points, calc_pointgroup=False, show_poly=dic_data["show_truncation"], posi_uc=posi_uc, wyckoff_uc=wyckoff_uc,
                                                    type_uc=type_uc, extra_positions=particles_outside_uc, extra_particles_types=dic_data["extra_particles_types"], 
                                                    extra_particles_wyckoffs=dic_data["extra_particles_wyckoffs"], directory=dic_data["directory"], shape_id=dic_data["shape_id"])
                        
                        truncated_verts = truncation_obj.updated_vertices
                        # pointgroup = truncation_obj.pointgroup
                        hierarchical_points_coords_temp = np.array(truncated_verts)
                        hierarchy_track.append([hierarchy_arr[j][(l_tr-2)], hierarchy_arr[j][(l_tr-1)]])

                        counter += 1

        poly_stepwise.append(hierarchical_points_coords_temp.copy())

        self.poly_stepwise = poly_stepwise