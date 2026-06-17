"""
Shape Analyzer Module
=====================

This module defines the :class:`ShapeAnalyzer` class, which computes the coordination polyhedron
(information polyhedron) around a reference particle by analyzing local environments,
constructing shell hierarchies, and applying hierarchical truncation.
"""

import logging
import coxeter
import numpy as np
import freud
from scipy.spatial import ConvexHull, Delaunay
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from skspatial.objects import Points
import sys
from typing import List, Dict, Tuple, Any, Optional, Set
import spglib

# Internal project imports
from ..utils.make_hierarchy import Hierarchy
from ..utils.hierarchy_truncation import hierarchy_truncation_class
from ..visualization.pyvista_plot import pyvista_plot_class
from ..visualization.plot_hierarchy import plot_hierarchy_class
from ..utils.find_asymmetric_unit import find_asymmetric_unit_class
from ..utils.utils_func import (
    _get_particle_ids_coords_rcut, 
    _get_pointgroup_symmetry_equivalent_invariant_points, 
    _get_particles_uc,
    _get_equivalent_invariant_points,
    _get_pointgroup_neighbors_of_neighbors,
    safe_input
)

# Initialize logging
logger = logging.getLogger(__name__)

class ShapeAnalyzer:
    """Computes coordination polyhedra (shapes) around reference crystal particles.

    This class coordinates the workflow of analyzing local neighbor shell hierarchies,
    performing hierarchical plane-cuts (truncations), and rendering intermediate or final
    polyhedra visualizations.

    Attributes:
        shape_poly (Optional[np.ndarray]): Vertices of the final computed polyhedron shape.
        ref_particle (Optional[int]): ID of the reference particle.
        hierarchy_arr (Optional[List[List[int]]]): Neighborhood shells grouping particle IDs.
        info_poly_coords (Optional[np.ndarray]): Coordinates of the info polyhedron vertices.
        info_poly_particle_ids (Optional[List[int]]): Particle IDs in the info polyhedron.
        hierarchy_coords (Optional[np.ndarray]): Relative coordinates of hierarchy particles.
        rcut_wyckoff (Optional[float]): Cutoff distance corresponding to the Wyckoff position.
        infopoly_pg (Optional[str]): Schoenflies point group symbol of the info polyhedron.
        line_colors (Optional[List[str]]): Colors for rendering hierarchy bonds.
    """

    def __init__(self):
        """Initializes the ShapeAnalyzer with default placeholder states."""
        # Attributes required for external main function compatibility
        self.shape_poly = None
        self.ref_particle = None
        self.hierarchy_arr = None
        self.info_poly_coords = None
        self.info_poly_particle_ids = None
        self.hierarchy_coords = None
        self.rcut_wyckoff = None
        self.infopoly_pg = None
        self.line_colors = None

    def construct_hierarchy(self, system: Any, unitcell: Any, uc_particle_indices: List[int], particle: int, 
                            rcut: float, atom_type: str, equiv_atom_list: Optional[List[int]], envelop_track_type: int,
                            rtol: float = 0.01, pointgroup: bool = False, hierarchy_tol: float = 0.1, showPoly: bool = True,
                            atom_type_selection: str = "True", **kwargs) -> 'ShapeAnalyzer':
        """Constructs the particle neighborhood hierarchy within a specified cutoff.

        Groups neighbor particles into discrete distance shells (hierarchy layers) relative to
        the central reference particle.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Indices of particles belonging to the reference unit cell.
            particle (int): ID of the central reference particle.
            rcut (float): Cutoff radius to search for neighbors.
            atom_type (str): Basis type name of the active atom being analyzed.
            equiv_atom_list (Optional[List[int]]): List of equivalent atom IDs.
            envelop_track_type (int): Track code determining behavior of equivalent atom filtering.
            rtol (float): Relative distance tolerance. Defaults to 0.01.
            pointgroup (bool): If True, analyzes and logs the point group symmetry of the shell. Defaults to False.
            hierarchy_tol (float): Distance tolerance for grouping neighbors into shells. Defaults to 0.1.
            showPoly (bool): If True, plots and saves the envelope polyhedron visualization. Defaults to True.
            atom_type_selection (str): 'True' to filter neighbors by `atom_type`, 'False' otherwise. Defaults to "True".
            **kwargs: Extra parameters dynamically set as instance attributes.

        Returns:
            ShapeAnalyzer: Self instance with computed `hierarchy_arr` and `hierarchy_coords`.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        rounding_factor = self.rd_factor

        uc_lattice_vectors = np.transpose(unitcell.box.to_matrix())
        ref_point = system.positions[particle]
        
        # 1. Gather neighbors
        particle_ids, _ = _get_particle_ids_coords_rcut(
            system, unitcell, particle, uc_particle_indices, rcut, rtol=rtol
        )

        # 2. Filter by type and equivalent atoms
        if atom_type_selection == "True":
            particle_ids = [j for j in particle_ids if system.typeids[j] == atom_type]

        if equiv_atom_list is not None and envelop_track_type == 0:
            particle_ids = [j for j in particle_ids if system.equiv_atoms[j] in equiv_atom_list]
        
        if len(particle_ids) > 0:
            particle_ids.append(particle)
            species_label = [system.typeids[j] for j in particle_ids]
            particle_coords = np.array([system.box.wrap(system.positions[j] - ref_point) for j in particle_ids])
            max_distance = max([round(float(np.linalg.norm(unitcell.box.wrap(system.positions[j] - ref_point))), self.r) for j in particle_ids])

            # 3. Hierarchy construction for the neighbors within rcut_wyckoff for the current basis type
            hierarchy_obj = Hierarchy()
            hierarchy_obj.make_hierarchy(system, particle_coords, max_distance, hierarchy_tol, r=self.r)
            
            self.hierarchy_arr = hierarchy_obj.hierarchy_arr
            self.hierarchy_coords = particle_coords
            
            h_lens = [len(h) for h in self.hierarchy_arr]
            vals, counts = np.unique(h_lens, return_counts=True)
            max_count = max(counts)
            max_count_indices = [i for i, count in enumerate(counts) if count == max_count]
            self.max_hierarchy_len = min([vals[i] for i in max_count_indices])
            hierarchy_arr_dic = {int(val): int(count) for val, count in zip(vals, counts)}
            print(f"Hierarchy lengths and counts: {hierarchy_arr_dic}")
            print(f"Max hierarchy length: {self.max_hierarchy_len}")

            # VisualizeAsymmetric unit 
            if showPoly == True:
                # Envelop poly
                hull_ = ConvexHull(particle_coords)
                envelop_poly = np.array([particle_coords[j] for j in hull_.vertices])
                poly_vertices = [envelop_poly + system.positions[particle]]
                self._asymmetricunit_visualization(system, unitcell, uc_particle_indices, particle,
                                                    poly_vertices=poly_vertices,
                                                    particles_outside_uc=self.particles_outside_uc, 
                                                    extra_particles_types=self.extra_particles_types, 
                                                    extra_particles_wyckoffs=self.extra_particles_wyckoffs,
                                                    line_points=None,
                                                    directory=self.directory, 
                                                    shape_id=self.shape_id,
                                                    outfilename=self.filename,
                                                    color=self.color)

            # 4. Point Group Analysis
            if pointgroup:
                mol = Molecule(species=species_label, coords=particle_coords)
                self.infopoly_pg = PointGroupAnalyzer(mol, tolerance=0.1).sch_symbol
                logger.info(f"Info poly PG detected: {self.infopoly_pg}")

        else:
            self.hierarchy_arr = []
            self.hierarchy_coords = None
            self.max_hierarchy_len = 0

        self.particle_ids = particle_ids
        self.particle_coords = particle_coords

    def compute(self, unitcell: Any, system: Any, uc_particle_indices: List[int], atom_type: Any, 
                ref_particle: int, ref_particle_arr: List[int], rcut_arr_all: List[float], tr_pt_dic: Dict[str, float], 
                tr_pt: List[float], basis_types: List[Any], rtol: float = 0.01, space_group_number: int = 1, 
                atom_type_selection: str = "True", **kwargs) -> 'ShapeAnalyzer':
        """Runs the interactive iterative truncation pipeline to compute the final shape.

        Iteratively queries the user for cutoff values (`rcut_wyckoff`) for each basis type,
        constructs local hierarchies, executes plane-cut truncations, and renders visualizations.

        Args:
            unitcell (freud.data.UnitCell): The unit cell object.
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_particle_indices (List[int]): Indices of particles belonging to the reference unit cell.
            atom_type (Any): Active atom type.
            ref_particle (int): Reference particle ID.
            ref_particle_arr (List[int]): List of reference particle IDs per basis type.
            rcut_arr_all (List[float]): Calculated max cutoff distances per basis type.
            tr_pt_dic (Dict[str, float]): Pairwise truncation distance ratios.
            tr_pt (List[float]): Self-truncation ratio values.
            basis_types (List[Any]): List of active basis atom types.
            rtol (float): Relative distance tolerance. Defaults to 0.01.
            space_group_number (int): Crystal space group number. Defaults to 1.
            atom_type_selection (str): 'True' to filter by type, 'False' otherwise. Defaults to "True".
            **kwargs: Extra parameters dynamically set as instance attributes.

        Returns:
            ShapeAnalyzer: Self instance with updated `shape_poly`.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        r = self.rd_factor
        
        # 1. Visualization: Voronoi (Optional)
        if str(self.show_voronoi) == "True":
            self._plot_voronoi(system, unitcell, uc_particle_indices, ref_particle)

        # 2. Visualization: Unit Cell (Optional)
        if str(self.show_unitcell) == "True":
            self._plot_unit_cell(system, unitcell, uc_particle_indices)

        # Take all equivalent atom types for the atom type of the reference particle
        equiv_atom_list = list(set(system.equiv_atoms))

        # 4. Iterative Truncation Pipeline
        modified_basis_type = [system.typeids[ref_particle]] + [t for t in basis_types if t != system.typeids[ref_particle]]
        equiv_atom_indices_type = np.unique([system.equiv_atoms[i] for i in uc_particle_indices if system.typeids[i] == system.typeids[ref_particle]]).tolist()
        equiv_atom_ids = [uc_particle_indices[i] for i in equiv_atom_indices_type]

        envelop_track_type = 0
        envelop_poly_coords = []

        for b_type in modified_basis_type:
            self.current_rcut_wyckoff = safe_input(
                prompt=f"Enter rcut_wyckoff for {b_type}: ",
                expected_type=float
            )

            # Interactive rcut_wyckoff and rcut fallback for the first iteration (reference particle type)
            try:
                self.rcut = round(float(rcut_arr_all[basis_types.index(system.typeids[ref_particle])][ref_particle_arr[basis_types.index(system.typeids[ref_particle])].index(ref_particle)]), self.rd_factor)
            except (ValueError, IndexError, KeyError, TypeError):
                self.rcut = self.current_rcut_wyckoff

            if self.current_rcut_wyckoff != -1 : # Set the current_rcut_wyckoff value to -1 if not found
                # 1. Particle ids and coordinates WITHIN the given rcut but outside unit cell -- for visualization
                particle_ids_rcut, particle_coords_rcut = _get_particle_ids_coords_rcut(system,      
                                                                                             unitcell,
                                                                                             ref_particle,
                                                                                             uc_particle_indices,
                                                                                             self.rcut,
                                                                                             rtol=rtol)
                
                particle_ids_rcut.append(ref_particle) # Including the reference particle id
                # 2. particles outside UC -- for visualization only
                particles_outside_uc = [system.positions[j].tolist() for j in particle_ids_rcut if j not in uc_particle_indices]
                if len(particles_outside_uc) == 0:
                    particles_outside_uc = None

                particle_ids_outside_uc = [int(j) for j in particle_ids_rcut if j not in uc_particle_indices]
                extra_particles_types = [system.typeids[j] for j in particle_ids_outside_uc] if particles_outside_uc is not None else None
                extra_particles_wyckoffs = [system.wyckoffs[j] for j in particle_ids_outside_uc] if particles_outside_uc is not None else None

                try:
                    # Get invariant and equivalent points  within max_len from the reference particle
                    equivalent_invariant_points = _get_equivalent_invariant_points(system, unitcell, ref_particle, uc_particle_indices, b_type, 
                                                                                round(self.rcut, r), rtol, pg_tolerance=self.pg_tolerance)
                    # Get PG for neighbors of neighbors at rcut_wyckoff for the current basis type
                    pg_sym = _get_pointgroup_neighbors_of_neighbors(system, 
                                        unitcell, 
                                        self.current_rcut_wyckoff,
                                        ref_particle, 
                                        uc_particle_indices, 
                                        atom_type=b_type,
                                        rtol=rtol,
                                        pg_tolerance=self.pg_tolerance,
                                        extra_positions=particles_outside_uc, 
                                        extra_particles_types=extra_particles_types,
                                        extra_particles_wyckoffs=extra_particles_wyckoffs,
                                        show=False,
                                        directory=self.directory,
                                        shape_id=self.shape_id,
                                        dist_map=self.dist_map,
                                        length_contraction=self.length_contraction)
                    
                    show_poly_flag = True if envelop_track_type == 0 else False

                    # 4. Construct environment hierarchy at rcut_wyckoff for the current basis type
                    outfilename = self.directory + "temp_files/" + self.shape_id + "_asymmetric_unit_" + str(b_type) + "_equivid" + str(system.equiv_atoms[ref_particle]) + "_rcut_" + str(self.current_rcut_wyckoff).replace(".", "p")+ ".png"
                    self.construct_hierarchy(
                        system, unitcell, uc_particle_indices, ref_particle, 
                        self.current_rcut_wyckoff, b_type, equiv_atom_list, envelop_track_type, 
                        rtol=rtol, atom_type_selection=atom_type_selection, 
                        hierarchy_tol=self.hierarchy_tol,
                        showPoly=show_poly_flag, particles_outside_uc=particles_outside_uc, 
                        extra_particles_types=extra_particles_types, 
                        extra_particles_wyckoffs=extra_particles_wyckoffs,
                        directory=self.directory, shape_id=self.shape_id,
                        filename=None,
                        r=r,
                        color="lightcoral",
                        _name="asymmetric_unit"
                    )

                    particle_ids_rcut = self.particle_ids
                    particle_coords_rcut = self.particle_coords
                    hierarchy_arr_rcut = self.hierarchy_arr
                    hierarchy_coords_rcut = self.hierarchy_coords
                    max_hierarchy_length_rcut = self.max_hierarchy_len

                    print(" ")

                    # 5. Decide which hierarchy to use for truncation
                    considered_particle_ids = particle_ids_rcut
                    considered_particle_coords = particle_coords_rcut
                    considered_hierarchy_arr = hierarchy_arr_rcut
                    considered_hierarchy_coords = hierarchy_coords_rcut
                    max_hierarchy_len = max_hierarchy_length_rcut
                    logging.info("Hierarchy array calculated!")

                    # 6. Show basis points if the user opts for it
                    if self.show_hierarchy == "True":
                        self._hierarchy_bonds(system, unitcell, uc_particle_indices, ref_particle, considered_hierarchy_arr, considered_hierarchy_coords,
                                            particles_outside_uc=particles_outside_uc, extra_particles_types=extra_particles_types, extra_particles_wyckoffs=extra_particles_wyckoffs)

                    # 7. Show replicated asymmetric unit if the user opts for it
                    if self.replicated_asym_unit_show == "True":
                        self._replicate_asymunit_get_basis_points(system, unitcell, uc_particle_indices, ref_particle,
                                                     r, rtol, atom_type, space_group_number, show_basis=self.show_basis, 
                                                     replicated_asym_unit_show=self.replicated_asym_unit_show, directory=self.directory, 
                                                     shape_id=self.shape_id, atom_type_selection=atom_type_selection,
                                                     type_id=b_type)

                    # Basic setup such as modified hierarchy arr (if necessary) and levelnum_neighbors_arr -- # Truncation Execution
                    if envelop_track_type == 0:
                        hull = ConvexHull(considered_particle_coords)
                        envelop_poly_coords = np.array([considered_particle_coords[i] for i in hull.vertices])

                        envelop_hull = ConvexHull(particle_coords_rcut)
                        self.envelop_vertices = [particle_coords_rcut[v] for v in envelop_hull.vertices]
                        
                        atom_type_arr = [system.typeids[j] for j in considered_particle_ids] # Atom types of the particles in the hierarchy for truncation purposes
                        levelnum_neighbors_arr = [max_hierarchy_len if len(h) >= max_hierarchy_len else len(h) for h in considered_hierarchy_arr]

                    elif envelop_track_type > 0: # For atom types different from the reference atom type
                        atom_type_arr = [system.typeids[j] for j in considered_particle_ids]
                        levelnum_neighbors_arr = [max_hierarchy_len if len(h) >= max_hierarchy_len else len(h) for h in considered_hierarchy_arr]

                    if considered_hierarchy_arr != None:
                        # 8. Hierarchical Truncation
                        trunc_obj = hierarchy_truncation_class()
                        trunc_obj.truncation_hierarchy(
                            system=system, unitcell=unitcell, group_arr=system.local_env_cluster_ids,
                            system_wyckoff=system.wyckoffs, levelnum_neighbors=levelnum_neighbors_arr,
                            ref_particle=ref_particle, uc_box=unitcell.box, positions=system.positions,
                            uc_ids=uc_particle_indices, hierarchy_arr=considered_hierarchy_arr,
                            hierarchy_coords=considered_hierarchy_coords, info_poly_coords=envelop_poly_coords,
                            basis_types=basis_types, atom_type_arr=atom_type_arr,
                            tr_pt_same_type=tr_pt, tr_pt_dic=tr_pt_dic, rtol=rtol, rounding_factor=r, line_points=self.line_points,
                            space_group_number=space_group_number, directory=self.directory, shape_id=self.shape_id,
                            particles_outside_uc=particles_outside_uc, extra_particles_types=extra_particles_types, 
                            extra_particles_wyckoffs=extra_particles_wyckoffs, show_truncation=self.show_truncation
                        )

                        # Update poly for next iteration
                        self.shape_poly = np.array(trunc_obj.poly_stepwise[-1])
                        envelop_poly_coords = self.shape_poly.copy()

                        # Increase the envelop_track_type to move to the next basis type in the next iteration
                        envelop_track_type += 1

                except Exception as e:
                    logger.error(f"Iterative truncation step failed for {b_type}: {e}")
                
                if atom_type_selection==False:
                    break

        self._final_visualization(system, unitcell, uc_particle_indices, ref_particle, 
                                poly_vertices=[self.shape_poly + system.positions[ref_particle]],
                                directory=self.directory, shape_id=self.shape_id,
                                color="melon")
        
    def _get_envelop_polyhedron(self, system, unitcell, uc_particle_indices, particle, type_ids, **kwargs) -> Optional[np.ndarray]:
        """Calculates the envelop polyhedron for a reference particle.

        Used for non-primitive lattices (face-centered, body-centered, base-centered) where the
        asymmetric unit does not equal the Bravais lattice envelope.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Indices of particles in the unit cell.
            particle (int): Reference particle ID.
            type_ids (str): Target atom type ID.
            **kwargs: Extra parameters dynamically set as instance attributes.

        Returns:
            Optional[np.ndarray]: Computed envelop polyhedron vertices, or None if calculation fails.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        try:
            p_ids = [i for i in uc_particle_indices if system.typeids[i] == type_ids and system.local_env_cluster_ids[i] == system.local_env_cluster_ids[particle]] # Consider only the same type and same environemnt as the reference particle for the envelop polyhedron
            max_dist = round(float(max([np.linalg.norm(unitcell.box.wrap(system.positions[p] - system.positions[particle])) for p in p_ids])), self.r)
            if max_dist == 0.0:
                return None
            
            p_ids, _ = _get_particle_ids_coords_rcut(system, unitcell, particle, uc_particle_indices, max_dist, rtol=pow(10, -self.r))
            p_ids = [j for j in p_ids if system.typeids[j] == type_ids]
            p_ids.append(particle) # Including the reference particle id
            p_coords = [(system.positions[i] - system.positions[particle]) for i in p_ids]
            hull = ConvexHull(p_coords)
            hull_vertices = np.array([p_coords[j] for j in hull.vertices])

            # Make hierarchy 
            hierarchy_obj = Hierarchy()
            hierarchy_obj.make_hierarchy(system, p_coords, max_dist, r=self.r)
            
            hierarchy_arr = hierarchy_obj.hierarchy_arr
            hierarchy_coords = p_coords

            h_lens = [len(h) for h in hierarchy_arr]
            vals, counts = np.unique(h_lens, return_counts=True)
            max_count = max(counts)
            max_count_indices = [i for i, count in enumerate(counts) if count == max_count]
            max_hierarchy_len = min([vals[i] for i in max_count_indices])
            hierarchy_arr_dic = {int(val): int(count) for val, count in zip(vals, counts)}
            print(f"Hierarchy lengths and counts: {hierarchy_arr_dic}")

            # Truncation at the maximum hierarchy length to get the envelop polyhedron for the reference particle type in the first iteration
            levelnum_neighbors_arr = [len(h) for h in hierarchy_arr]
            atom_type_arr = [system.typeids[j] for j in p_ids]
            
            trunc_obj = hierarchy_truncation_class()
            trunc_obj.truncation_single_level(
                system=system, unitcell=unitcell, group_arr=system.local_env_cluster_ids,
                system_wyckoff=system.wyckoffs, levelnum_neighbors=levelnum_neighbors_arr,
                ref_particle=particle, uc_box=unitcell.box, positions=system.positions,
                uc_ids=uc_particle_indices, hierarchy_arr=hierarchy_arr,
                hierarchy_coords=hierarchy_coords, info_poly_coords=hull_vertices,
                basis_types=self.basis_types, atom_type_arr=atom_type_arr,
                tr_pt_same_type=self.tr_pt_same_type, tr_pt_dic=self.tr_pt_dic, rtol=self.rtol, rounding_factor=self.r, line_points=None,
                space_group_number=self.space_group_number, directory=self.directory, shape_id=self.shape_id,
                particles_outside_uc=self.particles_outside_uc, extra_particles_types=self.extra_particles_types, 
                extra_particles_wyckoffs=self.extra_particles_wyckoffs, show_truncation=self.show_truncation
            )

            shape_poly = np.array(trunc_obj.poly_stepwise[-1])
            envelop_poly = shape_poly.copy()

            # Visulaize the shape
            pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_particle_indices],
                                types=[system.typeids[j] for j in uc_particle_indices], 
                                wyckoff_uc=[system.wyckoffs[j] for j in uc_particle_indices],
                                poly_vertices=[envelop_poly + system.positions[particle]],
                                line_points=self.line_points,
                                extra_positions=self.particles_outside_uc, 
                                extra_particles_types=self.extra_particles_types,
                                extra_particles_wyckoffs=self.extra_particles_wyckoffs,
                                center=None,
                                direction=None,
                                text="Envelop Polyhedron",
                                filename=None,
                                color=self.color).pyvista_plot_func()

            return envelop_poly
        except Exception as e:
            logger.error(f"Envelop polyhedron calculation failed for particle {particle}: {e}")
            return None

    def _check_geometry(self, vertices) -> Tuple[float, float]:
        """Analyzes geometric invariants (Moment of Inertia parameter and face/edge ratio) of a polyhedron.

        Args:
            vertices (np.ndarray): Polyhedron vertices.

        Returns:
            Tuple[float, float]: A tuple of the MOI shape parameter and the face-to-edge ratio.
        """
        import coxeter
        from ..utils.utils_func import _moment_of_inertia

        hull = ConvexHull(vertices)
        vertices = np.array([vertices[i] for i in hull.vertices])
        poly = coxeter.shapes.ConvexPolyhedron(vertices)
        moi = _moment_of_inertia(poly.vertices)
        # Reduced MOI parameter
        m = [moi[0], moi[3], moi[5]]
        moi_param = 1 - (((m[0]**2 - m[1]**2)**2 + (m[1]**2 - m[2]**2)**2 + (m[2]**2 - m[0]**2)**2) / (2 * sum(i**2 for i in m)**2))
        face_edge_ratio = (float(len(poly.faces)) / float(len(poly.edges)))

        return moi_param, face_edge_ratio

    def _check_points_inside_polyhedron(self, system, unitcell, uc_particle_indices, ref_particle, 
                                        particle_coords, particle_ids, polyhedron_vertices) -> Tuple[np.ndarray, List[List[int]], int, List[str]]:
        """Filters particles that lie inside a given polyhedron and returns their hierarchy properties.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
            particle_coords (np.ndarray): Relative coordinates of target particles.
            particle_ids (List[int]): IDs of target particles.
            polyhedron_vertices (np.ndarray): Vertices defining the polyhedron boundary.

        Returns:
            Tuple[np.ndarray, List[List[int]], int, List[str]]: A tuple containing:
                - Coordinates of points inside the polyhedron.
                - Hierarchy shell grouping array.
                - Chosen hierarchy cutoff length based on frequency.
                - Active atom type array of inside points.
        """
        extended_polyhedron_vertices = polyhedron_vertices * (1.05)
        delaunay = Delaunay(np.array(extended_polyhedron_vertices))
        inside_coords, inside_particle_ids = [], []
        for v in range(len(particle_coords)):
            if delaunay.find_simplex(np.array(particle_coords[v])) >= 0:
                inside_coords.append(particle_coords[v])
                inside_particle_ids.append(particle_ids[v])

        if len(inside_particle_ids) > 1:
            hierarchy_coords = inside_coords.copy()
            Hierarchy_obj = Hierarchy()
            Hierarchy_obj.make_hierarchy(system, hierarchy_coords, ref_particle)
            hierarchy_arr = Hierarchy_obj.hierarchy_arr
            hierarchy_len = [len(Hierarchy_obj.hierarchy_arr[i]) for i in range(len(Hierarchy_obj.hierarchy_arr))]
            hierarchy_coords = [[float(t[0]), float(t[1]), float(t[2])] for t in hierarchy_coords]
            hierarchy_coords = np.array(hierarchy_coords)

            vals, counts = np.unique(hierarchy_len, return_counts=True)
            vals, counts = [int(i) for i in vals], [int(i) for i in counts]
            print("Hierarchy dictionary  -- inside: ", dict(zip(vals, counts)))
            max_count = max(counts)
            indices = [i for i, j in enumerate(counts) if j == max_count]
            max_len_selected = max([vals[k] for k in indices]) # Choosing max frequency hierarchy length for truncation
            
            atom_type_arr = [system.typeids[j] for j in inside_particle_ids]

        else:
            hierarchy_coords = None
            hierarchy_arr = None
            max_len_selected = 0
            atom_type_arr = []

        return hierarchy_coords, hierarchy_arr, max_len_selected, atom_type_arr

    def _detect_symmetry(self, particle, rcut, system, unitcell, uc_particle_indices, atom_type, tolerance, wyckoff_choice=False) -> str:
        """Determines the point group symmetry Schoenflies symbol using Pymatgen PointGroupAnalyzer.

        Args:
            particle (int): Reference particle ID.
            rcut (float): Cutoff radius for neighbor retrieval.
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Indices of particles in the unit cell.
            atom_type (str): Target atom type.
            tolerance (float): Point group analysis tolerance.
            wyckoff_choice (bool): If True, filters neighbors by the same Wyckoff position. Defaults to False.

        Returns:
            str: Schoenflies point group symbol, or "None" if insufficient points or coplanar.
        """
        p_ids, p_coords = _get_particle_ids_coords_rcut(system, unitcell, particle, uc_particle_indices, rcut, rtol=0.01)
        # Filter for identical type
        if wyckoff_choice == True:
            coords = [p_coords[i] for i, pid in enumerate(p_ids) if system.typeids[pid] == atom_type and system.wyckoffs[pid]==system.wyckoffs[particle]]
        else:
            coords = [p_coords[i] for i, pid in enumerate(p_ids) if system.typeids[pid] == atom_type]

        if len(coords) < 4 or Points(coords).are_coplanar(tol=1e-2):
            return "None"
        
        else:
            mol = Molecule(species=[atom_type]*len(coords), coords=coords)
            pga = PointGroupAnalyzer(mol, tolerance=tolerance)
            return pga.sch_symbol

    def _plot_voronoi(self, system, unitcell, uc_indices, ref_particle):
        """Visualizes the Voronoi cell polytope of a reference particle.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
        """
        voro = freud.locality.Voronoi()
        cells = voro.compute((system.box, system.positions)).polytopes
        voro_vertices = cells[ref_particle]
        voro_centroid = np.mean(voro_vertices, axis=0)
        pyvista_plot_class(
            uc_box=unitcell.box, positions=[system.positions[j] for j in uc_indices],
            types=[system.typeids[j] for j in uc_indices], 
            wyckoff_uc=[system.wyckoffs[j] for j in uc_indices],
            poly_vertices=[cells[ref_particle]],
            text="Voronoi Polyhedron"
        ).pyvista_plot_func()

    def _plot_unit_cell(self, system, unitcell, uc_indices):
        """Visualizes the entire unit cell and its constituent particles.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
        """
        outfilename = self.directory + "temp_files/" + self.shape_id + "_unit_cell.png"
        pyvista_plot_class(
            uc_box=unitcell.box, positions=[system.positions[j] for j in uc_indices],
            types=[system.typeids[j] for j in uc_indices], 
            wyckoff_uc=[system.wyckoffs[j] for j in uc_indices],
            poly_vertices=None,
            text="Unit Cell View",
            filename=outfilename
        ).pyvista_plot_func()

    def _hierarchy_bonds(self, system, unitcell, uc_indices, ref_particle, hierarchy_arr, hierarchy_coords, **kwargs):
        """Visualizes the shell connections (hierarchy bonds) between neighbors.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
            hierarchy_arr (List[List[int]]): Neighborhood shells grouping array.
            hierarchy_coords (np.ndarray): Relative coordinates of the hierarchy particles.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        color_types = ["red", "navy", "purple", "green", "blue", "cyan", "lime", "brown", "teal", "maroon", "aquamarine", "coral"]
        show_hierarchy_bonds = hierarchy_arr[:]
        hierarchy_coords_toPlot = np.array(hierarchy_coords)
        all_line_points, line_colors = [], []
        for i in range(len(show_hierarchy_bonds)):
            if len(show_hierarchy_bonds[i]) > 1:
                for j in range(len(show_hierarchy_bonds[i])-1):
                    all_line_points.append([(hierarchy_coords_toPlot[show_hierarchy_bonds[i][j+1]]+system.positions[ref_particle]).tolist(), (hierarchy_coords_toPlot[show_hierarchy_bonds[i][j]]+system.positions[ref_particle]).tolist()])
                    line_colors.append(color_types[len(show_hierarchy_bonds[i])])

        # Plot hierarchy
        outfilename = self.directory + "temp_files/" + self.shape_id + "_hierarchy_bonds.png"
        pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_indices],
                                            types=[system.typeids[j] for j in uc_indices], 
                                            wyckoff_uc=[system.wyckoffs[j] for j in uc_indices],
                                            poly_vertices=None,
                                            line_points=all_line_points,
                                            line_colors=line_colors, 
                                            extra_positions=self.particles_outside_uc, 
                                            extra_particles_types=self.extra_particles_types,
                                            extra_particles_wyckoffs=self.extra_particles_wyckoffs,
                                            center=None,
                                            direction=None,
                                            text="Hierarchy Bonds",
                                            filename=outfilename).pyvista_plot_func()

    def _replicated_asym_unit(self, system, unitcell, uc_indices, particle, rcut, **kwargs):
        """Calculates and visualizes the replicated asymmetric unit for symmetry checks.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
            particle (int): Reference particle ID.
            rcut (float): Active cutoff radius.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        logger.info("Generating replicated asymmetric unit visualization...")
        filename_repl=self.directory + "temp_files/" + self.shape_id + "_replicated_asym_unit.png" # Filename for visualization of replicated asymmetric unit alone
        filename_combined=self.directory + "temp_files/" + self.shape_id + "_combined.png" # Filename for combined visualization of asymmetric unit and replicated asymmetric unit
        # Asymmetric Unit Check and Visualization
        asym_obj = find_asymmetric_unit_class()
        asym_obj.compute(
            system=system, unitcell=unitcell, uc_particle_indices=uc_indices,
            particle=particle, rcut=rcut, rtol=self.rtol,
            poly_pg=self.poly_pg_all, space_group_number=self.space_group_number,
            sys_prep_pf=self.sys_prep_pf, 
            show=self.show, pg_tolerance=1e-1, site_symmetry_dict=self.site_symmetry_dict,
            equivalent_invariant_points=self.equivalent_invariant_points,
            atom_type_selection=self.atom_type_selection,
            type_id=self.type_id,
            filename_repl=filename_repl,
            filename_combined=filename_combined
        )

        self.chosen_particle_arr = asym_obj.chosen_particle_arr
        self.asym_unit_sites = np.array(asym_obj.asym_unit_sites)

    def _replicate_asymunit_get_basis_points(self, system, unitcell, uc_particle_indices, ref_particle,
                                            r, rtol, atom_type, space_group_number, show_basis=True, 
                                            replicated_asym_unit_show=True, **kwargs):
        """Helper to replicate the asymmetric unit and render the multi-particle basis points.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
            r (int): Rounding factor.
            rtol (float): Relative distance tolerance.
            atom_type (str): Target atom type.
            space_group_number (int): Crystal space group number.
            show_basis (bool): If True, plots and highlights the basis particles. Defaults to True.
            replicated_asym_unit_show (bool): If True, displays the replicated asymmetric unit. Defaults to True.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        max_len = max([round(np.linalg.norm(unitcell.box.wrap(system.positions[pid]-system.positions[ref_particle])), r) for pid in uc_particle_indices])
        # 3. Identify Point Group Invariant and Equivalent Particles for the reference point
        equivalent_invariant_points = _get_equivalent_invariant_points(system, unitcell, ref_particle, uc_particle_indices, atom_type, max_len, rtol, pg_tolerance=1e-1)

        poly_pg_all = self._detect_symmetry(ref_particle, max_len, system, unitcell, uc_particle_indices, atom_type, tolerance=1e-1, wyckoff_choice=False)
        self._replicated_asym_unit(system, unitcell, uc_particle_indices, ref_particle, self.current_rcut_wyckoff,
                                rtol=rtol,
                                poly_pg_all=poly_pg_all, space_group_number=space_group_number,
                                sys_prep_pf=self.sys_prep_pf, 
                                show=replicated_asym_unit_show, pg_tolerance=1e-1,
                                site_symmetry_dict=self.site_symmetry_dict,
                                equivalent_invariant_points=equivalent_invariant_points,
                                atom_type_selection=self.atom_type_selection,
                                type_id=self.type_id
        )
        
        if show_basis == "True":
            # Highlight the basis particles in the visualization
            outfilename = self.directory + "temp_files/" + self.shape_id + "_basis_points.png"
            pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_particle_indices],
                                special_points=np.array([system.positions[bp] for bp in self.chosen_particle_arr]),
                                types=[system.typeids[j] for j in uc_particle_indices], 
                                wyckoff_uc=[system.wyckoffs[j] for j in uc_particle_indices],
                                poly_vertices=None,
                                line_points=self.line_points,
                                text="Basis Points Highlighted",
                                filename=outfilename).pyvista_plot_func()

    def _asymmetricunit_visualization(self, system, unitcell, uc_indices, ref_particle, poly_vertices, **kwargs):
        """Generates and saves the asymmetric unit polyhedron rendering.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
            poly_vertices (List[np.ndarray]): Vertices of the polyhedron.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        logger.info("Generating Asymmetric unit visualization...")
        outfilename = self.outfilename
        pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_indices],
                            types=[system.typeids[j] for j in uc_indices], 
                            wyckoff_uc=[system.wyckoffs[j] for j in uc_indices],
                            poly_vertices=poly_vertices,
                            line_points=self.line_points,
                            extra_positions=self.particles_outside_uc, 
                            extra_particles_types=self.extra_particles_types,
                            extra_particles_wyckoffs=self.extra_particles_wyckoffs,
                            center=None,
                            direction=None,
                            text="Asymmetric unit Polyhedron",
                            filename=outfilename,
                            color=self.color).pyvista_plot_func()

    def _final_visualization(self, system, unitcell, uc_indices, ref_particle, poly_vertices, **kwargs):
        """Generates and saves the final truncated shape polyhedron rendering.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
            poly_vertices (List[np.ndarray]): Vertices of the polyhedron.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        logger.info("Generating final shape visualization...")
        for key, value in kwargs.items():
            setattr(self, key, value)

        outfilename = self.directory + "temp_files/" + self.shape_id + "_final_shape_atomtype_" + str(system.typeids[ref_particle]) + "_wyckoff_" + str(system.wyckoffs[ref_particle]) + ".png"
        pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_indices],
                            types=[system.typeids[j] for j in uc_indices], 
                            wyckoff_uc=[system.wyckoffs[j] for j in uc_indices],
                            poly_vertices=poly_vertices,
                            line_points=self.line_points,
                            text="Final Shape Polyhedron",
                            filename=None,
                            color=self.color).pyvista_plot_func()
        
    def _shape_visualization(self, system, unitcell, uc_indices, ref_particle, poly_vertices, **kwargs):
        """Generates and saves a clean, standalone shape rendering without background particles.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Indices of particles in the unit cell.
            ref_particle (int): Reference particle ID.
            poly_vertices (List[np.ndarray]): Vertices of the polyhedron.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        logger.info("Generating shape visualization...")
        for key, value in kwargs.items():
            setattr(self, key, value)

        outfilename = self.directory + "temp_files/" + self.shape_id + "_shape_visualization_atomtype_" + str(system.typeids[ref_particle]) + "_wyckoff_" + str(system.wyckoffs[ref_particle]) + ".png"
        pyvista_plot_class(uc_box=None, positions=None,
                            types=None, 
                            wyckoff_uc=None,
                            poly_vertices=poly_vertices,
                            line_points=None,
                            extra_positions=None, 
                            extra_particles_types=None,
                            extra_particles_wyckoffs=None,
                            center=None,
                            direction=None,
                            text="Shape Visualization",
                            filename=None).pyvista_plot_func()
