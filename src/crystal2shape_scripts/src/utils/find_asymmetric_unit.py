"""
Asymmetric Unit Finder Module
=============================

This module defines the :class:`find_asymmetric_unit_class` class, which identifies, computes,
and replicates the asymmetric unit of a crystal structure to evaluate packing coverage.
"""

import logging
import numpy as np
import rowan
import spglib
from scipy.spatial import ConvexHull, Delaunay
from pymatgen.core import Structure
import math

# Internal project imports
from ..utils.utils_func import (
    _get_particle_ids_coords_rcut, 
    _get_pointgroup_symmetry_equivalent_invariant_points, 
    _get_particles_uc
)
from ..visualization.pyvista_plot import pyvista_plot_class
from ..utils.make_hierarchy import Hierarchy

# Set up professional logging
logger = logging.getLogger(__name__)

class find_asymmetric_unit_class:
    """Identifies and replicates the asymmetric unit of a crystal structure.

    Determines the minimum set of particles (asymmetric unit) that can reproduce
    the full unit cell via crystallographic translation and point group symmetry operations.

    Attributes:
        asym_unit_vertices_arr (Optional[List[np.ndarray]]): Nested list of vertices for each
            constituent asymmetric unit polyhedron.
        chosen_particle_arr (Optional[List[int]]): Indices of system particles chosen to define
            the asymmetric unit nodes.
        decision (Optional[str]): Coverage status; 'Repl' if the replicated asymmetric units fully
            cover the unit cell, 'No-Repl' otherwise.
        covered_particles (Optional[List[int]]): List of unique particle IDs in the unit cell covered
            by the replicated asymmetric units.
        replicated_asym_units_vertices (Optional[List[np.ndarray]]): List of vertices of all replicated
            asymmetric units.
        asym_unit_vertices (Optional[List[np.ndarray]]): Reference list of replicated unit vertices.
    """

    def __init__(self):
        """Initializes the find_asymmetric_unit_class with empty placeholders."""
        # Result attributes maintained for main function compatibility
        self.asym_unit_vertices_arr = None
        self.chosen_particle_arr = None
        self.decision = None
        self.covered_particles = None
        self.replicated_asym_units_vertices = None
        self.asym_unit_vertices = None

    def subfunc_compute(self, system, unitcell, uc_particle_indices, 
                        particle, rcut, particle_arr_list, atom_type_selection=True, 
                        equiv_atom_list=None, rtol=0.01, atol=1.0, **kwargs):
        """Evaluates specific candidate combinations of equivalent particles for asymmetric unit viability.

        Args:
            system (gsd.hoomd.Snapshot): The global system snapshot.
            unitcell (freud.data.UnitCell): Unit cell object with lattice parameters.
            uc_particle_indices (List[int]): Indices of particles belonging to the primary unit cell.
            particle (int): The reference particle index for evaluation.
            rcut (float): Cutoff distance for neighbor analysis.
            particle_arr_list (List[List[int]]): Nested list of particle index combinations to evaluate.
            atom_type_selection (bool): If True, filters candidates to match the reference particle type. Defaults to True.
            equiv_atom_list (Optional[List[int]]): List of equivalent atom IDs. Defaults to None.
            rtol (float): Relative tolerance for coordinate closeness check. Defaults to 0.01.
            atol (float): Absolute tolerance for coordinate closeness check. Defaults to 1.0.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        asym_unit_vertices_arr_collected = []
        chosen_particle_arr_collected = []
        dec = "No"
        covered_particles = []

        for c in particle_arr_list:
            dummy_asym_unit_vertices_arr = []
            dummy_chosen_particle_arr = []
            
            for q in list(c):
                ref_point_dummy = system.positions[q]
                particle_ids_dummy, _ = _get_particle_ids_coords_rcut(
                    system, unitcell, q, uc_particle_indices, rcut, rtol=None
                )

                if atom_type_selection:
                    particle_ids_dummy = [j for j in particle_ids_dummy if system.typeids[j] == system.typeids[q]]

                if equiv_atom_list is not None:
                    particle_ids_dummy = [j for j in particle_ids_dummy if system.equiv_atoms[j] in equiv_atom_list]

                particle_ids_dummy.append(q)
                particle_coords_dummy = np.array([system.box.wrap(system.positions[j] - ref_point_dummy) for j in particle_ids_dummy])
                verts = particle_coords_dummy + unitcell.box.wrap(ref_point_dummy)

                dummy_asym_unit_vertices_arr.append(verts)
                dummy_chosen_particle_arr.append(q)

            cons_asym_unit_vertices_arr = dummy_asym_unit_vertices_arr[:]
            cons_chosen_particle_arr = dummy_chosen_particle_arr[:]

            try:
                self.replicate(
                    system=system, unitcell=unitcell, 
                    uc_particle_indices=uc_particle_indices,
                    asym_unit_vertices_arr=cons_asym_unit_vertices_arr, 
                    equivalent_particles_chosen=cons_chosen_particle_arr,
                    r_show=False
                )
                
                asym_unit_vertices_flat = [u for i in self.asym_unit_vertices for u in i]
                
                # Filter unit cell particles by type for coverage check
                uc_particle_indices_type = [
                    idx for idx in uc_particle_indices 
                    if not atom_type_selection or system.typeids[idx] == system.typeids[particle]
                ]

                covered_particles = [
                    idx for idx in uc_particle_indices_type 
                    if any(np.allclose(system.positions[idx], vert, atol=rtol) for vert in asym_unit_vertices_flat)
                ]
                        
                if len(covered_particles) == len(uc_particle_indices_type):
                    dec = 'Repl'
                    asym_unit_vertices_arr_collected = cons_asym_unit_vertices_arr[:]
                    chosen_particle_arr_collected = cons_chosen_particle_arr[:]
                    break
                else:
                    dec = 'No-Repl'
                
            except Exception as e:
                logger.error(f"Replication error in subfunc_compute: {e}")
                dec = "None"

        self.replicated_asym_units_vertices = self.asym_unit_vertices
        self.asym_unit_vertices_arr_collected = asym_unit_vertices_arr_collected
        self.chosen_particle_arr_collected = chosen_particle_arr_collected
        self.decision = dec
        self.covered_particles = covered_particles

    def compute(self, system, unitcell, uc_particle_indices, particle, rcut,
                rtol=0.01, atol=1.0, poly_pg="C1",
                show=True, **kwargs):
        """Finds the asymmetric unit nodes and vertices around the reference particle.

        Determines the nodes, creates their local neighbor polyhedra, maps unit cell coverage,
        and saves intermediate representations or decisions.

        Args:
            system (gsd.hoomd.Snapshot): The global system snapshot.
            unitcell (freud.data.UnitCell): Unit cell object.
            uc_particle_indices (List[int]): Indices of particles in the unit cell.
            particle (int): Reference particle index.
            rcut (float): Cutoff radius for local neighbor analysis.
            rtol (float): Relative coordinate tolerance. Defaults to 0.01.
            atol (float): Absolute coordinate tolerance. Defaults to 1.0.
            poly_pg (str): Target point group symmetry of the environment. Defaults to "C1".
            show (bool): If True, plots and renders the coverage visualization. Defaults to True.
            **kwargs: Extra parameters dynamically set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        rd_factor = int(math.log10(1/rtol)) if rtol > 0 else 6

        # Get the nodes where the particles would start growing
        considered_particles = self._get_nodes_sites(system, unitcell, uc_particle_indices, particle, 
                                                     rcut, rtol, atom_type_selection=self.atom_type_selection, 
                                                     type_id=self.type_id)

        self.asym_unit_sites = np.array([system.positions[j] for j in considered_particles])
        considered_particles, _ = _get_particles_uc(system, unitcell, uc_particle_indices, considered_particles)

        # Construct Asymmetric Unit Vertices for all considered particles and check coverage
        asymunit_vertices, asymunit_centerids, point_ids, poit_coords = [], [], [], []
        for p_idx in considered_particles:
            p_ids_sel, _ = _get_particle_ids_coords_rcut(system, unitcell, p_idx, uc_particle_indices, rcut, rtol=rtol)
            if self.atom_type_selection == "True": target_typeid = self.type_id
            else: target_typeid = None

            if target_typeid:
                p_ids_sel = [j for j in p_ids_sel if system.typeids[j] == target_typeid]
            p_ids_sel.append(p_idx)

            p_ids_coords = np.array([system.box.wrap(system.positions[j] - system.positions[p_idx]) for j in p_ids_sel])
            verts = p_ids_coords + system.box.wrap(system.positions[p_idx])
            asymunit_vertices.append(verts)

            ids_in_uc, _ = _get_particles_uc(system, unitcell, uc_particle_indices, p_ids_sel)
            asymunit_centerids.append(ids_in_uc)
            point_ids.append(p_ids_sel[:-1])
            poit_coords.append(p_ids_coords[:-1])

        # Finalize Results
        self.asym_unit_vertices_arr = asymunit_vertices
        self.chosen_particle_arr = considered_particles
        self.combined_vertices = poit_coords
        
        uc_particle_indices_type = [idx for idx in uc_particle_indices 
            if system.typeids[idx] == self.type_id]

        uc_particle_indices_type.sort()
        
        self.covered_particles = list(set([j for sublist in asymunit_centerids for j in sublist]))
        self.covered_particles = [j for j in self.covered_particles if system.typeids[j] == self.type_id]
        self.covered_particles.sort()

        self.decision = 'Repl' if all(t in self.covered_particles for t in uc_particle_indices_type)==True else 'No-Repl'
        if show == "True":
            self._visualize_replicated_asym(system, unitcell, uc_particle_indices, particle)

    def _get_nodes_sites(self, system, unitcell, uc_particle_indices, particle, rcut, rtol, atom_type_selection=True, type_id=None):
        """Retrieves space group symmetry equivalent node sites in the unit cell.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Particle indices in the unit cell.
            particle (int): Reference particle index.
            rcut (float): Cutoff radius.
            rtol (float): Relative distance tolerance.
            atom_type_selection (bool): If True, filters by atom type. Defaults to True.
            type_id (Optional[str]): Specific atom type ID to filter by. Defaults to None.

        Returns:
            List[int]: Space group equivalent atom indices.
        """
        # Get Lattice vectors
        uc_lattice_vectors = np.transpose(unitcell.box.to_matrix())

        # Symmetry dataset retrieval (Spglib)
        basis_types = np.unique([system.typeids[i] for i in uc_particle_indices]).tolist()
        numbers = [basis_types.index(system.typeids[t]) for t in uc_particle_indices]
        lattice = [t.tolist() for t in uc_lattice_vectors]
        cell = (lattice, unitcell.fractional_positions, numbers)
        symm_dict = spglib.get_symmetry_dataset(cell, symprec=0.1, angle_tolerance=5.0)
        equiv_atoms_sg = symm_dict.equivalent_atoms

        # Find particles within rcut
        point_ids, _ = _get_particle_ids_coords_rcut(
                                system, unitcell, particle, uc_particle_indices, rcut, rtol=rtol
                            )
        if atom_type_selection == "True": target_typeid = type_id
        else: target_typeid = None
        if target_typeid is not None:
            point_ids = [j for j in point_ids if system.typeids[j] == target_typeid]
        point_ids.append(particle)

        point_ids, _ = _get_particles_uc(system, unitcell, uc_particle_indices, point_ids)
        point_ids = list(set(point_ids))
        
        # Space group equivalent sites in the unit cell
        equiv_atomsids_sg = [uc_particle_indices[j] for j in range(len(uc_particle_indices)) if equiv_atoms_sg[j] == equiv_atoms_sg[uc_particle_indices.index(particle)]]
        node_sites = equiv_atomsids_sg[:]

        return node_sites

    def _visualize_replicated_asym(self, system, unitcell, uc_particle_indices, particle):
        """Visualizes the replicated asymmetric unit polyhedra and combined polyhedron.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Particle indices in the unit cell.
            particle (int): Reference particle index.
        """
        posi_uc = [system.positions[t] for t in uc_particle_indices]
        type_uc = [system.typeids[t] for t in uc_particle_indices]
        wyckoff_uc = [system.wyckoffs[t] for t in uc_particle_indices]
        
        outfilename = getattr(self, 'filename_repl', None)
        pyvista_plot_obj = pyvista_plot_class(
            uc_box=unitcell.box, positions=posi_uc, types=type_uc, wyckoff_uc=wyckoff_uc,
            special_points=np.array([system.positions[bp] for bp in self.chosen_particle_arr]),
            poly_vertices=self.asym_unit_vertices_arr, line_points=None,
            text="Replicated Asymmetric unit polyhedra",
            filename=None,
            color="steelblue"
        )
        pyvista_plot_obj.pyvista_plot_func()

        # Visualize combined polyhedron at one position
        posi_uc = [system.positions[t] for t in uc_particle_indices]
        type_uc = [system.typeids[t] for t in uc_particle_indices]
        wyckoff_uc = [system.wyckoffs[t] for t in uc_particle_indices]

        trans_shifted_combined_vertices = [(np.array(verts)+system.positions[particle]) for verts in self.combined_vertices]
        outfilename = getattr(self, 'filename_combined', None)
        pyvista_plot_obj = pyvista_plot_class(
            uc_box=unitcell.box, positions=posi_uc, types=type_uc, wyckoff_uc=wyckoff_uc,
            special_points=None,
            poly_vertices=trans_shifted_combined_vertices, line_points=None,
            text="Combined polyhedron",
            filename=None,
            color="steelblue"
        )
        pyvista_plot_obj.pyvista_plot_func()

    def replicate(self, system, unitcell, uc_particle_indices, asym_unit_vertices_arr, 
                  equivalent_particles_chosen, r_show=True):
        """Replicates asymmetric unit vertices using Bravais lattice translations to check unit cell coverage.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Particle indices in the unit cell.
            asym_unit_vertices_arr (List[np.ndarray]): Vertices of the asymmetric unit polyhedra.
            equivalent_particles_chosen (List[int]): Indices of the chosen nodes.
            r_show (bool): If True, invokes the visualization helper. Defaults to True.
        """
        lattice_vectors = unitcell.lattice_vectors
        asym_unit_vertices_list = []

        for i, pv_orig in enumerate(asym_unit_vertices_arr):
            pv = np.vstack([pv_orig, system.positions[equivalent_particles_chosen[i]]])
            
            translations = [np.zeros(3)]
            for v in lattice_vectors:
                translations.extend([t + v for t in translations] + [t - v for t in translations])
            
            l0, l1, l2 = lattice_vectors
            extra_combos = [
                l0+l1+l2, -l0-l1-l2, l0+l1-l2, l0-l1+l2, 
                -l0+l1+l2, -l0-l1+l2, -l0+l1-l2, l0-l1-l2
            ]
            translations.extend(extra_combos)

            for trans in translations:
                asym_unit_vertices_list.append(pv + trans)

        self.asym_unit_vertices = asym_unit_vertices_list

        if r_show:
            self._visualize_replicated_asym(system, unitcell, uc_particle_indices)

    def _detect_symmetry(self, particle, rcut, system, unitcell, uc_indices, 
                         atom_type, wyckoff_type, tolerance, rtol):
        """Determines point group symmetry using Pymatgen PointGroupAnalyzer.

        Args:
            particle (int): Reference particle ID.
            rcut (float): Cutoff radius.
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Particle indices in the unit cell.
            atom_type (Optional[str]): Atom type filter.
            wyckoff_type (Optional[str]): Wyckoff position filter.
            tolerance (float): Symmetry matching tolerance.
            rtol (float): Relative coordinate tolerance.

        Returns:
            Tuple[Optional[str], Optional[Any]]: Schoenflies symbol and list of symmetry operations, or (None, None).
        """
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        from skspatial.objects import Points

        p_ids, p_coords = _get_particle_ids_coords_rcut(system, unitcell, particle, uc_indices, rcut, rtol=rtol)
        p_ids.append(particle)
        p_coords.append(np.array([0, 0, 0]))
        
        if atom_type:
            mask = [system.typeids[pid] == atom_type for pid in p_ids]
            p_coords = [p_coords[i] for i, val in enumerate(mask) if val]
            p_ids = [p_ids[i] for i, val in enumerate(mask) if val]

        if wyckoff_type:
            mask = [system.wyckoffs[pid] == wyckoff_type for pid in p_ids]
            p_coords = [p_coords[i] for i, val in enumerate(mask) if val]
            p_ids = [p_ids[i] for i, val in enumerate(mask) if val]

        species_labels = [system.typeids[pid] for pid in p_ids]

        if len(p_coords) < 4 or Points(p_coords).are_coplanar(tol=1e-2):
            return None, None
        
        try:
            mol = Molecule(species=species_labels, coords=p_coords)
            return PointGroupAnalyzer(mol, tolerance=tolerance).sch_symbol, PointGroupAnalyzer(mol, tolerance=tolerance).get_symmetry_operations()
        except Exception:
            return None, None
        
    def _get_pointgroup(self, pg_operations_arr):
        """Analyzes point group symmetry operations to identify rotation axis counts.

        Also renders the detected symmetry axes using pyvista.

        Args:
            pg_operations_arr (List[np.ndarray]): Symmetry rotation matrices.

        Returns:
            List[int]: Count of C6, C4, C3, C2, and other rotation operations.
        """
        symmetry_list = [0 for _ in range(5)]
        matrix_track = []
        for op in pg_operations_arr:
            rot = op[:]
            q = rowan.from_matrix(rot, require_orthogonal=False)
            axis_angle = rowan.to_axis_angle(q)
            axis, angle = axis_angle[0][0], round(np.rad2deg(axis_angle[1][0]), 2)
            if angle == 180.0:
                symmetry_list[3] += 1
            elif angle == 120.0 or angle == 240.0:
                symmetry_list[2] += 1
            elif angle == 90.0 or angle == 270.0:
                symmetry_list[1] += 1
            elif angle == 60.0 or angle == 300.0:
                symmetry_list[0] += 1
            else:
                symmetry_list[4] += 1

            if angle == 180.0:
                matrix_track.append(rot)
        
        print(f"Point group from symmetry operations: {symmetry_list}")
        self.symmetry_list = symmetry_list

        symm_axes, symm_angles = [], []
        for mt in matrix_track:
            quat = rowan.from_matrix(mt, require_orthogonal=False)
            axis_angle = rowan.to_axis_angle(quat)
            if round(float(np.rad2deg(axis_angle[1][0])), 2) == 120.0 :
                if np.round(axis_angle[0][0], decimals=2).tolist() not in symm_axes and np.round(axis_angle[0][0], decimals=2).tolist() != [0.0, 0.0, 0.0]:
                    symm_axes.append(np.round(axis_angle[0][0], decimals=2).tolist())
                    symm_angles.append(np.round(axis_angle[1][0], decimals=2))
        
        for axis in symm_axes.copy():
            neg_axis = [-x for x in axis]
            if neg_axis not in symm_axes:
                symm_axes.append(neg_axis)

        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        mol = Molecule(species=["C" for _ in range(len(symm_axes))], coords=symm_axes)
        pg_symm = PointGroupAnalyzer(mol, tolerance=1e-1).sch_symbol

        show_poly = True
        if show_poly is True:
            pyvista_plot_class(uc_box=None, positions=None,
                                types=None, 
                                wyckoff_uc=None,
                                poly_vertices=[np.array(symm_axes)],
                                line_points=None,
                                extra_positions=None, 
                                extra_particles_types=None,
                                extra_particles_wyckoffs=None,
                                center=None,
                                direction=None,
                                text="Axes Visualization",
                                filename=None).pyvista_plot_func()

        return symmetry_list