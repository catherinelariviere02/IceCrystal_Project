"""
Reference Particle and Cutoff Distance Calculator
=================================================

This module defines the :class:`RefParticleCalculator` class, which calculates the reference atom IDs
and optimal cutoff distances (rcut) to maintain Wyckoff position conservation and point group symmetry
in crystal structures.
"""

import os
import json
import logging
import math
from collections import Counter
from scipy.spatial import ConvexHull
from typing import List, Dict, Tuple, Any, Optional, Set

import numpy as np
import coxeter
import freud
import spglib
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from skspatial.objects import Points

# Internal project imports
from ..utils.utils_func import (
    _get_particle_ids_coords_rcut, 
    _get_particles_uc,
    _get_equivalent_invariant_points,
    _moment_of_inertia, _covariance_matrix,
    _get_pointgroup_neighbors_of_neighbors
)
from ..utils.make_hierarchy import Hierarchy
from ..utils.get_pointgroup_polyhedron_rcut import (
    _get_pointgroup_polyhedron_rcut, 
    _get_max_distances
)
from ..utils.find_asymmetric_unit import find_asymmetric_unit_class

# Configure logging
logger = logging.getLogger(__name__)

class RefParticleCalculator:
    """Calculates reference atom IDs and optimal cutoff distances (rcut).

    Maintains Wyckoff position conservation and symmetry in crystal structures by analyzing
    local coordinate environments and validating point group symmetry criteria.

    Attributes:
        pg_order_path (str): Path to the JSON file containing point group orders.
        pg_order_data (Dict[str, int]): Loaded point group symbol-to-order map.
        ref_particle_arr (List[List[int]]): Computed reference particle IDs per basis atom type.
        rcut_arr (List[List[Optional[float]]]): Computed optimal cutoff distances per basis type.
        rcut_all_arr (List[List[Optional[float]]]): Maximum cutoff distances checked per basis type.
        w_keys_type (List[List[str]]): Wyckoff symbols present in the system per atom type.
        site_symmetry_dict (Dict[str, str]): Map from Wyckoff letters to site symmetry symbols.
        unique_equiv_indices (List[int]): Sorted list of unique equivalent atom indices.
    """

    def __init__(self, pg_order_path: str = "PointGroupOrder.json"):
        """Initializes the RefParticleCalculator.

        Args:
            pg_order_path (str): Path to the JSON file containing point group orders.
                Defaults to "PointGroupOrder.json".
        """
        self.pg_order_path = pg_order_path
        self.pg_order_data = self._load_pg_order()
        
        # Result Storage
        self.ref_particle_arr: List[List[int]] = []
        self.rcut_arr: List[List[Optional[float]]] = []
        self.rcut_all_arr: List[List[Optional[float]]] = []
        
        # Metadata
        self.w_keys_type: List[List[str]] = []
        self.site_symmetry_dict: Dict[str, str] = {}
        self.unique_equiv_indices: List[int] = []

    def _load_pg_order(self) -> Dict[str, int]:
        """Loads point group order data from a JSON file.

        Returns:
            Dict[str, int]: Map from point group symbols (e.g., 'C1', 'D2h') to their order (multiplicity).
        """
        if not os.path.exists(self.pg_order_path):
            logger.warning(f"{self.pg_order_path} not found. Using empty defaults.")
            return {}
        with open(self.pg_order_path, "r") as f:
            return json.load(f)

    def compute(self, unitcell: Any, system: Any, uc_particle_indices: List[int], 
                basis_type: List[str], rcut: float, rd_factor: int = 2, 
                rtol: float = 0.01, **kwargs) -> 'RefParticleCalculator':
        """Calculates reference particles and optimal cutoff distances for the crystal system.

        Args:
            unitcell (freud.data.UnitCell): The freud unit cell object.
            system (gsd.hoomd.Snapshot): The hoomd/gsd system snapshot.
            uc_particle_indices (List[int]): Indices of particles belonging to the reference unit cell.
            basis_type (List[str]): Active basis atom types (e.g., ['Ga']).
            rcut (float): Starting cutoff distance for coordinate environment analysis.
            rd_factor (int): Precision scaling factor for rounding tolerances. Defaults to 2.
            rtol (float): Relative tolerance for floating-point comparisons. Defaults to 0.01.
            **kwargs: Additional parameters to inject as class attributes.

        Returns:
            RefParticleCalculator: Self instance with computed result attributes.
        """
        # Inject additional params
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.unique_wyckoff_atom_type = [np.unique([str(system.wyckoffs[i]) for i in uc_particle_indices if system.typeids[i] == b_type]).tolist() for b_type in basis_type]
        self._initialize_wyckoff_info(system, uc_particle_indices, basis_type)
        self.uc_wyckoff_ratio = [np.array(wyckoff)/math.gcd(*wyckoff) for wyckoff in self.modified_symm_cntr_wyckoff_info]

        if len(uc_particle_indices) > 1:
            self._compute_multi_particle(
                unitcell, system, uc_particle_indices, basis_type, 
                rd_factor, rtol, **kwargs
            )
        else:
            self._compute_single_particle(unitcell, uc_particle_indices)

        return self

    def _initialize_wyckoff_info(self, system: Any, particles: List[int], basis_type: List[str]):
        """Collects Wyckoff positions and ratios per atom type in the unit cell.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            particles (List[int]): Particle indices in the unit cell.
            basis_type (List[str]): Active basis atom types.
        """
        wyckoff_info_per_type = [[] for _ in range(len(basis_type))]
        for j, b_type in enumerate(basis_type):
            for p_idx in particles:
                if system.typeids[p_idx] == b_type:
                    wyckoff_info_per_type[j].append(system.wyckoffs[p_idx])

        self.w_keys_type = []
        self.modified_symm_cntr_wyckoff_info = []

        counter = 0
        for info in wyckoff_info_per_type:
            vals, counts = np.unique(info, return_counts=True)
            sorted_values_index = [self.unique_wyckoff_atom_type[counter].index(str(v)) for v in vals]
            sorted_counts = [counts[i] for i in sorted_values_index]
            self.w_keys_type.append([vals[i] for i in sorted_values_index])
            self.modified_symm_cntr_wyckoff_info.append(sorted_counts)
            counter += 1

    def _compute_multi_particle(self, unitcell: Any, system: Any, uc_indices: List[int], 
                                basis_type: List[str], rd_factor: int, rtol: float, 
                                **kwargs):
        """Core symmetry analysis logic for systems with multiple particles per unit cell.

        Iterates through equivalent atom lists and pairs of types to find optimal cutoff distances
        that conserve the local site symmetries.

        Args:
            unitcell (freud.data.UnitCell): The unit cell object.
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Particle indices in the unit cell.
            basis_type (List[str]): Active basis atom types.
            rd_factor (int): Precision scaling factor for rounding.
            rtol (float): Relative tolerance for floating-point checks.
            **kwargs: Additional keyword arguments.
        """
        self.unique_equiv_indices = np.unique([int(system.equiv_atoms[t]) for t in uc_indices]).tolist()
        unique_equiv_atoms_list = [
            np.unique([int(system.equiv_atoms[i]) for i in uc_indices if system.typeids[i] == b_type]).tolist() 
            for b_type in basis_type
        ]
        
        final_particles, final_rcuts, final_rcut_alls = [[] for _ in basis_type], [[] for _ in basis_type], [[] for _ in basis_type]

        # 1. Pre-calculate Point Group and Max Distance Dictionaries
        pg_dic, max_dist_dic = self._precompute_symmetry_maps(
            unitcell, system, uc_indices, basis_type, unique_equiv_atoms_list, rd_factor, rtol, **kwargs
        )

        # Get equiv_atom counts per type for the particles in the unit cell to use for weighted symmetry order calculations
        equiv_atoms_count_per_type, symm_cntr_per_type = self._get_equiv_atom_counts_per_type(system, uc_indices, basis_type) 

        # Get the pointgroup symmetry order for each equivalent id from pg_dic
        pointgroup_order_equiv_id = []
        for t in range(len(basis_type)):
            equiv_id_dict = {}
            for eq in unique_equiv_atoms_list[t]:
                target_type = "all" if self.atom_type_selection == "False" else system.typeids[uc_indices[eq]]
                dist_key = f"{eq}_{target_type}"
                max_d = max_dist_dic.get(dist_key, 0.0)
                pg_key = f"{eq}_{round(max_d, rd_factor)}"
                equiv_id_dict[eq] = pg_dic.get(pg_key, {"pg_sym": "C1", "pg_order": 0})["pg_order"]
            pointgroup_order_equiv_id.append(equiv_id_dict)
        print(f"[INFO] Point group order per equivalent ID: {pointgroup_order_equiv_id}")

        total_order_uc = sum([(list(equiv_atoms_count_per_type[type].values())[val] * list(pointgroup_order_equiv_id[type].values())[val]) for type in range(len(equiv_atoms_count_per_type)) for val in range(len(equiv_atoms_count_per_type[type]))])
        print(f"[INFO] Total symmetry order of the unit cell based on equivalent atoms and their point group orders: {total_order_uc}")

        # 2. Calculate symmetry order per Wyckoff site in the unit cell based on tracked unique local environments
        unique_wyckoff_type_list = [np.unique(self.w_keys_type[bt]).tolist() for bt in range(len(basis_type))]
        symmetry_order_type_uc = []
        for wk in range(len(unique_wyckoff_type_list)):
            symmetry_order_wyckoff = [0 for _ in unique_wyckoff_type_list[wk]]
            neighbor_env_track = []
            for (i, w) in enumerate(unique_wyckoff_type_list[wk]):
                particle_ids_wyckoff = [i for i in uc_indices if system.wyckoffs[i] == w and system.typeids[i] == basis_type[wk]]
                particle_ids_wyckoff_unique_env = []
                for pid in particle_ids_wyckoff:
                    if system.local_env_cluster_ids[pid] not in neighbor_env_track:
                        neighbor_env_track.append(system.local_env_cluster_ids[pid])
                        particle_ids_wyckoff_unique_env.append(pid)
                pg_order_particles = []
                for pid in particle_ids_wyckoff_unique_env:
                    eq_atom = uc_indices[system.equiv_atoms[pid]]
                    target_type = "all" if self.atom_type_selection == "False" else system.typeids[pid]
                    dist_key = f"{eq_atom}_{target_type}"
                    max_d = max_dist_dic.get(dist_key, 0.0)
                    pg_key = f"{eq_atom}_{round(max_d, rd_factor)}"
                    pg_order_particles.append(pg_dic.get(pg_key, {"pg_sym": "C1", "pg_order": 0})["pg_order"])
                symmetry_order_wyckoff[i] = int(np.sum(np.array(pg_order_particles)))

            symmetry_order_type_uc.append(symmetry_order_wyckoff)

        # 3. Iterate through atom pairs to find optimal rcuts
        for i, atom_type_ref in enumerate(basis_type):
            for j, atom_type_sel in enumerate(basis_type):
                if True:
                    print("\n--------------------------------------")
                    print(f"[INFO] Optimizing environment: {atom_type_ref} <-> {atom_type_sel}")
                    
                    max_order_uc = np.sum([pointgroup_order_equiv_id[i][eq] * len([k for k in uc_indices if system.equiv_atoms[k] == eq and system.typeids[k] == basis_type[i]]) for eq in unique_equiv_atoms_list[i]])
                    
                    results = self._search_optimal_rcut(
                        i, atom_type_ref, j, atom_type_sel, unitcell, system, 
                        uc_indices, self.unique_equiv_indices, rd_factor, rtol,
                        max_dist_dic, basis_type, pg_dic, symmetry_order_type_uc,
                        unique_wyckoff_type_list, max_order_uc, **kwargs
                    )
                    
                    final_particles[i].extend(results['particles'])
                    final_rcuts[i].extend(results['rcuts'])
                    final_rcut_alls[i].extend(results['rcut_alls'])

        self.ref_particle_arr = final_particles
        self.rcut_arr = final_rcuts
        self.rcut_all_arr = final_rcut_alls
        self.pg_map = pg_dic
        self.dist_map = max_dist_dic

    def _precompute_symmetry_maps(self, unitcell, system, uc_indices, basis_type, 
                                  unique_equiv_atoms_list, rd_factor, rtol, **kwargs) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """Builds cached maps for point groups and distances to avoid redundant calculations.

        Args:
            unitcell (freud.data.UnitCell): The unit cell object.
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Particle indices in the unit cell.
            basis_type (List[str]): Active basis atom types.
            unique_equiv_atoms_list (List[List[int]]): Nested list of equivalent atom indices per basis type.
            rd_factor (int): Precision scaling factor.
            rtol (float): Relative tolerance.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]: A tuple containing:
                - A dictionary mapping particle ID and distance keys to point group symmetry info.
                - A dictionary mapping particle ID and target type keys to maximum distances.
        """
        pg_map, dist_map = {}, {}
        for i, b_type_i in enumerate(basis_type):
            particle_ids_type_i = [t for t in uc_indices if system.typeids[t] == b_type_i]
            for j, b_type_j in enumerate(basis_type):
                particle_ids_type_j = [t for t in uc_indices if system.typeids[t] == b_type_j]
                if True: 
                    m_dist = _get_max_distances(
                        particle_ids_type_i, particle_ids_type_j, system, unitcell, uc_indices, rd_factor, 
                        basis_type, atom_type_selection=getattr(self, 'atom_type_selection', "True"),
                        d_tol=self.d_tol)
                    
                    p_group = _get_pointgroup_polyhedron_rcut(system,
                        unitcell,
                        uc_indices,
                        particle_ids_type_i, 
                        particle_ids_type_j, m_dist, 
                        rtol=rtol, basis_type=basis_type, 
                        atom_type_selection=getattr(self, 'atom_type_selection', "True")
                    )

                    dist_map.update(m_dist)
                    pg_map.update(p_group)
        
        return pg_map, dist_map

    def _search_optimal_rcut(self, idx_i, type_i, idx_j, type_j, unitcell, system, 
                             uc_indices, unique_equiv_indices, rd_factor, rtol,
                             dist_map, basis_type, pg_map, symmetry_order_type_uc,
                             unique_wyckoff_type_list, max_order_uc,
                             **kwargs) -> Dict[str, List]:
        """Core iterative search for the minimum valid cutoff distance (rcut) for an atom pair.

        Args:
            idx_i (int): Index of the reference basis atom type.
            type_i (str): Name of the reference basis atom type.
            idx_j (int): Index of the selected basis atom type.
            type_j (str): Name of the selected basis atom type.
            unitcell (freud.data.UnitCell): The unit cell object.
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Particle indices in the unit cell.
            unique_equiv_indices (List[int]): Unique equivalent atom indices.
            rd_factor (int): Precision scaling factor.
            rtol (float): Relative tolerance.
            dist_map (Dict[str, float]): Cached maximum distances map.
            basis_type (List[str]): Active basis atom types.
            pg_map (Dict[str, Dict[str, Any]]): Cached point group symmetry map.
            symmetry_order_type_uc (List[List[int]]): Symmetry orders of Wyckoff positions in the unit cell.
            unique_wyckoff_type_list (List[List[str]]): List of unique Wyckoff symbols per basis type.
            max_order_uc (int): Maximum symmetry order of the unit cell.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, List]: Dictionary containing list of particles, optimal rcuts, and max distances.
        """
        type_rcuts, type_particles, type_rcut_alls = [], [], []
        target_typeid = "all" if getattr(self, 'atom_type_selection', "True") == "False" else type_j

        required_condition = "not_satisfied"
        # Iterate over Wyckoff sites for the reference atom type
        for w_key in self.w_keys_type[idx_i]:
            target_ids = [i for i in uc_indices if system.wyckoffs[i] == w_key and system.typeids[i] == type_i]
            unique_equiv = np.unique([system.equiv_atoms[i] for i in target_ids])
            particle_samples = [uc_indices[eq] for eq in unique_equiv]

            if required_condition == "not_satisfied":
                for particle in particle_samples:
                    max_len = dist_map.get(f"{system.equiv_atoms[particle]}_{target_typeid}", 0.0)
                    candidates = self._get_rcut_candidates(particle, max_len, system, rd_factor)

                    # Symmetry validation
                    poly_pg_all, _ = self._detect_symmetry(particle, round(max_len, rd_factor), system, unitcell, uc_indices, type_j, None, self.pg_tolerance, rtol)
                    if type_i == type_j:
                        poly_pg_site, _ = self._detect_symmetry(particle, round(max_len, rd_factor), system, unitcell, uc_indices, type_j, system.wyckoffs[particle], self.pg_tolerance, rtol)
                    else:
                        poly_pg_site, _ = self._detect_symmetry(particle, round(max_len, rd_factor), system, unitcell, uc_indices, type_j, None, self.pg_tolerance, rtol)

                    crystal_pg_order = self.pg_order_data[self.pg_order_data.get(unitcell.crystal_pointgroup, 1)]
                    required_order = crystal_pg_order 

                    print(f"Required order for partticle ID {particle}: {required_order}")
                    print(f"With respect to equivalent particle: {system.equiv_atoms[particle]} -->  PG considering Wyckoff site {system.wyckoffs[particle]}: {poly_pg_site} | PG of Info poly: {poly_pg_all}")
                    print(" ")

                    if poly_pg_site == poly_pg_all:
                        found_valid = False
                        if candidates:
                            print(f"Evaluating candidate cutoffs for particle ID {particle} (Wyckoff: {system.wyckoffs[particle]}, Equiv: {system.equiv_atoms[particle]}, max_d_ref: {max_len:.3f}):")
                            print(f"{'rcut':>8} | {'rcut/max_d':>10} | {'Atom Counts':<20} | {'Orien Counts':<20} | {'Symm Order':>10} | {'PG Rcut':^10} | {'PG Ident':^10} | {'MoI Diff':>10}")
                            print("-" * 110)
                        for r_cand in candidates:
                            if self._validate_rcut_candidate(particle, r_cand, required_order, poly_pg_all, 
                                                            system, unitcell, uc_indices, basis_type, 
                                                            pg_map, dist_map, type_j, rtol, max_order_uc,
                                                            required_condition, **kwargs):
                                type_rcuts.append(r_cand)
                                type_particles.append(particle)
                                type_rcut_alls.append(round(max_len, rd_factor))
                                found_valid = True
                                required_condition = self.required_condition 

                                break

                            else:
                                pass  # Candidate failed validation, try next one

                        if not found_valid:
                            logger.warning(f"No valid rcut for particle {particle} at {w_key}. Using max_len.")
                            type_rcuts.append(max_len)
                            type_particles.append(particle)
                            type_rcut_alls.append(max_len)

                    print(" ")
            else:
                pass

        return {"particles": type_particles, "rcuts": type_rcuts, "rcut_alls": type_rcut_alls}

    def _validate_rcut_candidate(self, particle, rcut, req_order, target_pg, 
                                system, unitcell, uc_indices, basis_type, 
                                pg_map, dist_map, type_i, rtol, max_order_uc, 
                                required_condition, **kwargs) -> bool:
        """Validates if a specific rcut candidate meets symmetry and geometric requirements.

        Args:
            particle (int): Reference particle index.
            rcut (float): Cutoff distance candidate to validate.
            req_order (int): Required symmetry order.
            target_pg (str): Target point group symbol.
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Particle indices in the unit cell.
            basis_type (List[str]): Active basis atom types.
            pg_map (Dict[str, Dict[str, Any]]): Cached point group symmetry map.
            dist_map (Dict[str, float]): Cached maximum distances map.
            type_i (str): Selected atom type name.
            rtol (float): Relative tolerance.
            max_order_uc (int): Maximum symmetry order of the unit cell.
            required_condition (str): Current status of required conditions.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the candidate satisfies all symmetry, point group, and moment of inertia checks.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        rd_factor = int(math.log10(1 / rtol))
        max_len_ref = dist_map.get(f"{system.equiv_atoms[particle]}_{'all' if getattr(self, 'atom_type_selection', 'True') == 'False' else type_i}", 0.0)
        uc_lenscales = [round(float(np.linalg.norm(system.box.wrap(system.positions[j] - system.positions[particle]))), rd_factor) for j in range(len(system.positions)) if round(float(np.linalg.norm(system.box.wrap(system.positions[j] - system.positions[particle]))), rd_factor) <= round(max_len_ref, rd_factor)]
        max_len_ref = max(uc_lenscales)

        p_ids, p_coords = _get_particle_ids_coords_rcut(system, unitcell, particle, uc_indices, max_len_ref, rtol=rtol)
        p_ids.append(particle)

        ref_particle_sitesymmetry_order = pg_map.get(f"{system.equiv_atoms[particle]}_{round(dist_map.get(f'{system.equiv_atoms[particle]}_{type_i}', 0.0), rd_factor)}", {"pg_sym": "C1", "pg_order": 0})["pg_order"]

        equiv_atomsids_sg, equivalent_invariant_points = self._get_sg_symmmetry_equiv_pg_symmetry_equiv_inv_points(system, unitcell, particle, uc_indices, type_i, max_len_ref, rd_factor, rtol)
        pg_symm_inv_points, _ = _get_particles_uc(system, unitcell, uc_indices, equivalent_invariant_points[1])
        invariant_points_ref_particles = [p for p in pg_symm_inv_points if system.typeids[p] == type_i and system.wyckoffs[p] == system.wyckoffs[particle]]

        # 1. Total Symmetry Order Check
        self._track_particles_within_rcut(particle, rcut, system, unitcell, uc_indices, self.unique_equiv_indices, basis_type, 
                                          pg_map, dist_map, type_i, rtol, self.atom_type_selection, self.pg_tolerance)
        
        wyckoff_ratio_atom_type = [np.array(wyck)/math.gcd(*wyck) for wyck in self.wyckoff_counts_atom_type]
        if self.atom_type_selection == "True":
            wyckoff_ratio_atom_type = wyckoff_ratio_atom_type[basis_type.index(type_i)].tolist()
            uc_wyckoff_ratio = self.uc_wyckoff_ratio[basis_type.index(type_i)].tolist()
        else:
            uc_wyckoff_ratio = [t.tolist() for t in self.uc_wyckoff_ratio]
            wyckoff_ratio_atom_type = [wyck.tolist() for wyck in wyckoff_ratio_atom_type]

        # 2. Point Group Check
        if self.atom_type_selection == "True": target_typeid = type_i
        else: target_typeid = None

        pg_rcut, _ = self._detect_symmetry(particle, rcut, system, unitcell, uc_indices, target_typeid, None, self.pg_tolerance, rtol)

        # Get invariant and equivalent points within max_len from the reference particle
        equivalent_invariant_points = _get_equivalent_invariant_points(system, unitcell, particle, uc_indices, type_i, 
                                                                       round(max_len_ref, rd_factor), rtol, pg_tolerance=self.pg_tolerance)
        # Neighbors of reference particles
        pg_identical_wyckoff = _get_pointgroup_neighbors_of_neighbors(system, 
                                        unitcell, 
                                        rcut,
                                        particle, 
                                        uc_indices, 
                                        atom_type=target_typeid,
                                        rtol=rtol,
                                        pg_tolerance=self.pg_tolerance,
                                        extra_positions=None, 
                                        extra_particles_types=None,
                                        extra_particles_wyckoffs=None,
                                        show=False,
                                        dist_map=dist_map,
                                        length_contraction=getattr(self, 'length_contraction', 1.0)
        )
        
        
        # 3. Geometric Invariant Check
        iq_arr, moi, f_e_ratio, param_arr = self._geometry_poly(system, unitcell, uc_indices, particle, target_typeid, round(max_len_ref, rd_factor), rcut, rtol, 
                                                                length_contraction=getattr(self, 'length_contraction', 1.0))
        try:
            moi_difference = float(abs(moi[0] - moi[1]))
            iq_arr_difference = float(abs(iq_arr[0] - iq_arr[1]))
            f_e_ratio_difference = float(abs(f_e_ratio[0] - f_e_ratio[1]))
        except:
            moi_difference = float('inf')
            iq_arr_difference = float('inf')
            f_e_ratio_difference = float('inf')

        # 4. Asymmetric Unit Check within rcut
        try:
            asym_obj = find_asymmetric_unit_class()
            asym_obj.compute(
                system=system, unitcell=unitcell, uc_particle_indices=uc_indices,
                particle=particle, rcut=round(rcut, rd_factor), rtol=rtol, atol=None,
                poly_pg=pg_rcut, space_group_number=self.space_group_number,
                sys_prep_pf=self.sys_prep_pf, 
                show=False, pg_tolerance=self.pg_tolerance, site_symmetry_dict=self.site_symmetry_dict,
                equivalent_invariant_points=equivalent_invariant_points, 
                atom_type_selection=self.atom_type_selection,
                type_id=type_i
            )
            decision = asym_obj.decision
            chosen_particle_arr = len(asym_obj.chosen_particle_arr)

        except:
            asym_obj = find_asymmetric_unit_class()
            asym_obj.pg_symbol = None
            decision = None
            chosen_particle_arr = 0

        try:
            ratio = rcut / self.max_d_ref if self.max_d_ref else 0.0
            eq_counts_str = str(self.atom_counts_per_equiv_ids)
            orien_counts_str = str(self.count_orien_per_equiv_atoms)
            print(
                f"{rcut:8.3f} | "
                f"{ratio:10.3f} | "
                f"{eq_counts_str:<20} | "
                f"{orien_counts_str:<20} | "
                f"{self.total_symm_order:10d} | "
                f"{pg_rcut:^10} | "
                f"{pg_identical_wyckoff:^10} | "
                f"{moi_difference:10.3e} "
            )
            
            if type_i == system.typeids[particle] and self.total_symm_order >= req_order and pg_rcut != "None" and pg_rcut == target_pg and self.pg_order_data[unitcell.crystal_pointgroup] == pg_identical_wyckoff and round(moi_difference, rd_factor) <= rtol:
                self.required_condition = required_condition
                return True
            elif type_i != system.typeids[particle] and self.total_symm_order >= req_order and pg_rcut == target_pg and self.pg_order_data[unitcell.crystal_pointgroup] == pg_identical_wyckoff:
                self.required_condition = required_condition
                return True
            else:
                self.required_condition = required_condition
                return False
        
        except Exception as e:
            pass

    def _track_particles_within_rcut(
                                    self, 
                                    particle: int, 
                                    rcut: float, 
                                    system: Any, 
                                    unitcell: Any, 
                                    uc_particle_indices: List[int], 
                                    unique_equiv_indices: List[int],
                                    basis_type: List[str], 
                                    PointGroup_dic: Dict[str, Dict[str, Any]], 
                                    MaxDist_dic: Dict[str, float],
                                    type_i: str, 
                                    rtol: float, 
                                    atom_type_selection: str = "True",
                                    pg_tolerance: float = 0.1,
                                ) -> None:
        """Tracks and aggregates particle counts, Wyckoff types, and unique orientations.

        Args:
            particle (int): Central reference particle index.
            rcut (float): Cutoff radius to search within.
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Particle indices in the unit cell.
            unique_equiv_indices (List[int]): Unique equivalent atom indices.
            basis_type (List[str]): Active basis atom types.
            PointGroup_dic (Dict[str, Dict[str, Any]]): Map of point group symmetries.
            MaxDist_dic (Dict[str, float]): Map of maximum distances.
            type_i (str): Selected atom type name.
            rtol (float): Relative tolerance.
            atom_type_selection (str): 'True' to filter by atom type, 'False' otherwise. Defaults to "True".
            pg_tolerance (float): Point group tolerance. Defaults to 0.1.
        """
        rd_factor = int(math.log10(1 / rtol))
        ref_type = system.typeids[particle]
        self.max_d_ref = MaxDist_dic.get(f"{system.equiv_atoms[particle]}_{'all' if atom_type_selection == 'False' else type_i}", 0.0)
        
        # Mapping for O(1) index lookups
        equiv_to_idx = {eq_id: i for i, eq_id in enumerate(unique_equiv_indices)}
        basis_to_idx = {b_type: i for i, b_type in enumerate(basis_type)}

        # 1. Neighbor Retrieval
        neighbor_ids, neighbor_coords = _get_particle_ids_coords_rcut(
            system, unitcell, particle, uc_particle_indices, rcut, rtol=rtol
        )

        # Filter by atom type if requested
        if atom_type_selection == "True":
            neighbor_ids = [pid for pid in neighbor_ids if system.typeids[pid] == type_i]

        # Get the unique set of neighbor IDs within the unit cell
        neighbor_ids, _ = _get_particles_uc(system, unitcell, uc_particle_indices, neighbor_ids)

        # Include the reference particle itself
        neighbor_ids.append(particle)
        self.neigh_id = neighbor_ids

        # 2. State Initialization
        wyckoff_type_list = [[] for _ in basis_type]
        track_equiv_atoms_count = [0 for _ in unique_equiv_indices]
        orientations_per_equiv = [[] for _ in unique_equiv_indices]
        total_symm_order = 0
        sym_order_wyckoff = 0

        # 3. Particle Analysis Loop
        for p_id in neighbor_ids:
            p_type = system.typeids[p_id]
            p_equiv = int(system.equiv_atoms[p_id])
            
            # Determine the key for PointGroup_dic lookup
            type_key = "all" if atom_type_selection == "False" else str(type_i)
            lookup_id = p_equiv
            max_dist = MaxDist_dic.get(f"{lookup_id}_{type_key}", 0.0)
            
            # Track Wyckoff and Orientation data
            if p_type in basis_to_idx and p_equiv in equiv_to_idx:
                b_idx = basis_to_idx[p_type]
                e_idx = equiv_to_idx[p_equiv]
                
                wyckoff_type_list[b_idx].append(system.wyckoffs[p_id])
                track_equiv_atoms_count[e_idx] += 1
                
                if system.local_env_cluster_ids[p_id] not in orientations_per_equiv[e_idx]:
                    orientations_per_equiv[e_idx].append(system.local_env_cluster_ids[p_id])
                    pg_key = f"{lookup_id}_{round(max_dist, rd_factor)}"
                    total_symm_order += PointGroup_dic.get(pg_key, {}).get('pg_order', 0)
                    if system.wyckoffs[p_id] == system.wyckoffs[particle]:
                        sym_order_wyckoff += PointGroup_dic.get(pg_key, {}).get('pg_order', 0)

        # 4. Aggregation and Final Metrics  
        count_orien_per_equiv = [len(s) for s in orientations_per_equiv]
        
        # Count unique orientations per Wyckoff site per basis type
        wyckoff_type_orientations_count = [[] for _ in basis_type]
        for b_idx, b_type in enumerate(basis_type):
            for w_key in self.w_keys_type[b_idx]:
                relevant_equivs = [
                    eq for eq in unique_equiv_indices 
                    if system.typeids[uc_particle_indices[eq]] == b_type 
                    and system.wyckoffs[uc_particle_indices[eq]] == w_key
                ]
                
                site_count = sum(count_orien_per_equiv[equiv_to_idx[eq]] for eq in relevant_equivs)
                wyckoff_type_orientations_count[b_idx].append(site_count)

        # Wyckoff counts per atom type
        wyckoff_counts_atom_type = []
        counter = 0
        for info in wyckoff_type_list:
            vals, counts = np.unique(info, return_counts=True)
            sorted_counts = []
            for c in self.unique_wyckoff_atom_type[counter]:
                if c in vals:
                    sorted_counts.append(counts[list(vals).index(c)])
                else:
                    sorted_counts.append(0)
            wyckoff_counts_atom_type.append(sorted_counts)
            counter += 1

        # Update class attributes
        self.wyckoff_type_orientations_count = wyckoff_type_orientations_count
        self.count_orien_per_equiv_atoms = count_orien_per_equiv
        self.atom_counts_per_equiv_ids = track_equiv_atoms_count
        self.wyckoff_type_list = wyckoff_type_list
        self.total_symm_order = total_symm_order
        self.wyckoff_counts_atom_type = wyckoff_counts_atom_type
        self.sym_order_wyckoff = sym_order_wyckoff
        self.neighbor_coords = neighbor_coords

    def _get_rcut_candidates(self, particle: int, max_len: float, system: Any, rd_factor: int) -> List[float]:
        """Retrieves a sorted list of unique shell distances around a particle.

        Args:
            particle (int): Central reference particle index.
            max_len (float): Maximum search radius.
            system (gsd.hoomd.Snapshot): The system snapshot.
            rd_factor (int): Precision scaling factor for rounding.

        Returns:
            List[float]: Sorted list of unique rounding-adjusted neighbor distances.
        """
        vec_arr = [system.box.wrap(system.positions[j] - system.positions[particle]) 
                   for j in range(len(system.positions)) if j != particle]
        dists = [np.linalg.norm(v) for v in vec_arr if np.linalg.norm(v) <= max_len + 1e-5]
        rounded = sorted(list(set([round(float(d), rd_factor) for d in dists if d > 1e-4])))
        return rounded

    def _update_site_symmetry_cache(self, system, uc_indices, pg_map, dist_map, typeid, site_symmetry_dict, site_symmetry_order, rd_factor):
        """Updates the internal cache for site symmetries per Wyckoff position.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Particle indices in the unit cell.
            pg_map (Dict[str, Dict[str, Any]]): Point group symmetry map.
            dist_map (Dict[str, float]): Maximum distances map.
            typeid (str): Atom type ID.
            site_symmetry_dict (Dict[str, str]): Site symmetry dictionary to update.
            site_symmetry_order (Dict[str, int]): Site symmetry order dictionary to update.
            rd_factor (int): Precision scaling factor.

        Returns:
            Tuple[Dict[str, str], Dict[str, int]]: The updated site symmetry name and order dictionaries.
        """
        equiv_ids = [uc_indices[i] for i in self.unique_equiv_indices]
        for p_id in equiv_ids:
            max_d = dist_map.get(f"{p_id}_{typeid}", 0.0)
            pg_info = pg_map.get(f"{p_id}_{dist_map.get(f'{p_id}_{typeid}', 0.0)}", {"pg_sym": "C1", "pg_order": 0})
            site_symmetry_dict[system.wyckoffs[p_id]] = pg_info['pg_sym']
            site_symmetry_order[system.wyckoffs[p_id]] = pg_info['pg_order']

        return site_symmetry_dict, site_symmetry_order

    def _detect_symmetry(self, particle, rcut, system, unitcell, uc_indices, 
                         atom_type, wyckoff_type, tolerance, rtol) -> Tuple[str, Any]:
        """Determines point group symmetry using Pymatgen PointGroupAnalyzer.

        Args:
            particle (int): Central reference particle index.
            rcut (float): Cutoff radius for symmetry analysis.
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Particle indices in the unit cell.
            atom_type (Optional[str]): Atom type filter, or None.
            wyckoff_type (Optional[str]): Wyckoff position filter, or None.
            tolerance (float): Symmetry analyzer tolerance.
            rtol (float): Relative tolerance.

        Returns:
            Tuple[str, Any]: A tuple of the point group symbol (Schoenflies) and the list of symmetry operations.
        """
        p_ids, p_coords = _get_particle_ids_coords_rcut(system, unitcell, particle, uc_indices, rcut, rtol=rtol)
        p_ids.append(particle)
        p_coords.append(np.array([0, 0, 0]))
        
        if atom_type is not None:
            mask = [system.typeids[pid] == atom_type for pid in p_ids]
            p_coords = [p_coords[i] for i, val in enumerate(mask) if val]
            p_ids = [p_ids[i] for i, val in enumerate(mask) if val]

        if wyckoff_type is not None:
            mask = [system.wyckoffs[pid] == wyckoff_type for pid in p_ids]
            p_coords = [p_coords[i] for i, val in enumerate(mask) if val]
            p_ids = [p_ids[i] for i, val in enumerate(mask) if val]

        species_labels = [system.typeids[pid] for pid in p_ids]

        # Get PG symbol and operations
        pg_symbol, pg_operations = self._get_PG(p_coords, species_labels, tolerance)

        return pg_symbol, pg_operations

    def _get_PG(self, p_coords, species_labels, tolerance) -> Tuple[str, Any]:
        """Helper function to get point group symbol and operations using Pymatgen.

        Args:
            p_coords (List[np.ndarray]): Particle coordinates.
            species_labels (List[str]): Chemical species labels.
            tolerance (float): Point group analysis tolerance.

        Returns:
            Tuple[str, Any]: Schoenflies symbol and list of symmetry operations, or ("None", "None").
        """
        if len(p_coords) < 4 or Points(p_coords).are_coplanar(tol=1e-2):
            return "None", "None"
        
        try:
            mol = Molecule(species=species_labels, coords=p_coords)
            pg_analyzer = PointGroupAnalyzer(mol, tolerance=tolerance)
            return pg_analyzer.sch_symbol, pg_analyzer.get_symmetry_operations()
        except Exception:
            return "None", "None"

    def _geometry_poly(self, system, unitcell, uc_indices, particle, type_id, max_len, rcut, rtol, atom_type_selection="True",
                       length_contraction=1.0) -> Tuple[List[float], List[float], List[float], List[List[int]]]:
        """Analyzes geometric invariants (IQ, MOI) of the local coordinate shell.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_indices (List[int]): Particle indices in the unit cell.
            particle (int): Reference particle index.
            type_id (str): Atom type ID.
            max_len (float): Maximum length scale.
            rcut (float): Cutoff distance candidate.
            rtol (float): Relative tolerance.
            atom_type_selection (str): 'True' to filter by type, 'False' otherwise. Defaults to "True".
            length_contraction (float): Length contraction factor. Defaults to 1.0.

        Returns:
            Tuple[List[float], List[float], List[float], List[List[int]]]: A tuple containing:
                - List of isoperimetric quotients (IQ) for max_len and rcut.
                - List of moment of inertia parameters for max_len and rcut.
                - List of vertex/face ratio parameters for max_len and rcut.
                - List of polyhedral shape counts (vertices, faces, edges) or None.
        """
        iq_arr, moi_arr, f_e_ratio, param_arr = [], [], [], []
        vecs = None
        for rc in [max_len * length_contraction, rcut]:
            try:
                p_ids, p_coords = _get_particle_ids_coords_rcut(system, unitcell, particle, uc_indices, rc, rtol=rtol)
                if atom_type_selection == "True":
                    mask = [system.typeids[pid] == type_id for pid in p_ids]
                    p_coords = [p_coords[i] for i, val in enumerate(mask) if val]
                    p_ids = [p_ids[i] for i, val in enumerate(mask) if val]

                hull = ConvexHull(p_coords)
                vertices = np.array([p_coords[i] for i in hull.vertices])
                vertices = vertices * (1/hull.volume)**(1/3)
                hull = ConvexHull(vertices)
                vertices = np.array([vertices[i] for i in hull.vertices])
                poly = coxeter.shapes.ConvexPolyhedron(vertices)
                moi = _moment_of_inertia(poly.vertices)
                m = [moi[0], moi[1], moi[2]]
                param = 1 - ((m[0]**2 - m[1]**2)**2 + (m[1]**2 - m[2]**2)**2 + (m[2]**2 - m[0]**2)**2) / (2 * sum(i**2 for i in m)**2)
                iq_arr.append(float(poly.iq))
                moi_arr.append(param)
                f_e_ratio.append(float(len(poly.edges) / len(poly.faces)))
                param_arr.append([len(vertices), len(poly.faces), len(poly.edges)])
            except Exception:
                iq_arr.append(None); moi_arr.append(None); f_e_ratio.append(None); param_arr.append(None)

        return iq_arr, moi_arr, f_e_ratio, param_arr

    def _compute_single_particle(self, unitcell, uc_particle_indices):
        """Handles trivial unit cells with only one basis atom.

        Args:
            unitcell (freud.data.UnitCell): The unit cell object.
            uc_particle_indices (List[int]): Particle indices in the unit cell.
        """
        latt_vectors = np.transpose(unitcell.box.to_matrix())
        norms = [float(np.linalg.norm(v)) for v in latt_vectors]
        self.rcut_arr = [norms]
        self.rcut_all_arr = [norms]
        self.ref_particle_arr = [[uc_particle_indices[0]]]

    def _get_equiv_atom_counts_per_type(self, system, uc_indices, basis_type) -> Tuple[List[Dict[int, int]], List[Dict[str, int]]]:
        """Collects equivalent atom counts and symmetry centers per basis atom type.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Particle indices in the unit cell.
            basis_type (List[str]): Active basis atom types.

        Returns:
            Tuple[List[Dict[int, int]], List[Dict[str, int]]]: A tuple containing:
                - Equivalent atom counts per type.
                - Symmetry center counts per type.
        """
        equiv_atoms_count_per_type = [{} for _ in range(len(basis_type))]
        symm_cntr_per_type = [[] for _ in range(len(basis_type))]
        for t in range(len(basis_type)):
            particle_ids = [uc_indices[i] for i in range(len(uc_indices)) if system.typeids[uc_indices[i]] == basis_type[t]]
            indices = [[i for i, e in enumerate(uc_indices) if e == j][0] for j in particle_ids]
            symm_arr_type = [system.wyckoffs[i] for i in indices]
            symm_val_type, symm_cntr_type = np.unique(symm_arr_type, return_counts=True)
            symm_val_type, symm_cntr_type = symm_val_type.tolist(), symm_cntr_type.tolist()
            symm_cntr_per_type[t] = dict(zip(symm_val_type, symm_cntr_type))

            equiv_atoms_type = [system.equiv_atoms[i] for i in particle_ids]
            vals, counts = np.unique(equiv_atoms_type, return_counts=True)
            vals, counts = vals.tolist(), counts.tolist()
            vals_, counts_ = [], []
            for w in symm_val_type:
                for ei, e in enumerate(vals):
                    if system.wyckoffs[e] == w:
                        vals_.append(vals[ei])
                        counts_.append(counts[ei])

            equiv_atoms_type_cntr = dict(zip(vals_, counts_))
            equiv_atoms_count_per_type[t] = equiv_atoms_type_cntr

        return equiv_atoms_count_per_type, symm_cntr_per_type
    
    def _get_sg_symmmetry_equiv_pg_symmetry_equiv_inv_points(self, system, unitcell, particle, uc_indices, type_i, max_len_ref, rd_factor, rtol) -> Tuple[List[int], List[Any]]:
        """Retrieves space group and point group symmetry equivalent invariant points.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            unitcell (freud.data.UnitCell): The unit cell object.
            particle (int): Reference particle index.
            uc_indices (List[int]): Particle indices in the unit cell.
            type_i (str): Selected atom type name.
            max_len_ref (float): Maximum reference distance.
            rd_factor (int): Precision scaling factor.
            rtol (float): Relative tolerance.

        Returns:
            Tuple[List[int], List[Any]]: A tuple of space group equivalent atom IDs and point group equivalent invariant points.
        """
        # Symmetry dataset retrieval (Spglib)
        basis_types = np.unique([system.typeids[i] for i in uc_indices]).tolist()
        numbers = [basis_types.index(system.typeids[t]) for t in uc_indices]
        lattice = [t.tolist() for t in unitcell.box.to_matrix().T]
        cell = (lattice, unitcell.fractional_positions, numbers)
        symm_dict = spglib.get_symmetry_dataset(cell, symprec=0.1, angle_tolerance=5.0)
        equiv_atoms_sg = symm_dict.equivalent_atoms
        # Space group symmetry equivalent points
        equiv_atomsids_sg = [uc_indices[j] for j in range(len(uc_indices)) if equiv_atoms_sg[j] == equiv_atoms_sg[uc_indices.index(particle)]]

        equivalent_invariant_points = _get_equivalent_invariant_points(system, unitcell, particle, uc_indices, type_i, 
                                                                       round(max_len_ref, rd_factor), rtol, pg_tolerance=self.pg_tolerance)

        return equiv_atomsids_sg, equivalent_invariant_points