"""
Symmetry and Wyckoff Position Analysis Module
=============================================

This module defines the :class:`WyckoffAnalyzer` class, which categorizes atoms/particles 
in a crystal structure based on their local chemical environments, radial distribution functions (RDF),
Bond Order Diagrams (BOD), and crystallographic symmetries (Wyckoff positions).
"""

import numpy as np
import freud
from itertools import chain
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# Internal project imports
from ..environment.rdf import rdf_class
from ..environment.bod import bod_class

class WyckoffAnalyzer:
    """Identifies Wyckoff positions and local structural symmetries for crystal particles.

    This class categorizes atoms based on their chemical environment and 
    geometric symmetry, facilitating the selection of reference particles.

    Attributes:
        system_wyckoff (Optional[List[str]]): List of Wyckoff letter classifications mapped system-wide.
        local_env_cluster_ids (Optional[List[List[int]]]): Local environment cluster labels.
        cluster_centers (Optional[np.ndarray]): Computed mean center of each local environment cluster.
        rdf_rcut (Optional[float]): Radial cut-off distance determined from RDF analysis.
        equiv_atoms (Optional[List[int]]): List of equivalent atom IDs in the unit cell.
    """

    def __init__(self):
        """Initializes the WyckoffAnalyzer with default placeholder states."""
        self.system_wyckoff = None
        self.local_env_cluster_ids = None
        self.cluster_centers = None
        self.rdf_rcut = None
        self.equiv_atoms = None

    def compute_wyckoff_positions(self, 
                                 unitcell,
                                 system,
                                 uc_particle_indices,
                                 rmax, 
                                 bins,
                                 rcut=None,
                                 spg_mode=True,
                                 num_replicas=1,
                                 show_rdf=False):
        """Executes the symmetry detection, environment clustering, and Wyckoff classification.

        Args:
            unitcell (freud.data.UnitCell): The freud unit cell data.
            system (gsd.hoomd.Snapshot): The global snapshot state of the system.
            uc_particle_indices (List[int]): Particle indices belonging to the primary unit cell.
            rmax (float): Maximum cutoff distance for RDF analysis.
            bins (int): Number of bins for RDF calculation.
            rcut (Optional[float]): Cutoff distance for Bond Order Diagram analysis. 
                If None, a default heuristic is used. Defaults to None.
            spg_mode (bool): If True, classifies using Spglib database. If False, falls back to
                Pymatgen PointGroup analysis. Defaults to True.
            num_replicas (int): Linear replica multiplier of the unit cell in the system. Defaults to 1.
            show_rdf (bool): If True, plots the RDF curve. Defaults to False.
        """
        # 1. Radial Distribution Function Analysis
        rdf_obj = rdf_class()
        rdf_obj.rdf_func(system.positions, system.box, rmax=rmax, bins=bins, show_rdf=show_rdf)
    
        # Default to a heuristic if rcut is not provided.
        if rcut is None:
            rcut = rmax * 0.75 
            print(f"[INFO] No rcut provided. Defaulting to: {rcut:.2f}")
        
        self.rdf_rcut = rcut

        # 2. Local Environment Detection (BOD)
        bod_obj = bod_class(system.positions, system.box, rcut)
        pos_arr, neighbor_li, _ = bod_obj.bod_func()

        # 3. Clustering Environments (Freud)
        cl = freud.cluster.Cluster()
        box_size = 2 * rcut
        search_box = freud.box.Box.cube(box_size)
        aq = freud.locality.AABBQuery(search_box, pos_arr)
        
        rmax_env = 0.05
        cl.compute(aq, neighbors={"r_max": rmax_env})
        
        labels = cl.cluster_idx.astype(int)
        self.cluster_centers = self._calculate_cluster_centers(pos_arr, labels, cl.num_clusters)
                    
        neighbor_li_len = [len(t) for t in neighbor_li]
        separataion_arr_flattened = list(chain.from_iterable(neighbor_li))

        group_arr = []
        c = 0
        for i in range(len(neighbor_li_len)):
            sorted_arr = labels[c:c+neighbor_li_len[i]]
            sorted_arr.sort()
            group_arr.append(sorted_arr.tolist())
            c = c+neighbor_li_len[i]

        self.local_env_cluster_ids = group_arr

        group_arr_uc = [group_arr[i] for i in range(len(group_arr)) if i in uc_particle_indices]
        unique_group = []
        for i in range(len(group_arr_uc)):
            if group_arr_uc[i] not in unique_group:
                unique_group.append(group_arr_uc[i])

        uc_env_indices = []
        for i in range(len(uc_particle_indices)):
            indices = [j for j, e in enumerate(unique_group) if e == group_arr_uc[i]]
            uc_env_indices.append(indices[0])      

        # 4. Symmetry Categorization
        if spg_mode:
            self._analyze_with_spglib(unitcell, system, uc_particle_indices, labels, num_replicas)
        else:
            self._analyze_with_pymatgen(system, uc_particle_indices, rcut)

    def _calculate_cluster_centers(self, positions, labels, n_clusters) -> np.ndarray:
        """Computes the mean position for each detected cluster environment.

        Args:
            positions (np.ndarray): Array of 3D coordinates.
            labels (np.ndarray): Cluster classification labels.
            n_clusters (int): Total number of clusters.

        Returns:
            np.ndarray: Computed 3D centroids of each cluster.
        """
        centers = []
        for i in range(n_clusters):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                centers.append(np.mean(np.array([positions[idx] for idx in indices]), axis=0))

        return np.array(centers)

    def _analyze_with_spglib(self, unitcell, system, uc_indices, labels, num_replicas):
        """Internal helper to process Spglib-derived symmetry.

        Args:
            unitcell (freud.data.UnitCell): The unit cell object.
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Primary unit cell particle indices.
            labels (np.ndarray): Environment cluster labels.
            num_replicas (int): Replica count multiplier.
        """
        equiv_atoms = list(unitcell.equivalent_atoms)
        wyckoffs = unitcell.wyckoffs
        
        unique_ids, counts = np.unique(equiv_atoms, return_counts=True)
        id_to_wyckoff = {uid: wyckoffs[equiv_atoms.index(uid)] for uid in unique_ids}
        
        print(f"[DEBUG] Spglib Unique Environments: {len(unique_ids)}")

        symm_val, symm_cntr = np.unique(wyckoffs, return_counts=True)
        symm_val, symm_cntr = symm_val.tolist(), symm_cntr.tolist()
        symm_cntr = dict(zip(symm_val, symm_cntr))

        basis_type_ = np.unique(unitcell.typeids).tolist()
        system_typesids = system.typeids
        symm_cntr_per_type = [[] for _ in range(len(basis_type_))]
        equiv_atoms_count_per_type = [{} for _ in range(len(basis_type_))]
        for t in range(len(basis_type_)):
            particle_ids = [uc_indices[i] for i in range(len(uc_indices)) if system_typesids[uc_indices[i]] == basis_type_[t]]
            indices = [[i for i, e in enumerate(uc_indices) if e == j][0] for j in particle_ids]
            symm_arr_type = [wyckoffs[i] for i in indices]
            symm_val_type, symm_cntr_type = np.unique(symm_arr_type, return_counts=True)
            symm_val_type, symm_cntr_type = symm_val_type.tolist(), symm_cntr_type.tolist()
            symm_cntr_per_type[t] = dict(zip(symm_val_type, symm_cntr_type))

            equiv_atoms_type = [system.equiv_atoms[i] for i in particle_ids]
            vals, counts = np.unique(equiv_atoms_type, return_counts=True)
            vals, counts = vals.tolist(), counts.tolist()
            vals_, counts_ = [], []
            for w in symm_val_type:
                for ei, e in enumerate(vals):
                    if wyckoffs[e] == w:
                        vals_.append(vals[ei])
                        counts_.append(counts[ei])

            equiv_atoms_type_cntr = dict(zip(vals_, counts_))
            equiv_atoms_count_per_type[t] = equiv_atoms_type_cntr

        print(f"[INFO] Wyckoff per type: {symm_cntr_per_type}")
        print(f"[INFO] Equivalent atoms per type: {equiv_atoms_count_per_type}")

        symm_keys = list(symm_cntr.keys())
        symm_arr_index = [[k for k, e in enumerate(symm_keys) if e == i] for i in wyckoffs]
        N = np.prod((num_replicas, num_replicas, num_replicas))
        indices = np.repeat(np.arange(len(unitcell.basis_positions)), N)
        self.system_wyckoff = list(chain.from_iterable(np.array(symm_arr_index)[indices]))
        self.system_wyckoff = [symm_keys[i] for i in self.system_wyckoff]

        self.equiv_atoms = equiv_atoms

    def _analyze_with_pymatgen(self, system, uc_indices, rcut):
        """Fallback point-group analysis for non-standard structures.

        Args:
            system (gsd.hoomd.Snapshot): The system snapshot.
            uc_indices (List[int]): Unit cell particle indices.
            rcut (float): Cutoff radius for neighbor detection.
        """
        symm_labels = []
        for idx in uc_indices:
            neighbors = []
            
            if not neighbors:
                symm_labels.append("C1")
                continue

            mol = Molecule(species=['C'] * len(neighbors), coords=neighbors)
            pga = PointGroupAnalyzer(mol, tolerance=0.1)
            symm_labels.append(pga.sch_symbol)
            
        self.system_wyckoff = symm_labels