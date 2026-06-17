import math

import numpy as np
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.core import Molecule
import spglib

from ..utils.utils_func import _get_particle_ids_coords_rcut

import math
import logging
import numpy as np
import spglib
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def _get_pointgroup_polyhedron_rcut(
    system: Any, 
    unitcell: Any,
    uc_particle_indices: List[int], 
    particles_set1: List[int],
    particles_set2: List[int],
    distances: Dict[str, float], 
    rtol: float = 0.01,  
    atom_type_selection: str = "True",
    basis_type: List[str] = ["A", "B"]
) -> Dict[str, Dict[str, Any]]:
    """
    Computes site symmetry symbols and point group orders for a set of particles.
    
    This function leverages spglib for high-precision symmetry detection and 
    calculates the site symmetry order based on the stabilizer subgroup of the 
    space group operations.

    Returns:
        PointGroup_dic: Dictionary mapping 'EquivalentID_Type' to symmetry metadata.
    """
    rd_factor = int(math.log10(1/rtol))
    
    # Define the type identifier for dictionary keys
    target_typeid = "all" if atom_type_selection == "False" else system.typeids[particles_set2[0]]

    # Optimization: Extract symmetry dataset ONCE for the entire unit cell
    lattice = unitcell.lattice_vectors
    positions = np.array([system.positions[t] for t in uc_particle_indices])
    types = [basis_type.index(system.typeids[t]) for t in uc_particle_indices]

    try:  # Debugging block to verify input data for spglib
        logger.debug(f"Unit Cell Lattice:\n{lattice}")
        logger.debug(f"Particle Positions (fractional):\n{positions}")
        logger.debug(f"Particle Types (as indices):\n{types}")
    
    except Exception as e:
        logger.error(f"Debugging input data failed: {e}")
    
    # spglib site_symmetry_symbols detection
    site_symmetry_symbols = unitcell.site_symmetry_symbols
    point_group_dic = {}
    
    # Process each particle in the requested set
    for p_idx in particles_set1:
        # 1. Generate the unique key for this equivalent atom and rcut
        equiv_id = system.equiv_atoms[p_idx]
        key_dist = f"{equiv_id}_{target_typeid}"
        if key_dist not in distances:
            logger.debug(f"Distance key {key_dist} not found in input distances.")
            continue
        
        max_d = distances[key_dist]
        dict_key = f"{equiv_id}_{round(float(max_d), rd_factor)}"
        logger.debug(f"Max distance for key {key_dist}: {max_d}")

        # 2. Map global index back to unit cell local index for spglib data access
        try:
            local_uc_idx = uc_particle_indices.index(p_idx)
        except ValueError:
            logger.warning(f"Particle {p_idx} not found in uc_particle_indices.")
            continue

        # 3. Calculate Multiplicity and Symmetry Order
        # Multiplicity is the number of atoms in the UC equivalent to this site
        current_equiv_atom = system.equiv_atoms[p_idx]
        multiplicity = sum(
            1 for t in uc_particle_indices 
            if system.typeids[t] == system.typeids[p_idx] 
            and system.equiv_atoms[t] == current_equiv_atom
        )
        
        # Point group order (stabilizer) = Total Ops / Multiplicity
        # This is a fundamental crystallographic relation: |G| = |G_s| * |Orbit(s)|
        sym_order = int(unitcell.num_operations / multiplicity)
        
        # 4. Populate result
        point_group_dic[dict_key] = {
            "wyckoff": system.wyckoffs[p_idx],
            "pg_sym": site_symmetry_symbols[local_uc_idx],
            "pg_order": sym_order
        }

    return point_group_dic

def _get_max_distances(particle_ids_type_i, particle_ids_type_j, system, unitcell, uc_indices, rounding_factor, basis_type, atom_type_selection="True", d_tol=0.05):
        """Calculates maximum distance scales for each equivalent atom candidate."""
        # Define the type identifier for dictionary keys
        target_typeid = "all" if atom_type_selection == "False" else system.typeids[particle_ids_type_j[0]]
        max_distances_dic = {}
        for p1 in range(len(particle_ids_type_i)):
            vec_arr = [unitcell.box.wrap(system.positions[j] - system.positions[particle_ids_type_i[p1]]) for j in particle_ids_type_j]
            dists = max(sorted(list(set([float(np.linalg.norm(j)) for j in vec_arr]))))
            max_len = dists + d_tol
            dict_key = f"{system.equiv_atoms[particle_ids_type_i[p1]]}_{target_typeid}"
            if dict_key not in list(max_distances_dic.keys()):
                max_distances_dic[dict_key] = max_len

        return max_distances_dic


def _get_pointgroup(system, 
                     unitcell,
                     uc_particle_indices, 
                     particle,
                     rcut, 
                     pg_order_data, 
                     rtol=0.01, 
                     pg_tolerance=0.1,
                     atom_type_selection="False"):
    
    """Function to get the point group symmetry and order for a given particle based on a specified cutoff distance (rcut)."""

    particle_ids, particle_coords = _get_particle_ids_coords_rcut(system,
                                                                    unitcell,
                                                                    particle,
                                                                    uc_particle_indices,
                                                                    rcut,
                                                                    rtol=rtol)

    # Impose condition of specific atom type
    if atom_type_selection == "True":
        particle_ids = [pid for pid in particle_ids if system.typeids[pid] == system.typeids[particle]]

    particle_ids.append(particle) # Include reference particle in the list of particle ids for point group analysis
    particle_coords = [(system.positions[pid] - system.positions[particle]) for pid in particle_ids]

    # Species labels for point group analysis
    species_label = [system.typeids[j] for j in particle_ids]
    
    try:
        mol = Molecule(species=species_label, coords=particle_coords)
        pga = PointGroupAnalyzer(mol, tolerance=pg_tolerance)
        infopoly_pg = pga.sch_symbol

    except Exception as e:
        logger.debug(f"Point group analysis failed: {e}")
        infopoly_pg = None

    if infopoly_pg is not None:
        sym_order = pg_order_data[infopoly_pg]
    else:
        sym_order = 0

    return infopoly_pg, sym_order