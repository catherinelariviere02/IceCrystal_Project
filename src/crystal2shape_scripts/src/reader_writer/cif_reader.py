"""CIF structure reader module.

This module provides classes to parse structure definitions from Crystallographic
Information Files (CIF) and construct unit cell and global system configurations.
"""

import ase.io
import numpy as np
from collections import Counter
import freud
from ase.cell import Cell
import json, spglib
import os
from scipy.spatial import Delaunay, ConvexHull
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from ..utils.basis import basis_func
from ..environment.rdf import rdf_class
from ..utils.utils_func import safe_input


class UnitCell():
    """Unit cell definition representing box geometry, positions, and atom types."""

    def __init__(self):
        pass

    def construct(self, box, basis_positions, fractional_positions, atom_types):
        """Constructs a unit cell from the given geometric parameters.

        Args:
            box (freud.box.Box): The box representing the unit cell dimensions.
            basis_positions (np.ndarray): The positions of basis atoms.
            fractional_positions (np.ndarray): The fractional coordinate positions.
            atom_types (List[str]): List of species names mapping types.
        """
        
        self.box = box
        self.basis_positions = basis_positions
        self.fractional_positions = fractional_positions
        self.typeids = atom_types

        box_matrix = self.box.to_matrix()
        b1, b2, b3 = box_matrix[:,0], box_matrix[:,1], box_matrix[:,2]
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

        box_corners = np.array([boxvert1, boxvert3, boxvert2, boxvert5, boxvert7, boxvert6, boxvert4, boxvert8])
        self.box_corners = box_corners

class System:
    """Simulation system state container."""

    def __init__(self):
        pass

    def construct(self, box, positions, typeids):
        """Constructs a simulation system from coordinates and box dimensions.

        Args:
            box (freud.box.Box): The simulation box.
            positions (np.ndarray): Particle position coordinates.
            typeids (List[int] or np.ndarray): Type classification indices.
        """
        self.box = box
        self.positions = positions
        self.typeids = typeids

class CIF_reader:
    """
    Read CIF file and extract information about the unit cell, positions, and types of atoms.
    The CIF file is read using the ASE package and the unit cell is generated using the freud package.
    
    Parameters
    ----------
    input_CIF : str
        Path to the CIF file.

    Attributes
    ----------
    uc_box_ : freud.box.Box
        Box object representing the unit cell.

    pos : np.ndarray
        Positions of atoms in the unit cell.

    box_arr : freud.box.Box
        Box object representing the replicated system.

    positions : np.ndarray
        Positions of atoms in the replicated system.

    original_positions : np.ndarray
        Original positions of atoms in the CIF file.

    aq : freud.locality.AABBQuery
        AABB query object for the replicated system.

    particleids : list      
        List of particle IDs for the replicated system.

    original_particleids : list
        List of original particle IDs from the CIF file.

    uc_particle_indices : list
        Indices of unit cell particles in the replicated system.

    latt_vectors : np.ndarray
        Lattice vectors of the unit cell.

    atom_type_ : list
        List of atom types in the unit cell.

    basis_type_ : list
        List of basis types in the unit cell.

    original_basis_type_ : list
        List of original basis types from the CIF file.

    sys_types : np.ndarray
        System types of the replicated system.

    uc_indices : np.ndarray
        Indices of unit cell particles in the replicated system.
    """
    def __init__(self, input_CIF):
        self.input_CIF = input_CIF
    
    def _CIF_reader_func(self, num_replicas=3, target_pf=None, rounding_factor=1, rmax=3.0, bins=500, separate_equiv_atoms=False, **kwargs):
        """Parses a CIF file and replicates the unit cell lattice.

        Args:
            num_replicas (int): Replication factor along cell vectors. Defaults to 3.
            target_pf (Optional[float]): Optional target packing fraction. Defaults to None.
            rounding_factor (int): Precision decimal rounding count. Defaults to 1.
            rmax (float): Maximum RDF search distance. Defaults to 3.0.
            bins (int): RDF histogram bin count. Defaults to 500.
            separate_equiv_atoms (bool): Separate equivalent atom types if True. Defaults to False.
            **kwargs: Dynamic attributes set as class settings.
        """
        r = rounding_factor # Rounding factor

        from ase.io import read
        atoms = read(self.input_CIF, store_tags=True)
        print(atoms.get_array('occupancy'))


        from pymatgen.core import Structure
        structure = Structure.from_file(self.input_CIF)

        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        '''analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
        space_group_symbol = analyzer.get_space_group_symbol()
        symmetry_operations = analyzer.get_symmetry_operations(cartesian=True)
        point_group = analyzer.get_point_group_symbol()
        print("Crystallographic point group : ", point_group)'''

        data = {}
        for key, value in kwargs.items():
            data[key] = value

        # Read the CIF file using ASE
        atoms = ase.io.read(self.input_CIF)
        box = atoms.cell
        uc_positions = atoms.positions

        # print(box, len(positions))
        center = np.sum(np.array([t/2 for t in box]), axis=0)
        uc_positions = uc_positions - center
        uc_box = freud.box.Box.from_matrix(np.transpose(box))
        lattice_vectors = np.transpose(uc_box.to_matrix())
        fractional_positions = uc_box.make_fractional(uc_positions)
        original_fractional_positions = fractional_positions.copy()
        original_positions = uc_positions.copy()

        original_basis_type_ = [atoms.symbols[i] for i in range(len(fractional_positions))]
        raw_basis_type_ = original_basis_type_[:]
        type_cntr = Counter(original_basis_type_)
        basis_type_, atom_type_ = list(type_cntr.keys()), list(type_cntr.values())
        original_typeids = [basis_type_.index(original_basis_type_[i]) for i in range(len(original_basis_type_))]

        for t in range(len(basis_type_)):
            numbers = [j for j, e in enumerate(original_basis_type_) if e == basis_type_[t]]
            lattice = [t.tolist() for t in np.transpose(uc_box.to_matrix())]
            frac_coords = [fractional_positions[i] for i in numbers]
            cell = (lattice, frac_coords, [0 for _ in range(len(numbers))])

            sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0)
            print("Space group of ", basis_type_[t], " : ", sg) # Dummy print statement

        # Considering the same atom type in the basis
        cell = (lattice, fractional_positions, [0 for _ in range(len(fractional_positions))])
        sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0)
        print("Space group considering all atom types identical: ", sg) # Dummy print statement

        print("Basis types : ", basis_type_)
        valid_choices = [-1, -2] + list(range(len(basis_type_)))
        user_input = safe_input(
            prompt="Enter a number (-1 for all and -2 for identical atom type) : ",
            expected_type=int,
            valid_choices=valid_choices
        )

        if user_input == -1 :
            pass
        elif user_input == -2:
            original_basis_type_ = [basis_type_[0] for i in range(len(fractional_positions))]
            type_cntr = Counter(original_basis_type_)
            basis_type_, atom_type_ = list(type_cntr.keys()), list(type_cntr.values())
            print("Selected basis type : ", original_basis_type_)

        else:
            basis_type_ = [basis_type_[user_input]]
            atom_type_ = [atom_type_[user_input]]
            indices = [i for i, e in enumerate(original_basis_type_) if e == basis_type_[0]]
            original_basis_type_ = [original_basis_type_[i] for i in indices]
            fractional_positions = np.array([fractional_positions[i] for i in indices])
            original_fractional_positions = fractional_positions.copy()
            uc_positions = np.array([uc_positions[i] for i in indices])
            print("Selected basis type : ", original_basis_type_)

            # Separating equivalent atoms per Wyckoff site using spglib
            #----------------------------------------------------------
            if separate_equiv_atoms == True:
                basis_atom_types = basis_type_[:]
                numbers = [0 for _ in range(len(fractional_positions))]
                lattice = [t.tolist() for t in np.transpose(uc_box.to_matrix())]
                cell = (lattice, fractional_positions, numbers)
                symm_dict = spglib.spglib.get_symmetry_dataset(cell, symprec=0.01, angle_tolerance=0.1)
                equiv_atoms = symm_dict.equivalent_atoms
                equiv_atoms = equiv_atoms.tolist()
                wyckoffs = symm_dict.wyckoffs
                # print("Equivalent atoms from spglib: ", equiv_atoms) # Dummy print statement
                unique_equiv_atoms, symm_counts = np.unique(equiv_atoms, return_counts=True)
                unique_equiv_atoms = unique_equiv_atoms.tolist()
                print("Unique equivalent atoms from spglib: ", unique_equiv_atoms) # Dummy print statement
                equiv_atom_dict = dict(zip(unique_equiv_atoms, symm_counts.tolist()))
                print("Equivalent atom counts : ", equiv_atom_dict) # Dummy print statement

                unique_wyckoffs = [wyckoffs[i] for i in range(len(wyckoffs)) if equiv_atoms[i] in unique_equiv_atoms]
                sorted_indices = []
                for uw in range(len(unique_wyckoffs)):
                    equiv_ids = [i for i in range(len(wyckoffs)) if wyckoffs[i] == unique_wyckoffs[uw]]
                    unique_equiv_ids = list(set([equiv_atoms[i] for i in equiv_ids]))
                    chosen_id = unique_equiv_ids[0]
                    chosen_indices = [i for i in range(len(equiv_atoms)) if equiv_atoms[i] == chosen_id]
                    for c in chosen_indices:
                        if c not in  sorted_indices:
                            sorted_indices.append(c)

                # print("Sorted indices of equivalent atoms: ", sorted_indices)

                fractional_positions = np.array([fractional_positions[i] for i in sorted_indices])
                original_fractional_positions = fractional_positions.copy()
                uc_positions = np.array([uc_positions[i] for i in sorted_indices])
                original_basis_type_ = [original_basis_type_[i] for i in sorted_indices]

                # Considering the same atom type in the basis
                cell = (lattice, fractional_positions, [0 for _ in range(len(fractional_positions))])
                sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0)
                print("Space group after separating equiv atoms: ", sg) # Dummy print statement

            # Choose the particles corresponding to any specific Point Group symmetry of the coordination polyhedon
            #-------------------------------------------------------------------------------------------
            # Set at target pf
            '''if target_pf is not None:
                target_volume = len(fractional_positions)/target_pf
                uc_box = freud.box.Box(Lx=uc_box.Lx* (target_volume/uc_box.volume) ** (1/3),
                                    Ly=uc_box.Ly* (target_volume/uc_box.volume) ** (1/3),
                                    Lz=uc_box.Lz* (target_volume/uc_box.volume) ** (1/3),
                                    xy=uc_box.xy,
                                    xz=uc_box.xz,
                                    yz=uc_box.yz)

            uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions)
            dummy_box, dummy_positions = uc.generate_system(num_replicas=1)

            uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions)
            system_box, system_positions = uc.generate_system(num_replicas=num_replicas)
            unitcell_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in dummy_positions]
            # System positions rounded
            system_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in system_positions]
            uc_particle_indices = [[j for j in range(len(system_positions_rounded)) if system_positions_rounded[j]==p][0] for p in unitcell_positions_rounded]
            # print("UC particle ids for selected atom type : ", uc_particle_indices) # Dummy print
            rcut = round(float(max([np.linalg.norm(dummy_box.wrap(dummy_positions[m] - dummy_positions[n])) for m in range(len(dummy_positions)) for n in range(len(dummy_positions))])), 2)

            symm_arr = []
            # voronoi = freud.locality.Voronoi()
            # cells = voronoi.compute((system.box, system.positions)).polytopes
            for i in range(len(uc_particle_indices)):
                neighbor_posi = [system_box.wrap(system_positions[j] - system_positions[uc_particle_indices[i]]) for j in range(len(system_positions)) if round(float(np.linalg.norm(system_box.wrap(system_positions[j] - system_positions[uc_particle_indices[i]]))), 2) <= rcut]
                mol = Molecule(species=['C']*len(neighbor_posi), coords=neighbor_posi)
                pga = PointGroupAnalyzer(mol, tolerance=0.1)
                symm = pga.sch_symbol
                
                symm_arr.append(str(symm))

            symm_val, symm_count = np.unique(symm_arr, return_counts=True)
            symm_val, symm_count = [str(t) for t in symm_val], [int(t) for t in symm_count]
            symm_dict = dict(zip(symm_val, symm_count))
            print("Point group symmetries and counts for selected atom type : ", symm_dict)

            valid_choices = [-1] + list(range(len(symm_dict)))
            usr_input = safe_input(
                prompt="Please choose the required symmetry type (-1 for all) : ",
                expected_type=int,
                valid_choices=valid_choices
            )
            req_symm = [list(symm_dict.keys())[usr_input]] if usr_input != -1 else list(symm_dict.keys())
            basis_type_ = basis_type_[:]
            atom_type_ = atom_type_[:]
            indices = [i for i in range(len(original_basis_type_)) if symm_arr[i] in req_symm]
            original_basis_type_ = [original_basis_type_[i] for i in indices]
            fractional_positions = np.array([fractional_positions[i] for i in indices])
            original_fractional_positions = fractional_positions.copy()
            uc_positions = np.array([uc_positions[i] for i in indices])
            print("Chosen symmetries : ", req_symm)

            # Considering the same atom type in the basis
            cell = (lattice, fractional_positions, [0 for _ in range(len(fractional_positions))])
            sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0)
            print("Space group of considered particles: ", sg) # Dummy print statement'''


        # Set at target pf
        if target_pf != "None":
            target_volume = len(fractional_positions)/target_pf
            uc_box = freud.box.Box(Lx=uc_box.Lx* (target_volume/uc_box.volume) ** (1/3),
                                Ly=uc_box.Ly* (target_volume/uc_box.volume) ** (1/3),
                                Lz=uc_box.Lz* (target_volume/uc_box.volume) ** (1/3),
                                xy=uc_box.xy,
                                xz=uc_box.xz,
                                yz=uc_box.yz)
            
            lattice_vectors = np.transpose(uc_box.to_matrix())

        # In case we want to generate the Structure from space group number, lattice vectors, basis types and fractional positions
        #------------------------------------------------------------------------------------------
        # Crystallographic point group
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        basis_atom_types = basis_type_
        numbers = [basis_atom_types.index(t) for t in original_basis_type_]
        lattice = [t.tolist() for t in np.transpose(uc_box.to_matrix())]
        cell = (lattice, fractional_positions, numbers)
        symm_dict = spglib.spglib.get_symmetry_dataset(cell, symprec=0.1, angle_tolerance=5.0)
        equiv_atoms_uc = symm_dict.equivalent_atoms
        equiv_atoms = np.unique(equiv_atoms_uc)
        # print(equiv_atoms)

        fractional_positions_equiv = [fractional_positions[i] for i in equiv_atoms]
        basis_type_equiv = [original_basis_type_[i] for i in equiv_atoms]
        fractional_positions_equiv = [fractional_positions[i] for i in equiv_atoms]
        structure = Structure.from_spacegroup(
                    sg=data["space_group_number"],
                    lattice=lattice_vectors,
                    species=basis_type_equiv,
                    coords=fractional_positions_equiv,
                    tol=1e-2)
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
        symmetry_operations = analyzer.get_symmetry_operations(cartesian=True)
        # symmetry_operations = analyzer.get_point_group_operations()
        point_group = analyzer.get_point_group_symbol()
        print("Crystallographic point group : ", point_group)

        # Replicate only once
        uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions)
        uc_box, uc_positions = uc.generate_system(num_replicas=1)

        # Construct UnitCell object
        unitcell = UnitCell()
        unitcell.construct(box=uc_box,
                           basis_positions=uc_positions,
                           fractional_positions=fractional_positions,
                           atom_types=original_basis_type_)

        # Numbers for each basis id in the unit cell
        numbers = []
        for i in range(len(original_basis_type_)):
            ind = basis_type_.index(original_basis_type_[i])
            numbers.append(ind)
    
        unique_types_ = basis_type_[:]
        modified_basis_type_ = unique_types_[:]
        modified_atom_type_ = []
        for i in range(len(modified_basis_type_)):
            index = basis_type_.index(modified_basis_type_[i])
            modified_atom_type_.append(atom_type_[index])

        basis_type_ = modified_basis_type_[:]
        atom_type_ = modified_atom_type_[:]

        # Replicate the unit cell to create the system
        uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions)
        system_box, system_positions = uc.generate_system(num_replicas=num_replicas)
        indices = np.repeat(np.arange(len(uc.basis_positions)), num_replicas**3)
        system_typeids = np.array(unitcell.typeids)[indices]

        # Construct the System object
        system = System()
        system.construct(box=system_box, positions=system_positions, typeids=system_typeids)

        # Compare the basis positions to get unit cell particle ids from the replicated system
        # Unit cell positions rounded
        unitcell_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in unitcell.basis_positions]
        # System positions rounded
        system_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in system_positions]

        uc_particle_indices = [[j for j in range(len(system_positions_rounded)) if system_positions_rounded[j]==p][0] for p in unitcell_positions_rounded]
        print("UC particle ids : ", uc_particle_indices) # Dummy print

        # Collect all the attributes
        self.unitcell = unitcell
        self.system = system
        self.uc_particle_indices = uc_particle_indices
        self.equiv_atoms = equiv_atoms_uc
