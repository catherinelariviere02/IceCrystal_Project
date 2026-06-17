"""System construction module from crystal coordinates.

This module provides classes to build unit cells and replicated global simulation
configurations from parsed Crystallographic Information Files (CIF) and symmetry details.
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


class UnitCell():
    """Lattice box and basis configuration for a crystal unit cell.

    Attributes:
        box (freud.box.Box): The unit cell box.
        lattice_vectors (np.ndarray): 3x3 matrix of lattice vectors.
        basis_positions (np.ndarray): Cartesian positions of basis atoms.
        fractional_positions (np.ndarray): Fractional coordinates of basis atoms.
        typeids (List[str]): Particle types mapping.
        wyckoffs (List[str]): Wyckoff symbols of particles.
        box_corners (np.ndarray): Corner coordinates of the box.
        equivalent_atoms (np.ndarray): Mapping of equivalent atoms.
        sg_symmetry (List): Space group symmetry operations.
        pg_symmetry (List): Point group symmetry operations.
        crystal_pointgroup (str): Schoenflies crystallographic point group symbol.
        num_operations (int): Number of symmetry operations.
        site_symmetry_symbols (List[str]): Site symmetry labels.
    """

    def __init__(self):
        pass

    def construct(self, box, 
                  lattice_vectors,
                  basis_positions, 
                  fractional_positions, 
                  atom_types,
                  equiv_atoms,
                  sg_symmetry,
                  pg_symmetry,
                  crystal_point_group,
                  wyckoffs,
                  num_operations,
                  site_symmetry_symbols):
        """Constructs the UnitCell object with all lattice and symmetry parameters.

        Args:
            box (freud.box.Box): The unit cell box.
            lattice_vectors (np.ndarray): Lattice vectors.
            basis_positions (np.ndarray): Cartesian coordinates of atoms.
            fractional_positions (np.ndarray): Fractional coordinates.
            atom_types (List[str]): Atom species names.
            equiv_atoms (List[int] or np.ndarray): Equivalent atom indices.
            sg_symmetry (List): Space group symmetry operations.
            pg_symmetry (List): Point group symmetry operations.
            crystal_point_group (str): Crystallographic point group symbol.
            wyckoffs (List[str]): Wyckoff positions of atoms.
            num_operations (int): Number of space group operations.
            site_symmetry_symbols (List[str]): Site symmetry symbols.
        """
        self.box = box
        self.lattice_vectors = lattice_vectors
        self.basis_positions = basis_positions
        self.fractional_positions = fractional_positions
        self.typeids = atom_types
        self.wyckoffs = wyckoffs

        box_matrix = self.box.to_matrix()
        b1, b2, b3 = box_matrix[:,0], box_matrix[:,1], box_matrix[:,2]

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

        self.equivalent_atoms = equiv_atoms
        self.sg_symmetry = sg_symmetry
        self.pg_symmetry = pg_symmetry
        self.crystal_pointgroup = crystal_point_group
        self.num_operations = num_operations
        self.site_symmetry_symbols = site_symmetry_symbols


class System:
    """Simulation configuration state container.

    Attributes:
        box (freud.box.Box): The simulation box.
        positions (np.ndarray): Particle positions.
        typeids (List[str] or np.ndarray): Species name indices.
    """

    def __init__(self):
        pass

    def construct(self, box, positions, typeids):
        """Constructs the system state.

        Args:
            box (freud.box.Box): The simulation box.
            positions (np.ndarray): Particle coordinates.
            typeids (List[str] or np.ndarray): Particle types.
        """
        self.box = box
        self.positions = positions
        self.typeids = typeids


class CrystalStructure:
    """Uses spglib to analyze and compute crystallographic structure databases.

    Attributes:
        input_CIF (str): Path to the input CIF file.
        sg_symmetry (List): Space group symmetry operations.
        pg_symmetry (List): Point group symmetry operations.
        crystal_point_group (str): Schoenflies point group symbol.
        num_operations (int): Operation multiplicity.
        equiv_atoms_uc (np.ndarray): Equivalent atoms mapping array.
        wyckoffs (np.ndarray): Wyckoff symbol mapping array.
        site_symmetry_symbols (np.ndarray): Site symmetry labels.
    """

    def __init__(self, input_CIF):
        """Initializes CrystalStructure.

        Args:
            input_CIF (str): Path to input CIF.
        """
        self.input_CIF = input_CIF

    def construct(self, basis_types_uc, uc_box, fractional_positions, lattice_vectors, space_group_number=None):
        """Builds crystallographic dataset parameters from cell attributes.

        Args:
            basis_types_uc (List[str]): Unique basis types in the unit cell.
            uc_box (freud.box.Box): Unit cell box geometry.
            fractional_positions (np.ndarray): Fractional coordinates of basis atoms.
            lattice_vectors (np.ndarray): Lattice vector directions.
            space_group_number (Optional[int]): space group index number. Defaults to None.
        """
        basis_atom_types = basis_types_uc
        numbers = [basis_atom_types.index(t) for t in basis_atom_types]
        lattice = [t.tolist() for t in np.transpose(uc_box.to_matrix())]
        cell = (lattice, fractional_positions, numbers)
        symm_dict = spglib.get_symmetry_dataset(cell, symprec=0.1, angle_tolerance=5.0)
        equiv_atoms_uc = symm_dict.equivalent_atoms
        unique_equiv_atoms = np.unique(equiv_atoms_uc)
        wyckoffs = symm_dict.wyckoffs # Wyckoff positions of unit cell atoms
        crystal_point_group = symm_dict.pointgroup
        num_operations = len(symm_dict.rotations)
        site_symmetry_symbols = symm_dict.site_symmetry_symbols

        fractional_positions_equiv = [fractional_positions[i] for i in range(len(unique_equiv_atoms))]
        basis_type_equiv = [basis_types_uc[i] for i in range(len(unique_equiv_atoms))]
        fractional_positions_equiv = [fractional_positions[i] for i in range(len(unique_equiv_atoms))]
        
        sg_symmetry_operations = []
        pg_symmetry_operations = symm_dict.rotations

        self.sg_symmetry = sg_symmetry_operations
        self.pg_symmetry = pg_symmetry_operations
        self.crystal_point_group = crystal_point_group
        self.num_operations = num_operations
        self.equiv_atoms_uc = equiv_atoms_uc
        self.wyckoffs = wyckoffs
        self.site_symmetry_symbols = site_symmetry_symbols


class constructSystem:
    """Parses unit cell files and instantiates System snapshot classes.

    Attributes:
        input_CIF (str): Path to input CIF file.
        unitcell (UnitCell): The initialized unit cell.
        system (System): The initialized global system.
    """

    def __init__(self, input_CIF):
        """Initializes constructor.

        Args:
            input_CIF (str): Path to CIF.
        """
        self.input_CIF = input_CIF

    def construct(self, num_replicas=3, target_pf=None, **kwargs):
        """Performs unit cell parsing, scaling, replication, and snap construction.

        Args:
            num_replicas (int): Linear duplication factor along axes. Defaults to 3.
            target_pf (Optional[float]): Optional target packing fraction. Defaults to None.
            **kwargs: Dynamic class fields.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Read the CIF file using ASE
        atoms = ase.io.read(self.input_CIF, store_tags=True)
        box_ = atoms.cell
        uc_positions = atoms.positions

        center = np.sum(np.array([t/2 for t in box_]), axis=0) # center of the box
        uc_positions = uc_positions - center # Lattice sites with respect to center
        uc_box = freud.box.Box.from_matrix(np.transpose(box_)) # Freud box for unit cell
        lattice_vectors = np.transpose(uc_box.to_matrix()) # unit cell lattice vectors
        fractional_positions = uc_box.make_fractional(uc_positions) # Fractional positions of the lattice sites
        basis_types_uc = [str(atoms.symbols[i]) for i in range(len(fractional_positions))] # Atom types of the basis particles

        # Get symmetry information using CrystalStructure class
        crysStruct = CrystalStructure(self.input_CIF)
        crysStruct.construct(basis_types_uc, uc_box, fractional_positions, lattice_vectors, self.space_group_number)
        sg_symmetry_operations = crysStruct.sg_symmetry
        pg_symmetry_operations = crysStruct.pg_symmetry
        crystal_point_group = crysStruct.crystal_point_group
        num_operations = crysStruct.num_operations
        equiv_atoms_uc = crysStruct.equiv_atoms_uc
        wyckoffs = crysStruct.wyckoffs
        site_symmetry_symbols = crysStruct.site_symmetry_symbols
        print("Crystallographic point group : ", crystal_point_group)

        # Construct UnitCell object
        unitcell = UnitCell()
        unitcell.construct(box=uc_box,
                           lattice_vectors=lattice_vectors,
                           basis_positions=uc_positions,
                           fractional_positions=fractional_positions,
                           atom_types=basis_types_uc,
                           equiv_atoms=equiv_atoms_uc,
                           sg_symmetry=sg_symmetry_operations,
                           pg_symmetry=pg_symmetry_operations,
                           crystal_point_group=crystal_point_group,
                           wyckoffs=wyckoffs,
                           num_operations=num_operations,
                           site_symmetry_symbols=site_symmetry_symbols)


        lattice_vectors = np.transpose(uc_box.to_matrix())
        # Set at target pf
        if target_pf != 0.0:
            target_volume = len(fractional_positions)/target_pf
            uc_box = freud.box.Box(Lx=uc_box.Lx* (target_volume/uc_box.volume) ** (1/3),
                                Ly=uc_box.Ly* (target_volume/uc_box.volume) ** (1/3),
                                Lz=uc_box.Lz* (target_volume/uc_box.volume) ** (1/3),
                                xy=uc_box.xy,
                                xz=uc_box.xz,
                                yz=uc_box.yz)
            
            lattice_vectors = np.transpose(uc_box.to_matrix())

        # Replicate the unit cell to create the system
        uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions)
        system_box, system_positions = uc.generate_system(num_replicas=num_replicas)
        indices = np.repeat(np.arange(len(uc.basis_positions)), num_replicas**3)
        system_typeids = np.array(unitcell.typeids)[indices]
        system_typeids = [str(t) for t in system_typeids]

        # Construct the System object
        system = System()
        system.construct(box=system_box, positions=system_positions, typeids=system_typeids)

        # Collect all the attributes
        self.unitcell = unitcell
        self.system = system
