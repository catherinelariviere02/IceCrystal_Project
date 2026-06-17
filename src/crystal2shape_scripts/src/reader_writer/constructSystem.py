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
        
        """
        Construct a unit cell from the given parameters.

        Parameters
        ----------
        box : freud.box.Box
            The box representing the unit cell.

        lattice_vectors : np.ndarray
            The lattice vectors of the unit cell.

        positions : np.ndarray
            The positions of the atoms in the unit cell.

        fractional_positions : np.ndarray
            The fractional positions of the atoms in the unit cell.

        atom_types : list
            The types of atoms in the unit cell.

        equiv_atoms : list
            The equivalent atoms in the unit cell.

        sg_symmetry : list
            The space group symmetry operations.
        
        pg_symmetry : list
            The point group symmetry operations.

        crystal_point_group : str
            The crystallographic point group symbol.

        wyckoffs : list
            The Wyckoff positions of the atoms in the unit cell.

        num_operations : int
            The number of symmetry operations in the space group.

        site_symmetry_symbols : list
            The site symmetry symbols for each atom in the unit cell.

        """
        
        self.box = box
        self.lattice_vectors=lattice_vectors
        self.basis_positions = basis_positions
        self.fractional_positions = fractional_positions
        self.typeids = atom_types
        self.wyckoffs = wyckoffs

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

        self.equivalent_atoms = equiv_atoms
        self.sg_symmetry = sg_symmetry
        self.pg_symmetry = pg_symmetry
        self.crystal_pointgroup = crystal_point_group
        self.num_operations = num_operations
        self.site_symmetry_symbols = site_symmetry_symbols

class System:
    def __init__(self):
        pass

    def construct(self, box, positions, typeids):
        """
        Construct a system from the given parameters.

        Parameters
        ----------
        box : freud.box.Box
            The box representing the system.

        positions : np.ndarray
            The positions of the atoms in the system.

        typeids : list
            The types of atoms in the system.

        """
        self.box = box
        self.positions = positions
        self.typeids = typeids

class CrystalStructure:
    def __init__(self, input_CIF):
        self.input_CIF = input_CIF

    def construct(self, basis_types_uc, uc_box, fractional_positions, lattice_vectors, space_group_number=None):
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

    def construct(self, num_replicas=3, target_pf=None, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Read the CIF file using ASE
        atoms = ase.io.read(self.input_CIF, store_tags=True)
        box_ = atoms.cell
        uc_positions = atoms.positions
        # print(atoms.info.get('occupancy'))

        # print(box, len(positions))
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

        