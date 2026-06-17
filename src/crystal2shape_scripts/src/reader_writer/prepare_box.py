import numpy as np
import freud
import scipy
from collections import Counter
import spglib
from ..reader_writer.construct_system import constructSystem
from ..reader_writer.construct_system import UnitCell
from ..reader_writer.construct_system import System
from ..reader_writer.construct_system import CrystalStructure
from ..utils.utils_func import safe_input

class prepareBox():
    """
    A class to prepare the simulation box for the system.
        
    """

    def __init__(self, input_CIF):
        self.input_CIF = input_CIF

    def prepareBox_func(self, num_replicas, 
                        target_pf, 
                        write_gsd, 
                        directory,
                        shape_id,
                        space_group_number,
                        separate_equiv_atoms,
                        **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        try:
            r = self.rounding_factor # Rounding factor
        except AttributeError:
            r = 1

        reader = constructSystem(self.input_CIF)
        reader.construct(num_replicas=num_replicas, target_pf=target_pf, space_group_number=space_group_number)

        unitcell = reader.unitcell # UnitCell object
        system = reader.system # System object

        # Get Space group information for each atom type in the basis
        space_group_info = self._get_SpaceGroup_atom_type_info(unitcell)
        unique_basis_types = list(space_group_info.keys())
        unique_basis_types = [atom_type for atom_type in unique_basis_types if atom_type in np.unique(system.typeids)]

        print("Basis types : ", unique_basis_types)
        if len(unique_basis_types) == 1:
            fractional_positions_chosen = np.array(unitcell.fractional_positions) # Fractional positions of all basis types
            uc_positions_chosen = np.array(unitcell.basis_positions) # Lattice positions of all basis types
            basis_type_uc_chosen = unitcell.typeids # Basis types of all basis types

        else:
            basis_type_uc_chosen, fractional_positions_chosen, uc_positions_chosen = self._choose_basistypes(unitcell, unique_basis_types)

        # Get symmetry information using CrystalStructure class
        crysStruct = CrystalStructure(self.input_CIF)
        crysStruct.construct(basis_type_uc_chosen, unitcell.box, fractional_positions_chosen, unitcell.lattice_vectors, space_group_number)
        sg_symmetry_operations = crysStruct.sg_symmetry
        pg_symmetry_operations = crysStruct.pg_symmetry
        crystal_pointgroup = crysStruct.crystal_point_group
        equiv_atoms_uc = crysStruct.equiv_atoms_uc
        wyckoffs = crysStruct.wyckoffs
        num_operations = crysStruct.num_operations
        site_symmetry_symbols = crysStruct.site_symmetry_symbols
        print("Number of crystal symmetry operations : ", num_operations)
        
        uc_box = unitcell.box
        lattice_vectors = np.transpose(uc_box.to_matrix())
        # Set at target pf
        if target_pf != 0.0:
            target_volume = len(unitcell.fractional_positions)/target_pf
            uc_box = freud.box.Box(Lx=uc_box.Lx* (target_volume/uc_box.volume) ** (1/3),
                                Ly=uc_box.Ly* (target_volume/uc_box.volume) ** (1/3),
                                Lz=uc_box.Lz* (target_volume/uc_box.volume) ** (1/3),
                                xy=uc_box.xy,
                                xz=uc_box.xz,
                                yz=uc_box.yz)
            
            lattice_vectors = np.transpose(uc_box.to_matrix())

        # Construct UnitCell object
        uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions_chosen)
        uc_box, uc_positions = uc.generate_system(num_replicas=1)

        uc_ = UnitCell()
        uc_.construct(box=uc_box,
                       lattice_vectors=lattice_vectors,
                       basis_positions=uc_positions,
                       fractional_positions=fractional_positions_chosen,
                       atom_types=basis_type_uc_chosen,
                        equiv_atoms=equiv_atoms_uc,
                        sg_symmetry=sg_symmetry_operations,
                        pg_symmetry=pg_symmetry_operations,
                        crystal_point_group=crystal_pointgroup,
                        wyckoffs=wyckoffs,
                        num_operations=num_operations,
                        site_symmetry_symbols=site_symmetry_symbols)

        # Replicate the unit cell to create the system
        uc = freud.data.UnitCell(uc_.box, basis_positions=uc_.fractional_positions)
        system_box, system_positions = uc.generate_system(num_replicas=num_replicas)
        indices = np.repeat(np.arange(len(uc.basis_positions)), num_replicas**3)
        system_typeids = np.array(basis_type_uc_chosen)[indices]

        # Construct the System object
        system = System()
        system.construct(box=system_box, positions=system_positions, typeids=system_typeids)


        # Compare the basis positions to get unit cell particle ids from the replicated system
        # Unit cell positions rounded
        unitcell_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in uc_.basis_positions]
        
        # System positions rounded
        system_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in system_positions]

        uc_particle_indices = [[j for j in range(len(system_positions_rounded)) if system_positions_rounded[j]==p][0] for p in unitcell_positions_rounded]
        print("UC particle ids : ", uc_particle_indices) # Dummy print

        # Collect all the attributes
        self.unitcell = uc_
        self.system = system
        self.uc_particle_indices = uc_particle_indices

    def _get_SpaceGroup_atom_type_info(self, unitcell):
        """Helper function to get space group information for each atom type in the basis."""
        type_cntr = Counter(unitcell.typeids)
        unique_basis_types, occr_ = list(type_cntr.keys()), list(type_cntr.values())
        space_group_info = {}

        for t in range(len(unique_basis_types)):
            numbers_type_ = [j for j, e in enumerate(unitcell.typeids) if e == unique_basis_types[t]] # Indices of the specific atom type
            lattice_type_ = [t.tolist() for t in np.transpose(unitcell.box.to_matrix())] # lattice vectors
            frac_coords_type_ = [unitcell.fractional_positions[i] for i in numbers_type_] # Fractional coordinates for each basis type
            cell = (lattice_type_, frac_coords_type_, [0 for _ in range(len(numbers_type_))]) # Cell object for spglib

            sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0) # Space group of the specific atom type
            space_group_info[unique_basis_types[t]] = sg

        # Considering all atom types as identical and get the space group symmetry
        lattice = [t.tolist() for t in np.transpose(unitcell.box.to_matrix())] # lattice vectors
        cell = (lattice, unitcell.fractional_positions, [0 for _ in range(len(unitcell.fractional_positions))])
        sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0)
        print("Space group considering all atom types identical: ", sg) # Dummy print statement

        space_group_info["all_identical"] = sg

        return space_group_info
    
    def _choose_basistypes(self, unitcell, unique_basis_types):
        "Helper function to choose basis types for shape generation."

        valid_choices = [-1, -2] + list(range(len(unique_basis_types)))
        user_input = safe_input(
            prompt="Enter a number (-1 for all and -2 for identical atom type) : ",
            expected_type=int,
            valid_choices=valid_choices
        )

        if user_input == -1 :
            fractional_positions_chosen = np.array(unitcell.fractional_positions) # Fractional positions of all basis types
            uc_positions_chosen = np.array(unitcell.basis_positions) # Lattice positions of all basis types
            basis_type_uc_chosen = unitcell.typeids # Basis types of all basis types
            print("Selected all basis types.")

        elif user_input == -2:
            type_cntr = Counter(unitcell.typeids)
            unique_basis_types, occr_ = list(type_cntr.keys()), list(type_cntr.values())
            fractional_positions_chosen = np.array(unitcell.fractional_positions) # Fractional positions of all basis types
            uc_positions_chosen = np.array(unitcell.basis_positions) # Lattice positions of all basis types
            basis_type_uc_chosen = [unitcell.typeids[0] for _ in range(len(uc_positions_chosen))] # Basis types of all basis types
        

            print("Selected all basis types as identical atom type.")

        else:
            basis_type_chosen = [unique_basis_types[user_input]] # Chosen basis type by the user
            indices = [i for i, e in enumerate(unitcell.typeids) if e == basis_type_chosen[0]] # indices of the chosen basis type
            basis_type_uc_chosen = [unitcell.typeids[i] for i in indices] # Basis type in the unit cell for the chosen basis type
            fractional_positions_chosen = np.array([unitcell.fractional_positions[i] for i in indices]) # Fractional positions of the chosen basis type
            uc_positions_chosen = np.array([unitcell.basis_positions[i] for i in indices]) # Lattice positions of the chosen basis type
            print("Selected basis type : ", unique_basis_types[user_input])


        return basis_type_uc_chosen, fractional_positions_chosen, uc_positions_chosen

