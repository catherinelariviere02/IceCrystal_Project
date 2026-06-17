from tokenize import group
import numpy as np
import spglib
import freud
import ase.io
from pymatgen.symmetry.groups import SpaceGroup


class UnitCell():
    def __init__(self):
        pass

    def construct(self, box, 
                  basis_positions, 
                  fractional_positions, 
                  atom_types,
                  equiv_atoms,
                  wyckoffs):
        
        """
        Construct a unit cell from the given parameters.

        Parameters
        ----------
        box : freud.box.Box
            The box representing the unit cell.

        positions : np.ndarray
            The positions of the atoms in the unit cell.

        fractional_positions : np.ndarray
            The fractional positions of the atoms in the unit cell.

        atom_types : list
            The types of atoms in the unit cell.

        equiv_atoms : list
            The indices of equivalent atoms in the unit cell.

        wyckoffs: list
            The wyckoff positions of the atoms in the unit cell.
        """
        
        self.box = box
        self.basis_positions = basis_positions
        self.fractional_positions = fractional_positions
        self.typeids = atom_types
        self.equivalent_atoms = equiv_atoms
        self.wyckoffs = wyckoffs

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

class uc_class():
    def __init__(self, space_group_number):
        self.space_group_number = space_group_number

    def create_uc(self, num_replicas, target_pf, rounding_factor=1, **kwargs):
        r = rounding_factor
        '''sym_data = spglib.get_symmetry_from_database(self.space_group_number)

        rotations = sym_data['rotations']       # list of 3x3 rotation matrices
        translations = sym_data['translations'] # list of translation vectors

        lattice_parameters = 10.0, 10.0, 10.0, 90, 90, 90  # Example cubic lattice parameters
        Lx, Ly, Lz = lattice_parameters[0], lattice_parameters[1], lattice_parameters[2], 
        alpha, beta, gamma = np.deg2rad(lattice_parameters[3]), np.deg2rad(lattice_parameters[4]), np.deg2rad(lattice_parameters[5])
        uc_box = freud.box.Box.from_box_lengths_and_angles(L1=Lx, L2=Ly, L3=Lz, alpha=alpha, beta=beta, gamma=gamma)
        lattice_vectors = np.transpose(uc_box.to_matrix())

        ref_point = np.random.rand(3) * np.array([lattice_vectors[0][0], lattice_vectors[1][1], lattice_vectors[2][2]])
        
        uc_positions = [uc_box.wrap(ref_point)]
        for i, (R, t) in enumerate(zip(rotations, translations)):
            rotated_trans_point = np.dot(R, ref_point) + t[0]*lattice_vectors[0] + t[1]*lattice_vectors[1] + t[2]*lattice_vectors[2]
            cntr_arr = [1 for pos in uc_positions if np.allclose(uc_box.wrap(rotated_trans_point), pos, atol=1e-2, rtol=1e-2) == True]
            if len(cntr_arr) == 0:
                uc_positions.append(uc_box.wrap(rotated_trans_point))

        uc_positions = np.array(uc_positions)   
        fractional_positions = uc_box.make_fractional(uc_positions)'''


        lattice_parameters = 5.0, 5.0, 5.0, 90, 90, 90  # Example cubic lattice parameters
        Lx, Ly, Lz = lattice_parameters[0], lattice_parameters[1], lattice_parameters[2], 
        alpha, beta, gamma = np.deg2rad(lattice_parameters[3]), np.deg2rad(lattice_parameters[4]), np.deg2rad(lattice_parameters[5])
        uc_box = freud.box.Box.from_box_lengths_and_angles(L1=Lx, L2=Ly, L3=Lz, alpha=alpha, beta=beta, gamma=gamma)
        lattice_vectors = np.transpose(uc_box.to_matrix())

        '''dist_min, dist_max = max([Lx, Ly, Lz])/2 + 0.1, 0.0
        counter = 0
        while dist_min > max([Lx, Ly, Lz])/2 or dist_max < max([Lx, Ly, Lz])/2:
            sg = SpaceGroup.from_int_number(self.space_group_number)
            ops = sg.symmetry_ops
            ref_point = np.random.rand(3) # Generate a random reference point in fractional coordinates
            fractional_positions = [(op.operate(ref_point)) % 1 for op in ops]
            uc_positions = uc_box.make_absolute(fractional_positions)
            dist = [np.linalg.norm(uc_box.wrap(uc_positions[i] - uc_positions[j])) for i in range(len(uc_positions)) for j in range(len(uc_positions)) if i!=j]
            dist_min, dist_max = min(dist), max(dist)
            print(counter, dist_min, dist_max) # Dummy print statement
            counter += 1'''
        
        sg = SpaceGroup.from_int_number(self.space_group_number)
        ops = sg.symmetry_ops
        ref_point = np.random.rand(3) # Generate a random reference point in fractional coordinates
        fractional_positions = [(op.operate(ref_point)) % 1 for op in ops]
        uc_positions = uc_box.make_absolute(fractional_positions)

        # print("Fractional positions: ", fractional_positions) # Dummy print statement
        original_basis_type_ = ["A" for _ in range(len(fractional_positions))]
        basis_type_ = ["A"]
        atom_type_= ["A"]

        # Set at target pf
        if target_pf is not None:
            target_volume = len(fractional_positions)/target_pf
            uc_box = freud.box.Box(Lx=uc_box.Lx* (target_volume/uc_box.volume) ** (1/3),
                                Ly=uc_box.Ly* (target_volume/uc_box.volume) ** (1/3),
                                Lz=uc_box.Lz* (target_volume/uc_box.volume) ** (1/3),
                                xy=uc_box.xy,
                                xz=uc_box.xz,
                                yz=uc_box.yz)
            
            lattice_vectors = np.transpose(uc_box.to_matrix())

        # Crystallographic point group
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        basis_atom_types = basis_type_
        numbers = [basis_atom_types.index(t) for t in original_basis_type_]
        lattice = [t.tolist() for t in np.transpose(uc_box.to_matrix())]
        cell = (lattice, fractional_positions, numbers)
        symm_dict = spglib.spglib.get_symmetry_dataset(cell, symprec=0.1, angle_tolerance=5.0)
        equiv_atoms = symm_dict.equivalent_atoms
        equiv_atoms = np.unique(equiv_atoms)
        wyckoffs = symm_dict.wyckoffs

        fractional_positions_equiv = [fractional_positions[i] for i in equiv_atoms]
        basis_type_equiv = [original_basis_type_[i] for i in equiv_atoms]
        structure = Structure.from_spacegroup(
                    sg=self.space_group_number,
                    lattice=lattice_vectors,
                    species=basis_type_equiv,
                    coords=fractional_positions_equiv,
                    tol=1e-3)
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
        point_group = analyzer.get_point_group_symbol()
        print("Crystallographic point group : ", point_group)

        # Replicate only once
        uc = freud.data.UnitCell(uc_box, basis_positions=fractional_positions)
        uc_box, uc_positions = uc.generate_system(num_replicas=1)
        
        numbers = [0 for _ in range(len(fractional_positions))]
        lattice = [t.tolist() for t in np.transpose(uc_box.to_matrix())]
        cell = (lattice, fractional_positions, numbers)
        sg = spglib.get_spacegroup(cell, symprec=1e-02, angle_tolerance=1.0)
        dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
        print("Space group of the constructed unit cell: ", sg) # Dummy print statement

        dic_data = {}
        for key, value in kwargs.items():
            dic_data[key] = value

        if dataset.number == self.space_group_number:
            # Write GSD of the reduced unit cell
            from ..reader_writer import gsd_writer
            typeids, types = numbers[:], ["A"]
            shape_dic = [{"type": "Sphere", "diameter": 0.8}]

            if dic_data["write_gsd"] == True:
                tag = "sharp_" + dic_data["shape_id"] + "_constructed_uc"
                gsd_write_obj = gsd_writer.GSDWriter(filename= dic_data["directory"] + tag + ".gsd")
                gsd_write_obj.write_GSD(uc_box, uc_positions, None, type_ids=typeids, types=types, shape_dic=shape_dic, repeat=[1, 1, 1])

        else:
            raise ValueError("Space group number does not match with the reduced unit cell space group.")


        # Construct UnitCell object
        unitcell = UnitCell()
        unitcell.construct(box=uc_box,
                           basis_positions=uc_positions,
                           fractional_positions=fractional_positions,
                           atom_types=original_basis_type_,
                           equiv_atoms=equiv_atoms,
                           wyckoffs=wyckoffs)

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
        system_positions_rounded = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in system.positions]

        uc_particle_indices = [[j for j in range(len(system_positions_rounded)) if system_positions_rounded[j]==p][0] for p in unitcell_positions_rounded]
        print("UC particle ids : ", uc_particle_indices) # Dummy print

        # Collect all the attributes
        self.unitcell = unitcell
        self.system = system
        self.uc_particle_indices = uc_particle_indices