import ase.io
import freud 
import numpy as np 


def com(molecules, positions, masses):
    """
    inputs: 
    molecules - array of indices of each 
    positions - all positions in the system
    masses - mass of each of the 3 (in order [O, H, H])

    outputs: 
    center of mass with zero as oxygen centers, with size (len(molecules), 3) 
    """
    centers_of_mass = np.zeros((len(molecules), 3))
    for i, molecule in enumerate(molecules): 
        mol_pos = positions[molecule]
        normed_positions = mol_pos - np.array([mol_pos[0]]*3)
        print("normed pos ", normed_positions)
        com = np.transpose(normed_positions) @ masses / sum(masses)
        # com = np.sum(normed_positions, axis = 0) / 3
        print("com ", com)
        centers_of_mass[i, :] = com

    return centers_of_mass

def molecules(cif, masses):
    # define uc characteristics 
    atoms = ase.io.read(cif)
    types = list(atoms.symbols) # H then O 
    n_hydrogen = types.count("H")

    #replicate system 
    uc = freud.data.UnitCell.from_cif(cif)
    
    n_repeats = 3
    dimension = 3
    system = uc.generate_system(num_replicas = n_repeats)

    #indices of center unit cell - the center index of each set of replicated basis positions
    center_indices = int((n_repeats**dimension - n_repeats)/2) + np.arange(0, len(uc.basis_positions)) * (n_repeats**dimension) 
    print("indices ", center_indices) 

    #define position arrays 
    positions = np.array(system[1])
    center_positions = positions[center_indices]
    new_position = []

    #hydrogens are positions from start
    hyd_pos = positions[0:n_hydrogen * (n_repeats**dimension) - 1] 
    center_ox_pos = center_positions[n_hydrogen-1::]
    print("center ox pos len ", len(center_ox_pos))
    box = uc.box

    aq = freud.locality.AABBQuery.from_system(system)
    # query all points which have a distance of 0.98 between then (distance between hydrogen and oxygen)
    nlist = np.array(aq.query(system[1], {'r_max': 1, 'r_min': 0.5}).toNeighborList())

    # define molecule array: [O, H, H] which make up each molecule in unit cell 
    molecules = [np.insert(np.where(nlist[:, 1] == ind)[0], 0, ind) 
                 for ind in center_indices[n_hydrogen::]]
    print(molecules)

    center_of_mass = com(molecules, positions, masses)

    # first block (original cell) positisions 
    corner_indices = np.arange(0, len(uc.basis_positions)) * (n_repeats**dimension)

    # take only original oxygen positions, move them by center of mass
    com_pos = positions[corner_indices[n_hydrogen::]] + center_of_mass
    np.set_printoptions(precision=7, suppress=True)

    com_fractional = box.make_fractional(com_pos)
    print(f"center of mass pos: {(com_fractional)}")

    wrapped_com_pos = box.wrap(com_pos)

    return molecules, positions, wrapped_com_pos 
    com(molecules, positions, masses)


crystal_name = "141_H2O_0"
input_dir = "/home/clarivi/projects/IceCrystal_Project/inputs_updated/141_H2O_0/"

cif = input_dir + f"{crystal_name}.cif"

# mass of each atom in molecule 
masses = [15.99, 1.008, 1.008]
mol_inds, positions, com_pos = molecules(cif, masses)


