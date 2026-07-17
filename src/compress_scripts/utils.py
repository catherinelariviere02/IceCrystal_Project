import hoomd 
import ase.io
from ase.formula import Formula
import json 
import coxeter 

def create_simulation(filename, frame, shapes, atoms, communicator):
    """
    Inputs: 
    Filename: filename (including path) of gsd file 
    frame: frame to use of GSD file 
    shapes: list of crystal2shape formatted dictionaries giving info on shape vertices
    atoms: array of atom types (i.e. ["O", "H"])

    Outputs: HPMC simulation with atom shapes input to integrator 
    """
    cpu = hoomd.device.CPU(communicator=communicator)
    simulation = hoomd.Simulation(device = cpu, seed = 1)
    
    simulation.create_state_from_gsd(filename, frame)

    mc = hoomd.hpmc.integrate.ConvexPolyhedron()

    for i, shape in enumerate(shapes): 
        mc.shape[atoms[i]] = dict(
            vertices = shape["8_vertices"]
        ) 
        mc.a[atoms[i]] = 0.2
        mc.d[atoms[i]] = 0.5

    simulation.operations.integrator = mc 

    return simulation

def get_shape_info(input_dir, N_scaling, types, crystal_name):
    """
    Inputs: 
    N_scaling = replication size of simulation box 
    types = array of strings of atoms types (i.e. ["O", "H"]), order doesn't matter
    cif = cif file for crystal you're assembling 

    Outputs: 
    atoms = ase "Atom" object 
    uc_atom_counts = dictionary of atom type and number of atom type (i.e {'O': 36, 'H': 72}) 
    sim_type_counts = list of number of particles of each type  
    type_shapes = "type_shape" dictionary for hoomd: type, rounding radius, vertices from json file 
    typeid = list assigning type to every particle (in order of "sim_type_counts") in box 
    shape_json_dicts = list of dictionaries from crystal2shape json files (one for each type) 
    spacing = spacing between lattice points, 2.2 times the larger minimal bounding radius of the shapes in the uc 
    shape_volume = total volume taken by all the particles in the simulation 
    """
    cif = input_dir + f"{crystal_name}.cif"
    atoms = ase.io.read(cif)
    uc_atom_counts = Formula(str(atoms.symbols)).count()
    sim_type_counts = []
    type_shapes = [] 
    typeid = []
    shape_json_dicts = []
    radius = []
    shape_volume = 0
    for i, type in enumerate(types): 
        shape_file = input_dir + f"/shape_{crystal_name}_{type}_unit_volume_principal_frame.json"
        sim_type_counts.append(uc_atom_counts[type] * N_scaling ** 3) # N_scaling is "replicas" which could be made, so cubed for 3D  
        typeid = typeid + ([i] * sim_type_counts[i])
        with open(shape_file) as file: 
            shape_json_dicts.append(json.load(file))
            type_shapes.append(dict(type="ConvexPolyhedron", 
                                    rounding_radius = 0,
                                    vertices = shape_json_dicts[i]["8_vertices"]))
            poly=coxeter.shapes.ConvexPolyhedron(shape_json_dicts[i]["8_vertices"])
            radius.append(poly.minimal_bounding_sphere.radius)
            shape_volume += shape_json_dicts[i]["4_Volume"] * sim_type_counts[i]
    
    spacing = 2.2 * max(radius)

    return sim_type_counts, type_shapes, typeid, shape_json_dicts, spacing, shape_volume

