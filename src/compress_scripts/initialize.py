import math
import itertools 
import numpy as np
import gsd.hoomd
import hoomd 
from utils import get_shape_info, create_simulation 
import os 

def initialize_stability(job):
    """
    Initialize particle positions using unit cell
    -------------
    inputs: job (uses inputfile directory, gsd file, replicas)
    outputs: write particle positions and orientations into GSD file "initial_temp.gsd" (without shapes)
    """
    directory = job.statepoint.inputfile # "/Users/clarivi/Desktop/Research/IceCrystal/inputs/cif_files/temp_files/"
    gsdfile = directory + job.statepoint.gsd # "92_H2O_IceXI_nvt_traj_pf0p6_0.gsd"
    
    traj = gsd.hoomd.open(gsdfile, mode="r") 

    # INITIALIZE SIMULATION
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device = cpu, seed = 1)
    simulation.timestep = 0

    # create state from unit cell 
    simulation.create_state_from_gsd(filename = gsdfile, frame = 0)

    # create larger cell from unit cell (replication)
    replicas = job.statepoint.replicas #4
    simulation.state.replicate(nx = replicas, ny = replicas, nz = replicas)
    #hoomd.write.GSD.flush()

    hoomd.write.GSD.write(state = simulation.state, 
                        mode = "wb", 
                        filename = job.fn(f"initial_temp.gsd"))
    
def initialize_lattice(job): 
    """
    initialize particle positions on random lattice (using stoichiometry from unit cell)
    Note: uses get_shape_info from "utils" to get shape info from cif and atom types (get_shape_info working function)
    -------------------------
    inputs: job (uses job.sp.inputfile, job.sp.atoms, job.sp.replicas, job.sp.crystal_name)
    outputs: 
    """
    #initialize square lattice 
    sim_type_counts, type_shapes, typeid, _, spacing, _ = get_shape_info(job.sp.inputfile, 
                                                                         job.sp.replicas, 
                                                                         job.sp.atoms, 
                                                                         job.sp.crystal_name)
    N = sum(sim_type_counts) 

    K = math.ceil(N ** (1/3))
    L = K * spacing 

    x = np.linspace(-L/2, L/2, K, endpoint = False)
    position = list(itertools.product(x, repeat=3))
    position = position[0:N]

    orientation = [(1, 0, 0, 0)] * N #set orientation to all "vertical"

    #write initial configuration
    frame = gsd.hoomd.Frame()
    frame.particles.N = N
    frame.particles.position = position 
    frame.particles.orientation = orientation 
    frame.particles.typeid = typeid 
    frame.particles.types = job.sp.atoms
    frame.particles.type_shapes = type_shapes
    frame.configuration.box = [L, L, L, 0, 0, 0]

    with gsd.hoomd.open(name=job.fn("initial_temp.gsd"), 
                        mode="w") as f: 
        f.append(frame)

def initialize(*jobs):
    """
    initializes simulation for either stability test or self-assembly test (in lattice)
    -----------------------------
    inputs: 
    all jobs in a run 
    
    outputs: 
    initialize simulation, and gsd file "initialize.gsd" 
    """
    for job in jobs: 

        if job.sp.compression == True: 
            initialize_lattice(job)
        else:
            initialize_stability(job)

        _, _, _, shape_json_dicts, _, _ = get_shape_info(job.sp.inputfile, 
                                                         job.sp.replicas, 
                                                         job.sp.atoms, 
                                                         job.sp.crystal_name)

        simulation = create_simulation(filename = job.fn("initial_temp.gsd"), 
                                       frame = 0, 
                                       shapes = shape_json_dicts,
                                       atoms = job.sp.atoms)
        os.remove(job.fn("initial_temp.gsd")) #remove the initialized GSD without shape information 

        logger = hoomd.logging.Logger()
        logger.add(simulation.operations.integrator, quantities=["type_shapes"])

        if job.sp.compression == True: 
            # randomize positions 
            simulation.run(4000)
        else: 
            simulation.run(1)
 
        hoomd.write.GSD.write(state = simulation.state, 
                        mode = "wb", 
                        filename = job.fn(f"initialize.gsd"), 
                        logger = logger)

        return simulation