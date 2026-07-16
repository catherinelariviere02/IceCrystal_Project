### IMPORTS 
import math 
import hoomd 
import gsd.hoomd 
import freud
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import os 
import time
import ase.io
from ase.formula import Formula
import json 
import itertools
import coxeter

#parameters: 
input_dir = "../../inputs/"
output_dir = "../../data/IceIX/"
cif = input_dir + "cif_files/92_H2O_0.cif"
walltime_stop = 60 * 60 # 1 hour in seconds

# get shape information
types = ["O", "H"]
N_scaling = 50


def get_shape_info(N_scaling, types, crystal_name):
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
    cif = f"{crystal_name}.cif"
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

def initialize_from_uc(uc_file, N_scaling):
    gsdfile = uc_file

    cpu = hoomd.device.CPU()
    sim_temp = hoomd.Simulation(device = cpu, seed = 1)

    sim_temp.create_state_from_gsd(filename = gsdfile, frame = 0)

    # create larger cell from unit cell (replication)
    replicas = N_scaling
    sim_temp.state.replicate(nx = replicas, ny = replicas, nz = replicas)
    
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()

    #interate over all shapes (will be necessary for binary system)
    for i in range(len(shapes)):
        mc.shape[atoms[i]] = dict(
            vertices = shapes[i]["vertices"]
        )

    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=["type_shapes"])

    sim_temp.operations.integrator = mc

    sim_temp.run(1)

    hoomd.write.GSD.write(state = sim_temp.state, 
                        mode = "xb", 
                        filename = f"lattice.gsd", 
                        logger = logger)

    hoomd.write.GSD.write(state=sim_temp.state, mode="wb", filename="lattice.gsd")

def initialize_lattice(spacing, N, typeid, type_shapes): 
    #initialize square lattice 
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
    frame.particles.types = types
    frame.particles.type_shapes = type_shapes
    frame.configuration.box = [L, L, L, 0, 0, 0]

    with gsd.hoomd.open(name=output_dir + "lattice.gsd", 
                        mode="w") as f: 
        f.append(frame)

    return frame

def initialize():
    types = ["O", "H"]
    N_scaling = 100
    unitcell = True 
    uc_file = input_dir + "shapes/92_H2O_1/92_H2O_IceXI_nvt_final_pf0p6_0.gsd"
    
    # get shape info 
    N_types, type_shapes, typeid, shapes_info, spacing, shape_volume = get_shape_info(N_scaling, types, cif)
    N = sum(N_types)
    print(spacing)

    if unitcell == True:
        initialize_from_uc(uc_file, N_scaling)
    else:
        frame = initialize_lattice(spacing, N, typeid, type_shapes)

    # initialize simulation
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed = 1)

    # initialize integrator 
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    mc.nselect = 2 #number of trials moves per particle per timestep

    for i, type in enumerate(types):
        mc.shape[type] = dict(vertices=shapes_info[i]["8_vertices"])
        mc.a[type] = 1
        mc.d[type] = 1

    simulation.operations.integrator = mc 
    simulation.create_state_from_gsd(filename = output_dir + "lattice.gsd")

    simulation.run(1000)

    return simulation 

#initialize() 

# def tuner():


def compress(walltime, output_dir):
    simulation = initialize()




def equilibrate(): 
    simulation = initialize()

    logger = hoomd.logging.Logger()
    logger.add(simulation.operations.integrator, 
                quantities=["type_shapes"])

    #TUNING TRIAL MOVE SIZE 

    tune = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["a", "d"],
        target=0.2,
        trigger=hoomd.trigger.And(
            [
                hoomd.trigger.Periodic(100), 
                hoomd.trigger.Before(simulation.timestep + 5000)
                ]
        ),
    )
    simulation.operations.tuners.append(tune)

    # # FLOPPY BOX 
    sim_time = 10000
    log_time = int(sim_time/10)

    boxmc = hoomd.hpmc.update.BoxMC(trigger=2, P=0.0)

    boxmc.aspect = {"weight": 0.7, "delta": 0.3}
    boxmc.shear = {"weight": 0.7, "delta": (0.3, 0.3, 0.3), "reduce": 0.0}

    boxmc_tune = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.And(
            [
            hoomd.trigger.Periodic(10), # Longer tune at start of run, plus a small tune on restart
            hoomd.trigger.Before(simulation.timestep + 10_000),
            ]
    ),
    boxmc=boxmc,
    moves=["aspect", "shear_x", "shear_y", "shear_z"],
    max_move_size={
    "aspect": 0.3,
    "shear_x": 0.3,
    "shear_y": 0.3,
    "shear_z": 0},
    target=0.3,
    )

    logger.add(boxmc, ["shear_moves", "aspect_moves"])
    simulation.operations.updaters.append(boxmc)
    simulation.operations.tuners.append(boxmc_tune)

    # ADD WRITER (AFTER LOGGER IS FULLY CONSTRUCTED)

    gsd_writer = hoomd.write.GSD(
        filename=output_dir + "trajectory_temp.gsd", 
        trigger=hoomd.trigger.Periodic(log_time), 
        mode="xb", logger = logger)
    simulation.operations.writers.append(gsd_writer)
    
    t = time.time()

    print("starting equilibration...")

    simulation.run(sim_time)
    
    print(f"ran {sim_time} steps in {time.time() - t}")

    os.rename(output_dir + "trajectory_temp.gsd", output_dir + "trajectory.gsd")

    simulation.operations.writers.remove(gsd_writer) #GSD files usually close when python script completes, mostly necessary if you run in a notebook
    del gsd_writer

equilibrate() 


""" NEW ATTEMPT AT INITIALIZATION CODE """

def initialize_stability(job):
    directory = job.statepoint.inputfile # "/Users/clarivi/Desktop/Research/IceCrystal/inputs/cif_files/temp_files/"
    gsdfile = directory + job.statepoint.gsd # "92_H2O_IceXI_nvt_traj_pf0p6_0.gsd"
    atoms = job.statepoint.atoms # ["O", "H"] #atom names in order they are logged in "shapes"
    
    traj = gsd.hoomd.open(gsdfile, mode="r") 
    shapes = traj[0].particles.type_shapes 

    # INITIALIZE SIMULATION
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device = cpu, seed = 1)
    simulation.timestep = 0

    # create state from unit cell 
    simulation.create_state_from_gsd(filename = gsdfile, frame = 0)

    # create larger cell from unit cell (replication)
    replicas = job.statepoint.replicas #4
    simulation.state.replicate(nx = replicas, ny = replicas, nz = replicas)

    hoomd.write.GSD.flush()

    hoomd.write.GSD.write(state = simulation.state, 
                        mode = "wb", 
                        filename = job.fn(f"initial_temp.gsd"))


def initialize_lattice(job, spacing, N, typeid, type_shapes): 
    #initialize square lattice 
    
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

def create_simulation(filename, frame, shapes, atoms):
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device = cpu, seed = 1)
    
    simulation.create_state_from_gsd(filename, frame)

    mc = hoomd.hpmc.integrate.ConvexPolyhedron 

    for i, shapes in shapes: 
        mc.shape[atoms[i]] = dict(
            vertices = shapes[i]["vertices"]
        ) 

    simulation.operations.integrator.mc 

    return simulation

def initialize(*jobs):
    for job in jobs: 
        if job.sp.compress == True: 
            initialize_lattice()
        else:
            initialize_stability() 
        
        cif = job.sp.cif

        _, _, _, shape_json_dicts, _, _ = get_shape_info(job.sp.replicas, job.sp.atoms, job.sp.cif)

        simulation = create_simulation(filename = "initial_temp.gsd", 
                                       frame = 0, 
                                       shapes = shape_json_dicts,
                                       atoms = job.sp.atoms)
        os.remove(job.fn("initial_temp.gsd")) #remove the initialized GSD without shape information 

        logger = hoomd.logging.Logger()
        logger.add(simulation.operations.integrator, quantities=["type_shapes"])

        simulation.run(1)

        hoomd.write.GSD.flush()
        hoomd.write.GSD.write(state = simulation.state, 
                        mode = "wb", 
                        filename = job.fn(f"initialize.gsd"), 
                        logger = logger)

        return simulation 


""" NEW ATTEMPT AT COMPRESSION CODE """ 
def compress(*jobs):
    """
    slow compression of initialized (and randomized) system 
    ---------------
    inputs: jobs (uses job.fn, job.sp.inputfile, 
    job.sp.replicas, job.sp.atoms, job.sp.crystal_name, job.sp.walltime (new!))

    outputs: "compressed.gsd" 
    """
    for job in jobs: 
        _, _, _, shapes, _, shape_volume = get_shape_info(job.sp.inputfile, 
                                                            job.sp.replicas, 
                                                            job.sp.atoms, 
                                                            job.sp.crystal_name)
        
        simulation = create_simulation("initialize.gsd", 0, shapes = shapes, atoms = job.sp.atoms)
        logger = hoomd.logging.Logger()
        logger.add(simulation.operations.integrator, 
                   quantities=["type_shapes"])

        rho_init = shape_volume / simulation.state.box.volume 

        print(f"initial volume fraction = {rho_init}")

        #set compress variables 
        initial_box = simulation.state.box 
        final_box = hoomd.Box.from_box(initial_box)
        final_volume_fraction = 0.6 # possible make into a job quantity 
        final_box.volume = shape_volume / final_volume_fraction

        # move tuner 
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=hoomd.trigger.Periodic(10),
            types=job.sp.atoms,
            max_rotation_move=0.5,
            max_translation_move=0.5
        )

        tune_equilib = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"], 
            target=0.3, 
            trigger = hoomd.trigger.And(
                [
                    hoomd.trigger.Periodic(10), 
                    hoomd.trigger.Before(simulation.timestep + 5000)
                ]
            ), 
            types=job.sp.atoms
        )
        #slow compress  
        while simulation.state.box.volume > final_box.volume: 
            new_box = hoomd.Box.from_box(initial_box) 
            new_box.volume = simulation.state.box.volume * 0.99 # shrink by 1% of volume 
            
            compress = hoomd.hpmc.update.QuickCompress(trigger = hoomd.trigger.Periodic(10), 
                                                       target_box=new_box)
            simulation.operations.updaters.append(compress)

            if shape_volume/simulation.state.box.volume > 0.1: 
                #only tune once packing fraction is out of ideal gas regime 
                simulation.operations.tuners.append(tune) 

            while not compress.complete: 
                simulation.run(100) 
                next_walltime = simulation.device.communicator.walltime + simulation.walltime 
                if next_walltime >= job.sp.walltime:
                    print("Simulation timed out") 
                    hoomd.write.GSD.write(state= simulation.state, 
                                          mode = "wb", 
                                          filename = job.fn("timeout_config.gsd"))
                    break

            simulation.operations.updaters.remove(compress)

            if shape_volume / simulation.state.box.volume > 0.1: 
                #only tune once packing fraction is out of ideal gas regime
                simulation.operations.tuners.remove(tune)    

            if shape_volume / simulation.state.box.volume > 0.1:
                simulation.operations.tuners.append(tune_equilib)
            
            simulation.run(50_000)

            if shape_volume / simulation.state.box.volume > 0.1:
                simulation.operations.tuners.remove(tune_equilib)

            next_walltime = simulation.device.communicator.walltime + simulation.walltime
            if next_walltime >= job.sp.walltime: 
                print("simulation timed out")
                hoomd.write.GSD.write(state= simulation.state, 
                                          mode = "wb", 
                                          filename = job.fn("timeout_config.gsd"))
                break 
        
        rho_final = shape_volume / simulation.state.box.volume 
        print(f"final volume fraction: {rho_final}")

        hoomd.write.GSD.write(state=simulation.state, 
                              mode="wb", 
                              filename = job.fn("compressed.gsd"), 
                              logger = logger)