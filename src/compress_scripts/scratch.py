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
output_dir = "../../data/IceVIII/"
cif = input_dir + "cif_files/141_H2O_0.cif"
walltime_stop = 60 * 60 # 1 hour in seconds

# get shape information
types = ["O", "H"]
N_scaling = 100


def get_shape_info(N_scaling, types, cif):
    atoms = ase.io.read(cif)
    type_list = Formula(str(atoms.symbols)).count()
    N_types = []
    type_shapes = [] 
    typeid = []
    shapes_info = []
    radius = []
    shape_volume = 0
    for i, type in enumerate(types): 
        shape_file = input_dir + f"shapes/shape_141_H2O_0_{type}_unit_volume_principal_frame.json"
        N_types.append(type_list[type] * N_scaling)
        typeid = typeid + ([i] * N_types[i])
        with open(shape_file) as file: 
            shapes_info.append(json.load(file))
            type_shapes.append(dict(type="ConvexPolyhedron", 
                                    rounding_radius = 0,
                                    vertices = shapes_info[i]["8_vertices"]))
            poly=coxeter.shapes.ConvexPolyhedron(shapes_info[i]["8_vertices"])
            radius.append(poly.minimal_bounding_sphere.radius)
            shape_volume += shapes_info[i] * N_types[i]
    
    spacing = 2.2 * max(radius)

    return atoms, type_list, N_types, type_shapes, typeid, shapes_info, spacing, shape_volume


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
    
    # get shape info 
    atoms, type_list, N_types, type_shapes, typeid, shapes_info, spacing, shape_volume = get_shape_info(N_scaling, types, cif)
    N = sum(N_types)
    print(spacing)
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

initialize() 

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

    simulation.operations.writers.remove(gsd_writer) #GSD files usually close when python script completes, mostly necessary if you run in a notebook
    del gsd_writer

