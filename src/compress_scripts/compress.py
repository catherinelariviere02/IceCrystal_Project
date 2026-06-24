"""
HARD PARTICLE SELF ASSEMBLY SLOW-COMPRESS CODE 
----------------------------------------------
Takes crystal2shape shapes and runs monte carlo slow-compress of particles to test self-assembly. 
"""

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

### PARAMETERS 
input_dir = "../../inputs/"
output_dir = "../../data/IceVIII/"

walltime_stop = 60 * 60 * 16 # * 60 # 1 hour in seconds

## crystallographic parameters
types = ["O", "H"]
cif = input_dir + "cif_files/141_H2O_0.cif"
shape_files = []
for type in types: 
    shape_files.append(input_dir + f"shapes/shape_141_H2O_0_{type}_unit_volume_principal_frame.json")

# get number of atoms 
atoms = ase.io.read(cif)
type_list = Formula(str(atoms.symbols)).count()

# calculate number of each shape in initial simulation
N_scaling = 100
N_types = [] # number of each type of shape

for i, type in enumerate(types): #get number
    N_types.append(type_list[type] * N_scaling)

N = sum(N_types)
print(f"Number of particles {N}")

#initialize HPMC  
mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.nselect = 2 #number of trials moves per particle per timestep
type_shapes = []
for i, shape_file in enumerate(shape_files):
    with open(shape_file) as file:

        shape_info = json.load(file)
        mc.shape[types[i]] = dict(vertices=shape_info["8_vertices"])
        mc.d[types[i]] = 0.2
        mc.a[types[i]] = 0.2

        type_shapes.append(dict(type="ConvexPolyhedron", 
                                rounding_radius = 0,
                                vertices = shape_info["8_vertices"]))

logger = hoomd.logging.Logger()
logger.add(mc, quantities=["type_shapes"])

#initialize square lattice 
spacing = 5
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

#set type 
typeid = []
for i in range(len(types)):
    typeid = typeid + ([i] * N_types[i])

frame.particles.typeid = typeid 
frame.particles.types = types
frame.particles.type_shapes = type_shapes

frame.configuration.box = [L, L, L, 0, 0, 0]

with gsd.hoomd.open(name=output_dir + "lattice.gsd", 
                    mode="w") as f: 
    f.append(frame)

### RANDOMIZE SIMULATION 

#initialize simulation 
cpu = hoomd.device.CPU()
simulation = hoomd.Simulation(device=cpu, seed = 1)

simulation.operations.integrator = mc 
simulation.create_state_from_gsd(filename = output_dir + "lattice.gsd")

simulation.run(1)
print(f"initial overlaps: {mc.overlaps}")
print(simulation.timestep)
ti = time.time()
print("starting randomization")
simulation.run(1000)
print(f"ending randomization in {time.time()-ti}")
print(simulation.timestep)

simulation.run(100)
print("rotation acceptance ", mc.rotate_moves[0]/sum(mc.rotate_moves))

print("translation acceptance ", mc.translate_moves[0] / sum(mc.rotate_moves))

hoomd.write.GSD.write(state=simulation.state, mode="wb", 
                      filename=output_dir + "random.gsd", 
                      logger = logger) 

### SLOW COMPRESS BOX 

# calculate initial packing fraction 
shape_volume = 0
for i, shape_file in enumerate(shape_files):
    with open(shape_file) as file:
        shape_info = json.load(file)
        shape_volume += shape_info["4_Volume"] * N_types[i]
        print(f"shape volume {types[i]} is {shape_volume}")

rho_init = shape_volume / simulation.state.box.volume

print(f"initial volume fraction = {rho_init}")

#set compress variables 
initial_box = simulation.state.box 
final_box = hoomd.Box.from_box(initial_box)
final_volume_fraction = 0.2
final_box.volume = shape_volume / final_volume_fraction

#add move tuners 
tune = hoomd.hpmc.tune.MoveSize.scale_solver(
    moves=["a", "d"],
    target=0.3,
    trigger=hoomd.trigger.Periodic(10),
    types=types,
    max_rotation_move=0.3,
    max_translation_move=10
)
# simulation.operations.tuners.append(tune)

# compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10), target_box=final_box)
# simulation.operations.updaters.append(compress)

# # run code 
# print("simulation timestep before compression = ", simulation.timestep)
# print(compress.complete)

# simulation.run(1)
# print("initial trial move size")
# print("rotation ", mc.a["O"])
# print("translation ", mc.d["O"])

# t2 = time.time()
# while not compress.complete and simulation.timestep < 1e6: 
#     t1 = time.time()
#     simulation.run(100)
#     print(f"compression running...{time.time() - t1}")
#     print(f"current volume = {shape_volume / simulation.state.box.volume}")
#     print(f"time {time.time() - t2}")
#     print("rotation ", mc.a["O"])
#     print("translation ", mc.d["O"])

#     print("rotation acceptance = ", mc.rotate_moves[0]/sum(mc.rotate_moves))

#     print("translation acceptance = ", mc.translate_moves[0] / sum(mc.rotate_moves))

# print(f"time step: {simulation.timestep}")
# print("rotation ", mc.a["O"])
# print("translation ", mc.d["O"])

# if not compress.complete:
#     message = "Compression failed to complete"
#     raise RuntimeError(message)

# simulation.operations.updaters.remove(compress)

# slow compress - quick compress with equilibration 
final_box = hoomd.Box.from_box(initial_box)
final_volume_fraction = 0.6
final_box.volume = shape_volume / final_volume_fraction

print(f"initial volume fraction : {shape_volume / simulation.state.box.volume}")
print(f"final goal volume fraction: {shape_volume / final_box.volume}")

while simulation.state.box.volume > final_box.volume:
    # add quick compress to 99% of volume 
    new_box = hoomd.Box.from_box(initial_box)
    new_box.volume = simulation.state.box.volume * 0.99 # shrink by 0.99
    print("initial volume fraction ", shape_volume / simulation.state.box.volume) 
    print("goal volume fraction ", shape_volume / new_box.volume)

    compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10), target_box=new_box)
    simulation.operations.updaters.append(compress)
    simulation.operations.tuners.append(tune)
    # run until compressed to desired volume
    while not compress.complete: 
        simulation.run(100)
        print(f"current volume = {shape_volume / simulation.state.box.volume}")
        print(f"compress completeness: {compress.complete}")
        print(simulation.device.communicator.walltime)
        next_walltime = simulation.device.communicator.walltime + simulation.walltime
        if next_walltime >= walltime_stop: 
            print("Simulation timed out")
            break
    
    # check new volume (to make sure it worked) 
    print(f"quick compress complete to {shape_volume/simulation.state.box.volume}")

    print("rotation acceptance = ", mc.rotate_moves[0]/sum(mc.rotate_moves))

    print("translation acceptance = ", mc.translate_moves[0] / sum(mc.rotate_moves))

    # remove quick compress 
    simulation.operations.updaters.remove(compress)
    simulation.operations.tuners.append(tune)

    tune_equilib = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["a", "d"],
        target=0.5,
        trigger=hoomd.trigger.And(
            [ 
                hoomd.trigger.Periodic(10), 
                hoomd.trigger.Before(simulation.timestep + 5000)
                ]
        ),
        types=types
    )
    
    # equilibrate (run for 10,000 steps, for now) 
    simulation.operations.tuners.append(tune_equilib)
    simulation.run(50_000)
    simulation.operations.tuners.remove(tune_equilib)

    # add simulation break for equilibration
    print(simulation.device.communicator.walltime)
    next_walltime = simulation.device.communicator.walltime + simulation.walltime
    if next_walltime >= walltime_stop: 
        print("Simulation timed out")
        break

rho_final = shape_volume / simulation.state.box.volume
print(f"final volume fraction: {rho_final}")

hoomd.write.GSD.write(state=simulation.state, mode="wb", filename = output_dir + "compressed.gsd", logger = logger)

### EQUILIBRATE BOX - million time steps 

# floppy box

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

equilib_time = 10e7

while simulation.timestep < equilib_time:
    simulation.run(1000)
    if next_walltime >= walltime_stop: 
        print("Simulation timed out")
        break