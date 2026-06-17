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

### PARAMETERS 
input_dir = "../../inputs/"
output_dir = "../../data/IceVIII/"

## crystallographic parameters
types = ["O", "H"]
cif = input_dir + "cif_files/141_H2O_0.cif"
shape_files = []
for type in types: 
    shape_files.append(input_dir + f"shapes/shape_141_H2O_0_{type}_unit_volume_principal_frame.json")
print(shape_files)

# get number of atoms 
atoms = ase.io.read(cif) 
type_list = Formula(str(atoms.symbols)).count()

# calculate number of each shape in initial simulation
N_scaling = 20
N_types = [] # number of each type of shape

for i, type in enumerate(types): #get number
    N_types.append(type_list[type] * N_scaling)

N = sum(N_types)
print(f"Number of particles {N}")

#initialize HPMC  
mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.nselect = 4 #number of trials moves per particle per timestep
type_shapes = []
for i, shape_file in enumerate(shape_files):
    with open(shape_file) as file:

        shape_info = json.load(file)
        print(f"type of particle {types[i]}")
        mc.shape[types[i]] = dict(
            vertices = shape_info["8_vertices"]
        )

        type_shapes.append(dict(type="ConvexPolyhedron", 
                                rounding_radius = 0,
                                vertices = shape_info["8_vertices"]))

logger = hoomd.logging.Logger()
logger.add(mc, quantities=["type_shapes"])

#initialize square lattice 
spacing = 3
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

print(f"length of typeid is {len(typeid)}")
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

ti = time.time()
print("starting randomization")
simulation.run(10e3)
print(f"ending randomization in {time.time()-ti}")

# query mc move properties to see what it did
print(mc.translate_moves) #tuple of accepted and rejected moves 
print(mc.translate_moves[0] / sum(mc.translate_moves)) #acceptance ratio

print(mc.rotate_moves) # tuple of accepted and rejected moves 
print(mc.rotate_moves[0] / sum(mc.rotate_moves))

print(mc.overlaps)

hoomd.write.GSD.write(state=simulation.state, mode="wb", 
                      filename=output_dir + "random.gsd", 
                      logger = logger) 

### SLOW COMPRESS BOX 


### EQUILIBRATE BOX - million time steps 
