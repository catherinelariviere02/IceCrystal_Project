import freud
import matplotlib
import hoomd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import gsd.hoomd
import plotly.graph_objects as go
from initialize import create_simulation
import os 
import time

def equilibrate():
    
    simulation = create_simulation()

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

    # boxmc = hoomd.hpmc.update.BoxMC(trigger=2, P=0.0)

    # boxmc.aspect = {"weight": 0.7, "delta": 0.3}
    # boxmc.shear = {"weight": 0.7, "delta": (0.3, 0.3, 0.3), "reduce": 0.0}

    # boxmc_tune = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
    #     trigger=hoomd.trigger.And(
    #         [
    #         hoomd.trigger.Periodic(10), # Longer tune at start of run, plus a small tune on restart
    #         hoomd.trigger.Before(simulation.timestep + 10_000),
    #         ]
    # ),
    # boxmc=boxmc,
    # moves=["aspect", "shear_x", "shear_y", "shear_z"],
    # max_move_size={
    # "aspect": 0.3,
    # "shear_x": 0.3,
    # "shear_y": 0.3,
    # "shear_z": 0},
    # target=0.3,
    # )

    # logger.add(boxmc, ["shear_moves", "aspect_moves"])
    # simulation.operations.updaters.append(boxmc)
    # simulation.operations.tuners.append(boxmc_tune)

    # ADD WRITER (AFTER LOGGER IS FULLY CONSTRUCTED)
    sim_time = 10000
    log_time = int(sim_time/10)

    gsd_writer = hoomd.write.GSD(
        filename="../../data/iceVIII/trajectory_temp.gsd", 
        trigger=hoomd.trigger.Periodic(log_time), 
        mode="xb", logger = logger)
    simulation.operations.writers.append(gsd_writer)
    
    t = time.time()

    print("startin equilibration...")

    simulation.run(sim_time)
    
    print(f"ran {sim_time} steps in {time.time() - t}")

    #check acceptance rates: 

    simulation.run(100)

    simulation.operations.writers.remove(gsd_writer) #GSD files usually close when python script completes, mostly necessary if you run in a notebook
    del gsd_writer