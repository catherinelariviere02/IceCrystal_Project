import freud
import matplotlib
import hoomd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import gsd.hoomd
import plotly.graph_objects as go
from utils import create_simulation, get_shape_info
import os 
import time

#necessary job information: simulation length, number of log points, output file 
def equilibrate(*jobs):
    for job in jobs:
        _, _, _, shapes, _, _ = get_shape_info(job.sp.inputfile, 
                                                         job.sp.replicas, 
                                                         job.sp.atoms, 
                                                         job.sp.crystal_name)
        
        #possibly write if statement for compression or stability test 
        simulation = create_simulation(job.fn("initialize.gsd"), frame = 0, shapes = shapes, atoms = jobs.sp.atoms)

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

        # ADD WRITER (AFTER LOGGER IS FULLY CONSTRUCTED)
        sim_time = job.statepoint.runtime #10000
        log_time = int(sim_time/job.statepoint.logsteps)

        gsd_writer = hoomd.write.GSD(
            filename= job.fn("trajectory_temp.gsd"), #"../../data/iceIX/trajectory_temp.gsd" 
            trigger=hoomd.trigger.Periodic(log_time), 
            mode="wb", logger = logger)
        simulation.operations.writers.append(gsd_writer)
        
        t = time.time()

        print("starting equilibration...")

        simulation.run(sim_time)
        
        print(f"ran {sim_time} steps in {time.time() - t}")

        os.rename(job.fn("trajectory_temp.gsd"), job.fn("trajectory.gsd"))

        simulation.operations.writers.remove(gsd_writer) #GSD files usually close when python script completes, mostly necessary if you run in a notebook
        del gsd_writer