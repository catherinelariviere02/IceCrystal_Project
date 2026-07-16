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
    
    #import hoomd 
    # Set walltime limits: 
    CLUSTER_JOB_WALLTIME_MINUTES = int(os.environ.get("ACTION_WALLTIME_IN_MINUTES", "60"))

    # Allow up to 10 minutes for Python to launch and files to be written at the end.
    HOOMD_RUN_WALLTIME_LIMIT_SECONDS = CLUSTER_JOB_WALLTIME_MINUTES * 60 - 300

    for job in jobs:
        _, _, _, shapes, _, _ = get_shape_info(job.sp.inputfile, 
                                                         job.sp.replicas, 
                                                         job.sp.atoms, 
                                                         job.sp.crystal_name)
        
        # self-assembly test   
        if job.sp.compression == True: 
            try:
                simulation = create_simulation(job.fn("compressed.gsd"), frame = 0, shapes = shapes, atoms = job.sp.atoms)

            except OSError as err:
                print("OS error:", err)
        
        # stability test
        else:
            simulation = create_simulation(job.fn("initialize.gsd"), frame = 0, shapes = shapes, atoms = job.sp.atoms)

        logger = hoomd.logging.Logger()
        logger.add(simulation.operations.integrator, 
                    quantities=["type_shapes"])

        #TUNING TRIAL MOVE SIZE 

        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=hoomd.trigger.And(
                [
                    hoomd.trigger.Periodic(10), 
                    hoomd.trigger.Before(simulation.timestep + 5_000)
                    ]
            ),
            types = job.sp.atoms
        )
        simulation.operations.tuners.append(tune)

        # floppy box

        boxmc = hoomd.hpmc.update.BoxMC(trigger=2, P=0.0)

        boxmc.aspect = {"weight": 0.7, "delta": 0.3}
        boxmc.shear = {"weight": 0.7, "delta": (0.3, 0.3, 0.3), "reduce": 0.0}

        boxmc_tune = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
            trigger=hoomd.trigger.And(
                [
                hoomd.trigger.Periodic(10), # Longer tune at start of run, plus a small tune on restart
                hoomd.trigger.Before(simulation.timestep + 5_000),
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
        sim_time = job.statepoint.runtime #10000
        log_time = int(sim_time/job.statepoint.logsteps)

        gsd_writer = hoomd.write.GSD(
            filename= job.fn("trajectory_temp.gsd"), #"../../data/iceIX/trajectory_temp.gsd" 
            trigger=hoomd.trigger.Periodic(log_time), 
            mode="wb", logger = logger)
        simulation.operations.writers.append(gsd_writer)
        
        t = time.time()

        print("starting equilibration...")

        if simulation.timestep < sim_time: 
            simulation.run(10_000)

            next_walltime = simulation.device.communicator.walltime + simulation.walltime
            if next_walltime >= HOOMD_RUN_WALLTIME_LIMIT_SECONDS: 
                print("simulation timed out")
                hoomd.write.GSD.write(state= simulation.state, 
                                          mode = "wb", 
                                          filename = job.fn("timeout_config.gsd"))
        
        walltime = simulation.device.communicator.walltime
        print(
            f"{job.id} ended on step {simulation.timestep} after {walltime} seconds"
        )

        if simulation.timestep == sim_time: 
            os.rename(job.fn("trajectory_temp.gsd"), job.fn("trajectory.gsd"))

        simulation.operations.writers.remove(gsd_writer) #GSD files usually close when python script completes, mostly necessary if you run in a notebook
        del gsd_writer