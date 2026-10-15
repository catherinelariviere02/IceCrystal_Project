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
import h5py
 
#necessary job information: simulation length, number of log points, output file 
def equilibrate(*jobs):
    import hoomd 
    # Set walltime limits: 
    CLUSTER_JOB_WALLTIME_MINUTES = int(os.environ.get("ACTION_WALLTIME_IN_MINUTES", "60"))

    # Allow up to 10 minutes for Python to launch and files to be written at the end.
    HOOMD_RUN_WALLTIME_LIMIT_SECONDS = CLUSTER_JOB_WALLTIME_MINUTES * 60 - 600
    print(HOOMD_RUN_WALLTIME_LIMIT_SECONDS)
    processes_per_directory = int(os.environ['ACTION_PROCESSES_PER_DIRECTORY'])
    communicator = hoomd.communicator.Communicator(ranks_per_partition=processes_per_directory)
    job = jobs[communicator.partition]
    
    with job as job:
        device = hoomd.device.CPU(communicator=communicator)

        device.notice(f"job id {job.id}, rank {communicator.rank}")
        #if communicator.rank == 0:
        _, _, _, shapes, _, _ = get_shape_info(job.sp.inputfile, 
                                                        job.sp.replicas, 
                                                        job.sp.atoms, 
                                                        job.sp.crystal_name,
                                                        job.sp.stoich)
        #device.notice("got shape!")

        # self-assembly test   
         
        if job.sp.compression == True: 
            sim_time = job.statepoint.runtime + job.document["compressed_step"]
            # start from timeout_config for equilibration if it exists, otherwise 
            if os.path.isfile(job.fn("restart.gsd")) == True: 
                simulation = create_simulation(job.fn("trajectory_temp.gsd"),
                                                                frame = -1,
                                                                shapes = shapes, 
                                                                atoms = job.sp.atoms, 
                                                                updated = False if "updated" not in job.sp.inputfile else True,
                                                                communicator = communicator)
            else:
                try:
                    simulation = create_simulation(job.fn("compress.gsd"),
                                                frame = 0,
                                                shapes = shapes, 
                                                atoms = job.sp.atoms, 
                                                updated = False if "updated" not in job.sp.inputfile else True,
                                                communicator = communicator)
                except OSError as err:
                    print("OS error:", err)
        
        # stability test
        else:
            sim_time = job.statepoint.runtime
            if os.path.isfile(job.fn("restart.gsd")) == True: 
                            simulation = create_simulation(job.fn("trajectory_temp.gsd"),
                                                                            frame = -1,
                                                                            shapes = shapes, 
                                                                            atoms = job.sp.atoms, 
                                                                            updated = False if "updated" not in job.sp.inputfile else True,
                                                                            communicator = communicator)
            else:
                simulation = create_simulation(job.fn("initialize.gsd"), 
                                               frame = 0, 
                                               shapes = shapes, 
                                               atoms = job.sp.atoms, 
                                               updated = False if "updated" not in job.sp.inputfile else True,
                                               communicator = communicator)
        
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
        log_time = int(job.statepoint.runtime/(job.statepoint.logsteps*10))

        #device.notice to only print on one rank 
        # device.notice(f"{log_time}")
        # device.notice(f"{sim_time}")

        # write  
        # log_time = 100 # DEBUGGING, CHANGE AND DELETE LATER 
        gsd_writer = hoomd.write.GSD(
            filename= job.fn("trajectory_temp.gsd"), #"../../data/iceIX/trajectory_temp.gsd" 
            trigger=hoomd.trigger.Periodic(log_time),
            mode="ab", logger = logger)

        # dynamically update image if specified 
        if job.sp.image_write == True: 
            gsd_writer.dynamic = ["property", "particles/image"]

        device.notice("checking progress...")
        simulation.operations.writers.append(gsd_writer)

        device.notice(f"starting equilibration...{simulation.timestep} is {simulation.timestep < sim_time}")

        while simulation.timestep < sim_time: 
            # device.notice("equilibrating ... ") 
            simulation.run(min(1000, sim_time))
            # device.notice(f"mps: {simulation.operations.integrator.mps}")
            # c = simulation.operations.integrator.counters
            # device.notice(f"overlap checks per trial ove: {c.overlap_checks / (sum([*c.translate, *c.rotate]))}")

            # t_accept = (simulation.operations.integrator.translate_moves[0]
            #              / sum(simulation.operations.integrator.translate_moves))
            # device.notice(f"translate acceptance {t_accept}")
            # r_accept = (simulation.operations.integrator.rotate_moves[0]
            #                          / sum(simulation.operations.integrator.rotate_moves))
            # device.notice(f"rotate acceptance {r_accept}")

            # break
            next_walltime = simulation.device.communicator.walltime + simulation.walltime
            if next_walltime >= HOOMD_RUN_WALLTIME_LIMIT_SECONDS: 
                print("simulation timed out")
                hoomd.write.GSD.write(state = simulation.state, 
                                          mode = "wb", 
                                          filename = job.fn("restart.gsd"))

        walltime = simulation.device.communicator.walltime
        device.notice(
            f"{job.id} ended on step {simulation.timestep} after {walltime} seconds"
        )

        job.document["equilib_time"] = walltime

        if device.communicator.rank == 0:
            if simulation.timestep >= sim_time: 
                os.rename(job.fn("trajectory_temp.gsd"), job.fn("trajectory.gsd"))

        simulation.operations.writers.remove(gsd_writer) #GSD files usually close when python script completes, mostly necessary if you run in a notebook
        del gsd_writer
