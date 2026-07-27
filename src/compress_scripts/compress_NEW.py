from utils import get_shape_info, create_simulation 
import hoomd
import os 
#import mpi4py.MPI

""" NEW ATTEMPT AT COMPRESSION CODE """ 
def compress(*jobs):
    """
    slow compression of initialized (and randomized) system 
    ---------------
    inputs: jobs (uses job.fn, job.sp.inputfile, 
    job.sp.replicas, job.sp.atoms, job.sp.crystal_name)

    outputs: "compressed.gsd" 
    """

    # Set walltime limits: 
    CLUSTER_JOB_WALLTIME_MINUTES = int(os.environ.get("ACTION_WALLTIME_IN_MINUTES", "60"))

    # Allow up to 5 minutes for Python to launch and files to be written at the end.
    HOOMD_RUN_WALLTIME_LIMIT_SECONDS = CLUSTER_JOB_WALLTIME_MINUTES * 60 - 300

    for job in jobs: 
        job.document["walltime_limit"] = HOOMD_RUN_WALLTIME_LIMIT_SECONDS
        
        _, _, _, shapes, _, shape_volume = get_shape_info(job.sp.inputfile, 
                                                            job.sp.replicas, 
                                                            job.sp.atoms, 
                                                            job.sp.crystal_name)
        
        simulation = create_simulation(job.fn("initialize.gsd"), 0, shapes = shapes, atoms = job.sp.atoms, communicator = None)
        logger = hoomd.logging.Logger()
        logger.add(simulation.operations.integrator, 
                   quantities=["type_shapes"])

        rho_init = shape_volume / simulation.state.box.volume 

        print(f"initial volume fraction = {rho_init}")
        job.document["initial_packingfrac"] = rho_init
        
        #set compress variables 
        initial_box = simulation.state.box 
        final_box = hoomd.Box.from_box(initial_box)
        final_volume_fraction = 0.6 # possible make into a job quantity 
        final_box.volume = shape_volume / final_volume_fraction

        # move tuner 
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=hoomd.trigger.Periodic(100),
            types=job.sp.atoms,
            max_rotation_move=0.5,
            max_translation_move=0.5
        )

        tune_equilib = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"], 
            target=0.3, 
            trigger = hoomd.trigger.And(
                [
                    hoomd.trigger.Periodic(100), 
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

            tune_appended = False
            if shape_volume/simulation.state.box.volume > 0.1: 
                #only tune once packing fraction is out of ideal gas regime 
                simulation.operations.tuners.append(tune) 
                tune_appended = True

            while not compress.complete: 
                simulation.run(100) 
                print(f"compressing, currently time {simulation.walltime}")
                next_walltime = simulation.device.communicator.walltime + simulation.walltime 
                if next_walltime >= HOOMD_RUN_WALLTIME_LIMIT_SECONDS:
                    print("Simulation timed out") 
                    hoomd.write.GSD.write(state= simulation.state, 
                                          mode = "wb", 
                                          filename = job.fn("timeout_config.gsd"))
                    break
            
            print("compression running ... current pf: ", (shape_volume / simulation.state.box.volume))

            simulation.operations.updaters.remove(compress)

            if tune_appended == True: 
                #only tune once packing fraction is out of ideal gas regime
                simulation.operations.remove(tune)     

            tune_equilib_appended = False
            if shape_volume / simulation.state.box.volume > 0.1:
                simulation.operations.tuners.append(tune_equilib)
                tune_equilib_appended = True
            initial_timestep = simulation.timestep 
        
            if (simulation.timestep - initial_timestep) < 50_000:
                simulation.run(1_000)

                next_walltime = simulation.device.communicator.walltime + simulation.walltime
                if next_walltime >= HOOMD_RUN_WALLTIME_LIMIT_SECONDS: 
                    print("simulation timed out")
                    hoomd.write.GSD.write(state= simulation.state, 
                                            mode = "wb", 
                                            filename = job.fn("timeout_config.gsd"))
                    # Save the timestep that compression completed
                    job.document["timeout_step"] = simulation.timestep
                    job.document["timeout_time"] = simulation.device.communicator.walltime 
                    job.document["timeout_pf"] = shape_volume / simulation.state.box.volume
                    break
            
            if tune_equilib_appended == True:
                simulation.operations.remove(tune_equilib)

        
        rho_final = shape_volume / simulation.state.box.volume 
        print(f"final volume fraction: {rho_final} in time {next_walltime}")

        hoomd.write.GSD.write(state=simulation.state, 
                              mode="wb", 
                              filename = job.fn("compress.gsd"), 
                              logger = logger)

        # Save the timestep that compression completed
        job.document["compressed_step"] = simulation.timestep
        job.document["compression_time"] = simulation.walltime
        job.document["final_pf"] = shape_volume / simulation.state.box.volume