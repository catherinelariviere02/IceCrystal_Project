import numpy as np
import hoomd
import gsd.hoomd
import os
import coxeter

from ..hoomd_hpmc.hpmc_integrator import create_simulation

class SpringUpdaterAction(hoomd.custom.Action):
    def __init__(self, springs, variant, final_scale_factor):
        """Make variant go from 0 to 1"""
        self.springs = springs
        self.variant = variant
        self.final_scale = final_scale_factor
        self.initial_k_translational = self.springs.k_translational

    @hoomd.logging.log(requires_run=True)
    def k_translational(self):
        return self.springs.k_translational(self._timestep)

    def act(self, timestep):
        self._timestep = timestep

def apply_gaussianPotential_func(shape_data, patch_info):
    """
    Apply a Gaussian potential to the shape data.
    
    """

    # Set parameters for Expanded Gaussian potential
    patch_opening_angles = [patch_info[0] for _t1 in shape_data.keys()]
    epsilon, lambda_, delta = patch_info[1], patch_info[2], patch_info[0]

    # Define Expanded Gaussian potential
    # expanded_gaussian = hoomd.hpmc.pair.ExpandedGaussian(default_r_cut=5, mode="shift")

    # Define Step potential to be used as isotropic part of AngularStep
    isotropic_potential = hoomd.hpmc.pair.Step()

    for i, typeA in enumerate(shape_data.keys()):
        for j, typeB in enumerate(shape_data.keys()):
            type_keyA = list(shape_data.keys())[i]
            type_keyB = list(shape_data.keys())[j]
            shape_vertices_A = shape_data[type_keyA]["Vertices"]
            shape_vertices_B = shape_data[type_keyB]["Vertices"]
            poly_typeA = coxeter.shapes.ConvexPolyhedron(vertices=shape_vertices_A)
            poly_typeB = coxeter.shapes.ConvexPolyhedron(vertices=shape_vertices_B)
            # Calculate minimum distance between two polyhedra as r_cut
            sigma = (poly_typeA.minimal_centered_bounding_sphere_radius + poly_typeB.minimal_centered_bounding_sphere_radius)
            isotropic_potential.params[(typeA, typeB)] = dict(
                epsilon=[0, -epsilon], r=[sigma, sigma*lambda_]
            )

    # Apply Angular Step potential with patches
    patch_potential = hoomd.hpmc.pair.AngularStep(isotropic_potential=isotropic_potential)

    for _t1 in range(len(shape_data.keys())):
        type_key = list(shape_data.keys())[_t1]
        shape_vertices = shape_data[type_key]["Vertices"]
        poly = coxeter.shapes.ConvexPolyhedron(vertices=shape_vertices)
        faces = poly.faces
        face_vertices = [[poly.vertices[f] for f in face] for face in faces] # Vertices of each face
        face_midpoints = [np.mean(face, axis=0) for face in face_vertices] # Face midpoints
        face_normals = [f_midpt/np.linalg.norm(f_midpt) for f_midpt in face_midpoints] # Face normals
        directors = [tuple(f) for f in face_normals]

        patch_potential.mask[type_key] = dict(directors=directors, deltas=[np.deg2rad(patch_opening_angles[_t1]) for _ in range(len(directors))])

    return patch_potential

def nvt(directories, N_particles, shape_data, communicator, tag, floppy_nvt=False, target_pf=0.6, translational_spring="False",
        end_step=1e5, write_period=1e3):

    # Shape data info
    shape_data_keys, shape_data_values = list(shape_data.keys()), list(shape_data.values())
    shape_vertices, types, volume, num_particles = [], [], [], []
    for i in range(len(shape_data_keys)):
        shape_vertices.append(shape_data_values[i]["Vertices"])
        types.append(shape_data_keys[i])
        volume.append(shape_data_values[i]["Volume"])
        num_particles.append(shape_data_values[i]["Num_particles"])

    if translational_spring == "True":
        move_sizes = [[0.05, 0.0001] for _ in range(len(types))]
    else:
        move_sizes = [[0.05, 0.05] for _ in range(len(types))]

    V_particle = [n * volume[i] for i, n in enumerate(num_particles)] # Volume per particle

    final_pf_string = str(target_pf).replace(".", "p") # String of final packing fraction float number
    # Create simulation
    simulation = create_simulation(communicator, shape_vertices, types, move_sizes)
    mc = simulation.operations.integrator

    # Read input GSD
    state_id = 0
    if os.path.exists(directories + tag + "_pf" + final_pf_string + "_" +str(0)+".gsd"):
        c = 1
        while c < 100:
            output_check = directories + tag + "_pf" + final_pf_string + "_" +str(c)+".gsd"
            if os.path.exists(output_check):
                c = c + 1
                continue
            else:
                break

        state_id = c
        input_GSD_filename = directories + tag + "_pf" + final_pf_string + "_" +str(state_id-1)+".gsd"
        simulation.create_state_from_gsd(filename=input_GSD_filename, frame=-1)

    elif os.path.exists(directories + tag + "_nvt_restart.gsd"):
        # Read the final system configuration from a previous execution.
        input_GSD_filename = directories + tag + "_nvt_restart.gsd"
        simulation.create_state_from_gsd(filename=input_GSD_filename, frame=-1)

    else:
        # Or read `compressed.gsd` for the first execution of equilibrate.
        input_GSD_filename = directories + tag + "_compressed_to_pf" + final_pf_string + "_final.gsd"
        simulation.create_state_from_gsd(filename=input_GSD_filename, frame=-1)

    #**********************************************************************
    # Move size Tuner
    if translational_spring == "True":
        periodic = hoomd.trigger.Periodic(10)
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a"],
            target=0.2,
            trigger=periodic,
            types=types,
            max_rotation_move=0.5,
            )
        simulation.operations.tuners.append(tune)
    else:
        periodic = hoomd.trigger.Periodic(10)
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.2,
            trigger=periodic,
            types=types,
            max_rotation_move=0.5,
            )
        simulation.operations.tuners.append(tune)

    # Patchy potential
    '''patch_info = [45, 1.0, 2.0, 1.0]
    patch_potential = apply_gaussianPotential_func(shape_data, patch_info)
    simulation.operations.integrator.pair_potentials = [patch_potential]'''

    if translational_spring == "True":
        # Spring variant for harmonic field
        scale_variant = hoomd.variant.Power(
                # Only decrease spring strenght to 800 - need to strongly maintain structure
                A=500_000,
                B=500_000,
                power=1,
                t_start=simulation.timestep,
                t_ramp=1,
        )
        lattice = gsd.hoomd.open(input_GSD_filename)[-1]

        # Harmonic updater
        harmonic_field = hoomd.hpmc.external.Harmonic(
            reference_positions=lattice.particles.position,
            reference_orientations=lattice.particles.orientation,  # Wrong if k_rot != 0
            k_translational=scale_variant,  # Translational spring strength scaling
            k_rotational=0,  # Rotational spring disabled
            symmetries=[[1, 0, 0, 0]],  # No symmetries - does not matter with k_rot=0
        )

        assert simulation.timestep == simulation.timestep

        spring_updater_action = SpringUpdaterAction(
            springs=harmonic_field, variant=scale_variant, final_scale_factor=0
        )
        spring_updater = hoomd.update.CustomUpdater(
            action=spring_updater_action,
            trigger=1,
        )
        simulation.operations.add(spring_updater)
        simulation.operations.integrator.external_potential = harmonic_field # Add the harmonic field to the integrator
    #**********************************************************************

    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=["type_shapes"])

    if floppy_nvt == True:
        # BoxMC updater
        boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.And([hoomd.trigger.Periodic(100), hoomd.trigger.After(simulation.timestep + 5000)]), P=0)
        boxmc.shear = {"weight":0.7, "delta":tuple([0.3, 0.3, 0.3]), "reduce":0.0}
        boxmc.aspect = {"weight":0.7, "delta":0.3}
        simulation.operations.updaters.append(boxmc)

        boxmc_tune = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
                trigger=hoomd.trigger.And(
                    [
                        hoomd.trigger.Periodic(100),
                        # Longer tune at start of run, plus a small tune on restart
                        hoomd.trigger.Before(simulation.timestep + 1000),
                    ]
                ),
                boxmc=boxmc,
                moves=["shear_x", "shear_y", "shear_z", "aspect"],
                max_move_size={
                    "shear_x": 0.3,
                    "shear_y": 0.3,
                    "shear_z": 0.3,
                    "aspect": 0.3,
                },
                target=0.3,
            )
        logger.add(boxmc, ["shear_moves"])
        simulation.operations.tuners.append(boxmc_tune)

    # NVT trajectory
    if os.path.exists(directories + tag + "_nvt_traj_pf" + final_pf_string + "_" + str(0) + ".gsd"):
        output_GSD_traj = directories + tag + "_nvt_traj_pf" + final_pf_string + "_" + str(state_id) + ".gsd"

    else:
        output_GSD_traj = directories + tag + "_nvt_traj_pf" + final_pf_string + "_" + str(0) + ".gsd"

    gsd_writer = hoomd.write.GSD(
            filename=output_GSD_traj,
            trigger=hoomd.trigger.Periodic(write_period),
            mode="ab",
            filter=hoomd.filter.All(),
            logger=logger)
    simulation.operations.writers.append(gsd_writer)
        

    print(str(simulation.timestep) + " / " + str(end_step) + " completed")

    file = open("log.txt", mode="a", newline="\n")
    # Print on terminal
    '''table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=write_period), logger=logger)
    simulation.operations.writers.append(table)''' 

    '''# Redefine the logger
    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=["type_shapes"])'''

    # Print log file
    '''table_file = hoomd.write.Table(trigger=hoomd.trigger.Periodic(period=write_period), output=file, logger=logger)
    simulation.operations.writers.append(table_file)'''

    # Restart file
    '''restart_writer = hoomd.write.GSD(filename=directories + tag + "_nvt_restart.gsd", trigger=hoomd.trigger.Periodic(100), mode="wb", truncate=True, filter=hoomd.filter.All(), logger=logger)
    simulation.operations.writers.append(restart_writer)'''

    # Run simulation
    simulation.run(end_step)
    gsd_writer.flush() # Flush the GSD writer to ensure all data is written to file


    # Output final_nvt GSD
    if os.path.exists(directories + tag + "_nvt_final_pf" + final_pf_string + "_" + str(0) + ".gsd"):
        output_GSD_final = directories + tag + "_nvt_final_pf" + final_pf_string + "_" + str(state_id) + ".gsd"

    else:
        output_GSD_final = directories + tag + "_nvt_final_pf" + final_pf_string + "_" + str(0) + ".gsd"


    hoomd.write.GSD.write(state=simulation.state, mode="xb", filename=output_GSD_final, logger=logger)

    print("Run complete!")
    print("Translational acceptance : ", mc.translate_moves[0] / sum(mc.translate_moves))
    print("Rotational acceptance : ", mc.rotate_moves[0] / sum(mc.rotate_moves))
    print("Overlaps : ", mc.overlaps)