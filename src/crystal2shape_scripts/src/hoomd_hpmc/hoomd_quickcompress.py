import hoomd
import gsd.hoomd
from matplotlib import patches
import numpy as np
import os, rowan, freud, coxeter

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

def hoomd_quickcompress_func(directories, N_particles, shape_data, communicator, tag, quickCompress=True, 
                             target_pf=0.6, sys_prep_pf=0.1, patches="False",
                             translational_spring=True):
    """
    Function to perform quick compression on a shape using HOOMD-blue.

    Parameters
    ----------
    directories : str
        Directory where the simulation files are stored.

    N_particles : int
        Number of particles in the simulation.

    shape_data : dict
        Dictionary containing shape data, including vertices and volume.

    communicator : hoomd.communicator.Communicator
        The communicator for parallel execution.

    tag : str
        Tag for the output GSD file.

    boxResize : bool, optional
        If True, uses box resizing instead of quick compression. Default is False.

    translational_spring : bool, optional
        If True, applies translational springs to the particles during compression. Default is True.


    Returns
    -------
    N_particles : int
        Number of particles after compression.

    shape_data : dict
        Updated shape data after compression.

    Notes
    -----
    This function initializes a HOOMD-blue simulation, applies a quick compression updater,
    and runs the simulation until the compression is complete. It also handles the writing
    of GSD files for the simulation state and trajectory.
    """

    # Read parameters
    write_period = 100_000
    final_pf = target_pf
    end_step = 2_000

    # Shape data info
    shape_data_keys, shape_data_values = list(shape_data.keys()), list(shape_data.values())
    shape_vertices, basis_types, volume, num_particles, gsd_types = [], [], [], [], []
    for i in range(len(shape_data_keys)):
        shape_vertices.append(shape_data_values[i]["Vertices"])
        basis_types.append(shape_data_keys[i])
        volume.append(shape_data_values[i]["Volume"])
        num_particles.append(shape_data_values[i]["Num_particles"])
        gsd_types.append(shape_data_values[i]["types"])


    # move_sizes = [[[0.05, 0.02], [0.05, 0.02]] for _ in range(len(basis_types))] # If the crystal melts, decrease the translational move size. This causes increasing the runtime though.
    # shape_vertices = [[sv, sv] for sv in shape_vertices]

    move_sizes = [[[0.01, 0.0001]] for _ in range(len(basis_types))] # Move sizes for mobile types only
    shape_vertices = [[sv] for sv in shape_vertices] # vertices for mobile types only
    # Flatten the move_sizes list and gsd_types list and shape_vertices to match the mobile and fixed types for each basis type
    gsd_types = [t for sublist in gsd_types for t in sublist]
    move_sizes = [move for sublist in move_sizes for move in sublist]
    shape_vertices = [sv for sublist in shape_vertices for sv in sublist]
    V_particle = [n * volume[i] for i, n in enumerate(num_particles)] # Volume per particle

    # Create simulation
    simulation = create_simulation(communicator, shape_vertices, gsd_types, move_sizes)
    mc = simulation.operations.integrator

    final_pf_string = str(final_pf).replace(".", "p")
    sys_prep_pf_str = str(sys_prep_pf).replace(".", "p")
    # Read input GSD based on the requirements
    if os.path.exists(directories + tag + "_compressed_to_pf" + final_pf_string + "_final.gsd"):
        input_GSD_filename = directories + tag + "_compressed_to_pf" + final_pf_string + "_final.gsd"
        simulation.create_state_from_gsd(filename=input_GSD_filename)

    elif os.path.exists(directories + tag + "_compression_restart.gsd"):
        input_GSD_filename = directories + tag + "_compression_restart.gsd"
        simulation.create_state_from_gsd(filename=input_GSD_filename)

    else:
        input_GSD_filename = directories + tag + "_uc_pf" + sys_prep_pf_str + ".gsd"
        simulation.create_state_from_gsd(filename=input_GSD_filename)


    # simulation.run(100)  # Randomize the state

    initial_volume_fraction = (np.sum(np.array(V_particle)) / simulation.state.box.volume)
    print("Initial volume fraction", round(initial_volume_fraction, 5))

    # Move size Tuner
    periodic = hoomd.trigger.Periodic(100)
    tune = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["a"],
        target=0.2,
        trigger=periodic,
        types=gsd_types,  # Only mobile types - [gsd_types[i*2] for i in range(len(basis_types))]
        max_rotation_move=0.5,
        )
    simulation.operations.tuners.append(tune)

    # Patchy potential
    if patches == "True":
        patch_info = [45, 2.0, 3.0, 3.0]
        patch_potential = apply_gaussianPotential_func(shape_data, patch_info)
        simulation.operations.integrator.pair_potentials = [patch_potential]

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
            trigger=100,
        )
        simulation.operations.add(spring_updater)
        simulation.operations.integrator.external_potential = harmonic_field # Add the harmonic field to the integrator
        

    # Logger
    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=["type_shapes"])

    # Output GSD trajectory file
    output_GSD_traj = directories + tag + "_compression_traj_pf" + final_pf_string + ".gsd"

    gsd_writer = hoomd.write.GSD(
            filename=output_GSD_traj,
            trigger=hoomd.trigger.Periodic(write_period),
            mode="ab",
            filter=hoomd.filter.All(),
            logger=logger)
    simulation.operations.writers.append(gsd_writer)

    # Restart writer
    '''restart_writer = hoomd.write.GSD(filename=directories + tag + "_compression_restart.gsd", trigger=hoomd.trigger.Periodic(job.restart_frequency), mode="wb", truncate=True, filter=hoomd.filter.All(), logger=logger)
    simulation.operations.writers.append(restart_writer)'''

    # Quick Compress
    if quickCompress == True:
        if round(initial_volume_fraction, 5) < final_pf:
            # target_box_volume = (np.sum(np.array(V_particle)))/final_pf
            # target_box = hoomd.variant.box.InverseVolumeRamp(simulation.state.box, target_box_volume, simulation.timestep, 200)
            initial_box = simulation.state.box
            initial_box = hoomd.Box.from_box(initial_box)
            initial_box_volume = initial_box.volume
            initial_box_matrix = initial_box.to_matrix()
            final_volume_fraction = final_pf
            final_box_volume = np.sum(np.array(V_particle)) / final_volume_fraction
            final_box_matrix = initial_box_matrix*(final_box_volume/initial_box_volume)**(1/3)
            target_box = hoomd.Box.from_matrix(final_box_matrix)

            compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(100), target_box=target_box, min_scale=0.9)
            simulation.operations.updaters.append(compress)

            while not compress.complete:
                pf = round((np.sum(np.array(V_particle)))/simulation.state.box.volume, 5)
                simulation.run(end_step)

                print(simulation.timestep, round((np.sum(np.array(V_particle)))/simulation.state.box.volume, 5))
    else:
        while round(initial_volume_fraction, 5) < final_pf:
            initial_box = simulation.state.box
            initial_box = hoomd.Box.from_box(initial_box)
            initial_box_volume = initial_box.volume

            volume_factor = 0.999
            updated_box_volume = initial_box_volume * volume_factor
            initial_box_matrix = initial_box.to_matrix()
            updated_box_matrix = initial_box_matrix*(updated_box_volume/initial_box_volume)**(1/3)
            updated_box = hoomd.Box.from_matrix(updated_box_matrix)
            
            inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
                initial_box=simulation.state.box,
                final_volume=updated_box.volume,
                t_start=simulation.timestep,
                t_ramp=200,
            )

            box_resize = hoomd.update.BoxResize(
                trigger=hoomd.trigger.Periodic(100),
                filter=hoomd.filter.All(),
                box=inverse_volume_ramp,
            )
            simulation.operations.updaters.append(box_resize)
            simulation.run(end_step)

            while mc.overlaps != 0:  # Remove overlaps
                simulation.run(100)
                if simulation.timestep > 500_000:
                    print("Too many overlaps, running expansion.")
                    
                    volume_factor = 1.01
                    updated_box_volume = initial_box_volume * volume_factor
                    initial_box_matrix = initial_box.to_matrix()
                    updated_box_matrix = initial_box_matrix*(updated_box_volume/initial_box_volume)**(1/3)
                    updated_box = hoomd.Box.from_matrix(updated_box_matrix)
                    
                    inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
                        initial_box=simulation.state.box,
                        final_volume=updated_box.volume,
                        t_start=simulation.timestep,
                        t_ramp=200,
                    )

                    box_resize = hoomd.update.BoxResize(
                        trigger=hoomd.trigger.Periodic(100),
                        filter=hoomd.filter.All(),
                        box=inverse_volume_ramp,
                    )
                    simulation.operations.updaters.append(box_resize)
                    simulation.run(end_step)
                    break
        
            initial_volume_fraction = (np.sum(np.array(V_particle)) / simulation.state.box.volume)
            if initial_volume_fraction > 0.5 and initial_volume_fraction <final_pf:
                simulation.run(2_000)
            elif initial_volume_fraction <= 0.5:
                simulation.run(end_step)
            else:
                break

            print(simulation.timestep, round(initial_volume_fraction, 3), mc.overlaps)

    # Run NVT at final stage
    print("Running to relax structure after compression...")
    simulation.run(5_000)

    final_volume_fraction = (np.sum(np.array(V_particle)) / simulation.state.box.volume)
    final_volume_rounded_string = str(round(final_volume_fraction, 5)).replace(".", "p")
    print("Final volume fraction", round(final_volume_fraction, 5))

    # Output compressed GSD
    output_GSD_final = directories + tag + "_compressed_to_pf" + final_pf_string + "_final.gsd"
    if os.path.exists(output_GSD_final):
        pass

    else:
        logger = hoomd.logging.Logger()
        logger.add(simulation.operations.integrator, quantities=["type_shapes"])
        hoomd.write.GSD.write(state=simulation.state, mode="xb", filename=output_GSD_final, logger=logger)

    print("Overlaps particle-particle: ", mc.overlaps)

    return N_particles, shape_data, mc.overlaps