# IMPORTS 
import math 
import hoomd 
import gsd.hoomd 

# necessary job information: input file, atom types, gsd name, number of replicas, output file 
def create_simulation(job):
     
    # IMPORT UNIT CELL AND CRYSTAL SHAPES
    directory = job.statepoint.inputfile # "/Users/clarivi/Desktop/Research/IceCrystal/inputs/cif_files/temp_files/"
    gsdfile = directory + job.statepoint.gsd # "92_H2O_IceXI_nvt_traj_pf0p6_0.gsd"
    atoms = job.statepoint.atoms # ["O", "H"] #atom names in order they are logged in "shapes"
    print(f"atoms, {atoms}")

    traj = gsd.hoomd.open(gsdfile, mode="r") 
    shapes = traj[0].particles.type_shapes 

    # INITIALIZE SIMULATION
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device = cpu, seed = 1)
    simulation.timestep = 0

    # create state from unit cell 
    simulation.create_state_from_gsd(filename = gsdfile, frame = 0)

    # create larger cell from unit cell (replication)
    replicas = job.statepoint.replicas #4
    simulation.state.replicate(nx = replicas, ny = replicas, nz = replicas)

    # SET UP MC SIMULATION 
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()

    #interate over all shapes (will be necessary for binary system)
    for i in range(len(shapes)):
        print(atoms[i])
        mc.shape[atoms[i]] = dict(
            vertices = shapes[i]["vertices"]
        )

    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=["type_shapes"])

    simulation.operations.integrator = mc

    simulation.run(1)

    hoomd.write.GSD.write(state = simulation.state, 
                        mode = "wb", 
                        filename = job.fn(f"initialize_{atoms}.gsd"), 
                        logger = logger)

    return simulation 
