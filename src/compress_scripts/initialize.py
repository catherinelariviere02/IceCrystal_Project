# IMPORTS 
import math 
import hoomd 
import gsd.hoomd 

def create_simulation():
     
    # IMPORT UNIT CELL AND CRYSTAL SHAPES
    directory = "/Users/clarivi/Desktop/Research/IceCrystal/inputs/cif_files/temp_files/"
    gsdfile = directory + "92_H2O_IceXI_nvt_traj_pf0p6_0.gsd"
    atoms = ["O", "H"] #atom names in order they are logged in "shapes"

    traj = gsd.hoomd.open(gsdfile, mode="r") 
    shapes = traj[0].particles.type_shapes 

    # INITIALIZE SIMULATION
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device = cpu, seed = 1)
    simulation.timestep = 0

    # create state from unit cell 
    simulation.create_state_from_gsd(filename = gsdfile, frame = 0)

    # create larger cell from unit cell (replication)
    replicas = 4
    simulation.state.replicate(nx = replicas, ny = replicas, nz = replicas)

    # SET UP MC SIMULATION 
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()

    #interate over all shapes (will be necessary for binary system)
    for i in range(len(shapes)):
        mc.shape[atoms[i]] = dict(
            vertices = shapes[i]["vertices"]
        )

    logger = hoomd.logging.Logger()
    logger.add(mc, quantities=["type_shapes"])

    simulation.operations.integrator = mc

    simulation.run(1)

    hoomd.write.GSD.write(state = simulation.state, 
                        mode = "xb", 
                        filename = f"../../data/iceIX/initialize_{atoms}.gsd", 
                        logger = logger)

    return simulation 
