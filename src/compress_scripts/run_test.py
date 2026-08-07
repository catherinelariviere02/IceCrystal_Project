import hoomd 
from utils import create_simulation, get_shape_info 
import signac 

#open signac jobs
project = signac.get_project("../../data/workspace")

for job in project: 
    if job.sp.compression == True:
        _, _, _, shapes, _, shape_volume = get_shape_info(job.sp.inputfile, 
                                                            job.sp.replicas, 
                                                            job.sp.atoms, 
                                                            job.sp.crystal_name, 
                                                            job.sp.stoich)
        
        simulation = create_simulation(job.fn("timeout_config.gsd"),
                                                frame = 0,
                                                shapes = shapes, 
                                                atoms = job.sp.atoms)

        packing_fraction = shape_volume /  simulation.state.box.volume
        print(f"job {job.id} finished compression with packing_fraction {packing_fraction}")