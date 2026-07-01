import signac 
import gsd.hoomd 
import itertools
import math
import numpy

project = signac.init_project("../../data")

list = [1, 2, 3, 4, 5, 6]  
phase_num = "92_H2O" 
phase_name = "IceXI"     
# could be made more efficient by changing how list comprehension works, possibly change once running for more ice
for d in list:
    statepoint = dict(inputfile = f"../../inputs/{phase_num}/{phase_num}_{str(d)}/", 
                      phase_num = "92_H20" + str(d),
                      runtime = 100_000,
                      logsteps = 100,
                      replicas = 10, 
                    compression=False, 
                    gsd=f"{phase_num}_{phase_name}_nvt_final_pf0p6_0.gsd", #insert gsd file here
                    atoms=["O, H"])
    job = project.open_job(statepoint)
    job.init()

project = signac.get_project("../../data")
for job in project: 
        if job.sp.atoms == ["O, H"]: 
              job.sp.atoms = ["O", "H"]