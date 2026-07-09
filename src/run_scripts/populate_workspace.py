import signac 
import gsd.hoomd 
import itertools
import math
import numpy

project = signac.init_project("../../data")

list = [1]
phase_num = "141_H2O" 
phase_name = "1"     
# could be made more efficient by changing how list comprehension works, possibly change once running for more ice
for d in list:
    statepoint = dict(inputfile = f"../../inputs/{phase_num}_{phase_name}/", 
                      phase_num = "92_H20" + str(d),
                      runtime = 100_000,
                      logsteps = 100,
                      replicas = 4, 
                    compression=False, 
                    gsd=f"{phase_num}_{phase_name}_nvt_final_pf0p6_0.gsd", #insert gsd file here
                    atoms=["O", "H"], 
                    rdf_rmax=3.0,
                    bod_rmax=3.0)
    job = project.open_job(statepoint)
    if job not in project:
        job.init()

# for job in project:
#         if "rdf_rmax" not in job.sp:
#                 assert "rdf_rmax" not in job.sp, f"includes 'rdf_rmax' in {job.id}"
#                 job.sp.rdf_rmax = job.statepoint.pop("rdf_max")


