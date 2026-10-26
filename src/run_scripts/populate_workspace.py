import signac 
import gsd.hoomd 
import itertools
import math
import numpy
import os
import ase.io 
from ase.formula import Formula

project = signac.init_project("/home/clarivi/projects/IceCrystal_Project/data/data/")

phase_names = ["141_H2O_2"] # ice II
comp_list = [True]
# could be made more efficient by changing how list comprehension works, possibly change once running for more ice
pf = 0.63
pf_name = str(pf).replace(".", "p")

for phase_name in phase_names:
    print(f"{phase_name}_nvt_final_pf{pf_name}_0.gsd")
    for comp_state in comp_list:
        if os.path.isdir(f"../../inputs/{phase_name}") != True: 
            os.mkdir(f"../../inputs/{phase_name}") #this should maybe just raise an error 
        statepoint = dict(inputfile = f"/home/clarivi/projects/IceCrystal_Project/inputs/{phase_name}/", 
                        crystal_name = phase_name,
                        runtime = 100_000_000,
                        logsteps = 100,
                        replicas = 5, 
                        compression=comp_state, 
                        gsd= f"{phase_name}_nvt_final_pf{pf_name}_0.gsd", #insert gsd file here
                        atoms=["O"], 
                        rdf_rmax=5,
                        bod_rmax=1.5, 
                        stoich={"O": 1}, 
                        pf = pf, 
                        seed = 1, 
                        image_write = True)
        job = project.open_job(statepoint)
        if job not in project:
            job.init()

# for job in project: 
#     if "image_write" not in job.sp:
#         job.sp.image_write = False

# import csv 
# import json 

# fieldnames = []

# with open("statepoints.csv", "w", newline ='') as csvfile: 
#     for job in project: 
#         if fieldnames == []: 
        
#             fieldnames.append("job_name")
#             fieldnames.extend(list(job.sp))
#             print(fieldnames)
            
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
        
#         input = job.sp().copy()
#         input.update({"job_name": job.id})
    
#         writer.writerow(input)
       