import signac 
import gsd.hoomd 
import itertools
import math
import numpy
import os

project = signac.init_project("/home/clarivi/projects/IceCrystal_Project/data/")
list = [1]
phase_names = ["36_H2O_0", "141_H2O_0", "92_H2O_3", "92_H2O_5"] # ice II 
comp_list = [False]
# could be made more efficient by changing how list comprehension works, possibly change once running for more ice
for phase_name in phase_names:
    for comp_state in comp_list:
        if os.path.isdir(f"../../inputs/{phase_name}") != True: 
            os.mkdir(f"../../inputs/{phase_name}") #this should maybe just raise an error 
        statepoint = dict(inputfile = f"/home/clarivi/projects/IceCrystal_Project/inputs/{phase_name}/", 
                        crystal_name = phase_name,
                        runtime = 1_000,
                        logsteps = 100,
                        replicas = 5, 
                        compression=comp_state, 
                        gsd=f"{phase_name}_nvt_final_pf0p6_0.gsd", #insert gsd file here
                        atoms=["O", "H"], 
                        rdf_rmax=5,
                        bod_rmax=1.5)
        job = project.open_job(statepoint)
        if job not in project:
            job.init()

