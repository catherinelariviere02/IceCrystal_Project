import signac 
import gsd.hoomd 
import itertools
import math
import numpy

project = signac.init_project()

directory = "/Users/clarivi/Desktop/Research/IceCrystal/temp_files/"
list = [{"gsd": directory + "/NaCl/NaCl_nvt_final_pf0p6_0.gsd", "atoms":["Na", "Cl"]}, 
        {"gsd": directory + "/cubicdiamond/cubicdiamond_nvt_final_pf0p6_0.gsd", "atoms":["C"]}, 
        {"gsd": directory + "/cubicdiamond/cubicdiamond_compressed_to_pf0p6_final.gsd", "atoms":["C"]}] 
        
# could be made more efficient by changing how list comprehension works, possibly change once running for more ice
for d in list:
    statepoint = dict(seed=10, 
                    N=1000,
                    final_vol = 0.6, 
                    replicas = 10, 
                    equilib_runs = 1e7,
                    compression=False, 
                    unitcell=d["gsd"], #insert gsd file here
                    atom_types=d["atoms"])
    job = project.open_job(statepoint)
    job.init()
