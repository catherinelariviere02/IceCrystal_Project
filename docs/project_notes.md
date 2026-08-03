# Project Notes: Ice Crystals 

## Notes on Code Flow 

### guidelines for inputs 

Input file should be organized first by crystal type. For each crystal type, there will be a json file for each type, the cif file used to generate the shapes, the gsd file of the generated unit cell (after nvt equilibration), and a README.md file with information on the r_cuts used to generate the shapes from crystal2shape. 

#### Guideline for naming: 
Folder: name of crystal (and replica number)

json: "shape_[crystal_name]_[type]_unit_volume_principal_frame.json" for each shape 

cif file: "[crystal_name].cif" 

gsd file of uc: "[crystal_name]_nvt_final_pf0p6_0.gsd"

#### Information in job project: 
input file path 

name of crystal 

### Thought process:
if each input cif is in its own folder, crystal2shape will drop all additional input files into that folder - I can always go back in and change the naming convention slightly if necessary, but it will streamline the process and minimize information stored in job statepoint. 
