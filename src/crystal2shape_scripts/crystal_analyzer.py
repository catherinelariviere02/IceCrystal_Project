"""
Crystal Analyzer Module
=======================

This module defines the :class:`CrystalAnalyzer` class, which serves as the main controller
for the crystal structure analysis, shape generation, and HPMC (Hard Particle Monte Carlo) simulation pipelines.
It coordinates the parsing of CIF files or space group creation, symmetry/Wyckoff position analysis,
construction of coordination polyhedra/shapes, and execution of HOOMD-blue simulations.

Example:
    To run the complete pipeline using a configuration file:

    .. code-block:: python

        from crystal_analyzer import CrystalAnalyzer

        # Initialize analyzer with a configuration file
        analyzer = CrystalAnalyzer("param_file.json")

        # Prepare the system from space group or CIF
        analyzer.prepare_system()

        # Analyze system symmetry and local environments
        analyzer.analyze_symmetry()

        # Generate coordination shapes
        shape_verts, types, ref_particles, envelopes = analyzer.generate_shapes(auto_select=True)

        # Write shape definitions to JSON
        analyzer._JSON_writer(shape_verts, ref_particles, types)

        # Run HOOMD-HPMC compression and NVT simulations
        analyzer.run_simulations(
            shape_verts=shape_verts,
            types=types,
            sys_prep_pf=analyzer.config["sys_prep_pf"],
            target_pf=analyzer.config["target_pf"]
        )
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import freud
import gsd.hoomd
import hoomd
import rowan
from scipy.spatial import ConvexHull

# Project-specific imports
from src.reader_writer import prepare_box, json_writer, gsd_writer
from src.symmetry import wyckoff
from src.utils import calc_ref_particle_rcut, create_uc_from_sg
from src.utils.get_shape import ShapeAnalyzer
from src.visualization import pyvista_plot
from src.hoomd_hpmc import hoomd_quickcompress, hpmc_nvt
from src.utils.utils_func import safe_input

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CrystalAnalyzer:
    """Main controller for crystal structure analysis, shape generation, and HPMC simulation pipelines.

    This class coordinates the workflow of importing/generating a crystal structure, calculating
    Wyckoff positions, identifying coordination polyhedra (shapes), and setting up/running
    HOOMD-HPMC compression and NVT simulations.

    Attributes:
        config_path (str): Path to the JSON configuration parameters file.
        config (dict): The loaded configuration dictionary.
        directory (str): Base working directory path.
        data (dict): Reference to the configuration dictionary.
        spg_num (int): Space group number of the crystal.
        cif_file (str): Full path to the input CIF file.
        shape_id (str): Identifier/name derived from the input CIF filename.
        rd_factor (int): Precision scaling factor for rounding tolerances.
        rtol (float): Relative tolerance derived from `rd_factor`.
        pg_tolerance (float): Point group symmetry matching tolerance.
        target_pf (float): Target packing fraction for simulations.
        temp_dir (str): Directory path for storing temporary/output files.
        unitcell (freud.data.UnitCell): The unit cell object representing the crystal lattice.
        system (gsd.hoomd.Snapshot): The hoomd/gsd snapshot of the crystal system.
        uc_particle_indices (List[int]): Indices of particles that belong to the reference unit cell.
        equiv_atoms (List[int]): List indicating equivalent atoms in the unit cell.
        rdf_rcut (float): Radial cut-off distance for coordinate environment analysis.
    """

    def __init__(self, config_path: str = "./param_file.json"):
        """Initializes the CrystalAnalyzer with configuration settings.

        Args:
            config_path (str): Path to the configuration JSON file. Defaults to "./param_file.json".
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._set_attributes()
        
        # Internal State
        self.unitcell = None
        self.system = None
        self.uc_particle_indices: List[int] = []
        self.equiv_atoms: List[int] = []
        self.rdf_rcut: float = 0.0

    def _load_config(self) -> Dict[str, Any]:
        """Loads and parses the input JSON configuration.

        Returns:
            Dict[str, Any]: Parsed configuration parameters.

        Raises:
            FileNotFoundError: If the configuration file cannot be found at `config_path`.
            ValueError: If the configuration file contains invalid JSON.
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Configuration file {self.config_path} not found.")
            raise FileNotFoundError(f"Config file missing: {self.config_path}")
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.critical(f"Failed to parse configuration file '{self.config_path}'. Invalid JSON structure.")
            logger.critical(f"JSON Error: {e.msg} at line {e.lineno}, column {e.colno}")
            raise ValueError(f"Invalid JSON format in configuration file: {e}")

    def _set_attributes(self):
        """Maps configuration settings to instance attributes and initializes directories."""
        c = self.config
        
        # Define required keys with their expected types
        required_keys = {
            "space_group_number": int,
            "input_CIF": str,
            "rd_factor": int,
            "pg_tolerance": float,
            "target_pf": float,
            "num_replicas": int,
            "type_arr": list,
            "radii_arr": list,
            "target_volume": list,
            "sys_prep_pf": float
        }
        
        missing_or_invalid = []
        for key, expected_type in required_keys.items():
            if key not in c:
                missing_or_invalid.append(f"Missing required configuration key: '{key}'")
            else:
                val = c[key]
                if expected_type is float and isinstance(val, (int, float)):
                    continue
                elif not isinstance(val, expected_type):
                    missing_or_invalid.append(
                        f"Invalid type for '{key}': expected {expected_type.__name__}, got {type(val).__name__}"
                    )
        
        if missing_or_invalid:
            for error_msg in missing_or_invalid:
                logger.error(error_msg)
            raise ValueError(f"Configuration validation failed: {', '.join(missing_or_invalid)}")

        # Value bounds validation
        if c["space_group_number"] <= 0 or c["space_group_number"] > 230:
            raise ValueError(f"Invalid 'space_group_number' ({c['space_group_number']}). Must be in range [1, 230].")
        if not (0 < c["target_pf"] <= 1.0):
            raise ValueError(f"Invalid 'target_pf' ({c['target_pf']}). Must be in range (0, 1.0].")
        if not (0 < c["sys_prep_pf"] <= 1.0):
            raise ValueError(f"Invalid 'sys_prep_pf' ({c['sys_prep_pf']}). Must be in range (0, 1.0].")
        if c["rd_factor"] < 0:
            raise ValueError(f"Invalid 'rd_factor' ({c['rd_factor']}). Must be a non-negative integer.")
        if len(c["type_arr"]) != len(c["radii_arr"]):
            raise ValueError("Configuration mismatch: 'type_arr' and 'radii_arr' must have matching lengths.")
        if len(c["type_arr"]) != len(c["target_volume"]):
            raise ValueError("Configuration mismatch: 'type_arr' and 'target_volume' must have matching lengths.")

        self.directory = c.get("directory", "./")
        self.data = c # store full config for reference
        self.spg_num = c["space_group_number"]
        self.cif_file = os.path.join(self.directory, c["input_CIF"])
        self.shape_id = os.path.splitext(c["input_CIF"])[0]
        
        # Tolerances
        self.rd_factor = c["rd_factor"]
        self.rtol = 1 / pow(10, self.rd_factor)
        self.pg_tolerance = c["pg_tolerance"]
        self.target_pf = c["target_pf"]
        
        # Ensure temp directory exists
        self.temp_dir = os.path.join(self.directory, "temp_files/")
        os.makedirs(self.temp_dir, exist_ok=True)

    def prepare_system(self):
        """Constructs the crystal system using CIF parser or Space Group builder.

        This method initializes the unit cell (`unitcell`), system coordinates snapshot (`system`), 
        and mapping indices depending on whether the configuration specifies generating the crystal 
        from a CIF file (`use_CIF`) or from space group parameters (`create_struc`).
        """
        if str(self.config.get("create_struc")) == "True":
            logger.info(f"Creating structure from Space Group: {self.spg_num}")
            builder = create_uc_from_sg.uc_class(space_group_number=self.spg_num)
            builder.create_uc(
                self.config["num_replicas"], self.config["sys_prep_pf"],
                write_gsd=True, directory=self.directory, shape_id=self.shape_id
            )
            self.unitcell, self.system = builder.unitcell, builder.system
            self.uc_particle_indices = builder.uc_particle_indices

        elif str(self.config.get("use_CIF")) == "True":
            logger.info(f"Parsing CIF File: {self.cif_file}")
            prep_obj = prepare_box.prepareBox(self.cif_file)
            prep_obj.prepareBox_func(
                num_replicas=self.config["num_replicas"],
                target_pf=self.config["sys_prep_pf"],
                write_gsd=True, directory=self.directory,
                shape_id=self.shape_id, space_group_number=self.spg_num,
                separate_equiv_atoms=True
            )
            self.unitcell, self.system = prep_obj.unitcell, prep_obj.system
            self.uc_particle_indices = prep_obj.uc_particle_indices

        self.equiv_atoms = self.unitcell.equivalent_atoms
        self.system.equiv_atoms = np.repeat(
            self.equiv_atoms, self.config["num_replicas"]**3
        ).tolist()

    def analyze_symmetry(self):
        """Analyzes crystal symmetry, Wyckoff positions, and coordination clusters.

        Computes Wyckoff coordinates for the system particles and performs local environment
        clustering. Updates the system snapshot with Wyckoff classifications and clusters.
        """
        logger.info("Analyzing Wyckoff positions and local environment...")
        w_obj = wyckoff.WyckoffAnalyzer()
        w_obj.compute_wyckoff_positions(
            unitcell=self.unitcell, system=self.system,
            uc_particle_indices=self.uc_particle_indices,
            rmax=self.config["rmax"], bins=self.config["bins"], rcut=self.config["rmax"],
            spg_mode=True, num_replicas=self.config["num_replicas"],
            show_rdf=self.config["show_rdf"]
        )
        
        self.system.wyckoffs = [str(t) for t in w_obj.system_wyckoff]
        self.rdf_rcut = w_obj.rdf_rcut
        self.system.local_env_cluster_ids = w_obj.local_env_cluster_ids
        self.system.cluster_centers = w_obj.cluster_centers

    def _get_truncation_dict(self, basis_types: List[str]) -> Dict[str, float]:
        """Calculates pairwise relative truncation distances based on species radii.

        Args:
            basis_types (List[str]): List of active basis atom types in the crystal.

        Returns:
            Dict[str, float]: Pairwise truncation ratios mapping 'typeA_typeB' to a float value.
        """
        radii = self.config["radii_arr"]
        tr_dic = {}
        for i, type_a in enumerate(basis_types):
            for j, type_b in enumerate(basis_types):
                tr_dic[f"{type_a}_{type_b}"] = round(radii[i] / (radii[i] + radii[j]), 2)
        return tr_dic

    def generate_shapes(self, auto_select: bool = False) -> Tuple[List, List, List, List]:
        """Generates representative polyhedral shapes for each basis atom type.

        Computes coordination polyhedra around selected reference particles based on local
        voronoi coordination and truncation settings. The generated vertices are centered
        about their center of mass and scaled to target volumes.

        Args:
            auto_select (bool): If True, defaults to the first unit cell particle of each type as
                the reference particle. If False, prompts the user interactively in the terminal
                for the reference particle ID. Defaults to False.

        Returns:
            Tuple[List, List, List, List]: A tuple containing:
                - List[List[float]]: Centered and normalized vertex coordinates for each type.
                - List[str]: The active basis type names.
                - List[int]: Reference particle IDs used for the shape generations.
                - List[List[float]]: Enclosing envelope vertices.
        """
        ref_calc = calc_ref_particle_rcut.RefParticleCalculator()
        basis_types = [t for t in self.config["type_arr"] if t in np.unique(self.system.typeids)]
        
        ref_calc.compute(
            self.unitcell, self.system, self.uc_particle_indices,
            basis_type=basis_types, rcut=self.rdf_rcut, rd_factor=self.rd_factor,
            rtol=self.rtol, pg_tolerance=self.pg_tolerance,
            space_group_number=self.spg_num,
            sys_prep_pf=self.config["sys_prep_pf"],
            atom_type_selection=self.config["atom_type_selection"],
            length_contraction=self.config["length_contraction"],
            d_tol=self.config["d_tol"]
        )

        tr_pt_dic = self._get_truncation_dict(basis_types)
        final_verts, final_envelopes, ref_particles = [], [], []

        for i, atom_type in enumerate(basis_types):
            logger.info(f"Processing shape for atom type: {atom_type}")
            trial_counter = "y" # initialize trial counter to allow user to try different reference particles
            while trial_counter.lower() == "y":
                # Reference Selection
                if not auto_select:
                    valid_particle_ids = [j for j in self.uc_particle_indices if self.system.typeids[j] == atom_type]
                    if not valid_particle_ids:
                        raise ValueError(f"No particles of type {atom_type} found in unit cell indices.")
                    ref_id = safe_input(
                        prompt=f"Enter Reference Particle ID for {atom_type}: ",
                        expected_type=int,
                        valid_choices=valid_particle_ids
                    )
                else:
                    # Default to first particle of type
                    ref_id = [j for j in self.uc_particle_indices if self.system.typeids[j] == atom_type][0]

                poly_gen = ShapeAnalyzer()
                poly_gen.compute(
                    self.unitcell, self.system, self.uc_particle_indices,
                    atom_type, ref_id, ref_calc.ref_particle_arr,
                    ref_calc.rcut_all_arr, tr_pt_dic, 
                    self.config["tr_pt"][i], basis_types,
                    rtol=self.rtol, space_group_number=self.spg_num,
                    rd_factor=self.rd_factor, directory=self.directory, shape_id=self.shape_id,
                    sys_prep_pf=self.config["sys_prep_pf"], show_poly=True,
                    pg_tolerance=self.pg_tolerance, site_symmetry_dict=ref_calc.site_symmetry_dict,
                    atom_type_selection=ref_calc.atom_type_selection,
                    unique_equiv_indices=ref_calc.unique_equiv_indices,
                    hierarchy_tol=self.config["hierarchy_tol"],
                    show_unitcell=self.config["show_unitcell"],
                    show_voronoi=self.config["show_voronoi"], 
                    show_hierarchy=self.config["show_hierarchy"],
                    show_truncation=self.config["show_truncation"], 
                    show_basis=self.config["show_basis"],
                    replicated_asym_unit_show=self.config["replicated_asym_unit_show"],
                    pg_map=ref_calc.pg_map, dist_map=ref_calc.dist_map,
                    length_contraction=self.config["length_contraction"]
                )

                # Prompt to save shapes from multiple reference particles  
                # save = input(f"Save this shape? y/n: ")
                # if save == "y":
                #     # Center and Normalize Vertices
                #     verts = np.array(poly_gen.shape_poly)
                #     verts -= np.mean(verts, axis=0)
                    
                #     if self.config["target_volume"][i] == 1:
                #         vol = ConvexHull(verts).volume
                #         verts *= (1.0 / vol)**(1/3)

                #     final_verts.append(verts.tolist())
                #     final_envelopes.append(poly_gen.envelop_vertices)
                #     ref_particles.append(ref_id)

                #     self._JSON_writer(final_verts, ref_particles, [atom_type])
                
                trial_counter = safe_input(
                    prompt=f"Try another reference particle for {atom_type}? (y/n): ",
                    expected_type=str,
                    valid_choices=["y", "n", "Y", "N"]
                ).lower()

            # Center and Normalize Vertices
            verts = np.array(poly_gen.shape_poly)
            verts -= np.mean(verts, axis=0)
            
            if self.config["target_volume"][i] == 1:
                vol = ConvexHull(verts).volume
                verts *= (1.0 / vol)**(1/3)

            final_verts.append(verts.tolist())
            final_envelopes.append(poly_gen.envelop_vertices)
            ref_particles.append(ref_id)

        return final_verts, basis_types, ref_particles, final_envelopes
    
    def _JSON_writer(self, shape_vertices, ref_particles, types):
        """Writes the scaled convex hull shape vertices to JSON description files.

        Computes convex hulls, scales shape sizes such that the minimum volume shape is unity, 
        and outputs configuration files detailing coordinates in the target temporary directory.

        Args:
            shape_vertices (List): Nested lists of shape vertices.
            ref_particles (List[int]): Reference particle IDs matched with the generated shapes.
            types (List[str]): List of active basis atom types.
        """
        # Scale the violume of the shapes such that the minimum one is unity

        shape_vertices_list, particle_vols = [], []
        for t in range(len(types)):
            vertices = shape_vertices[ref_particles.index([i for i in ref_particles if self.system.typeids[i] == types[t]][0])]
            hull = ConvexHull(vertices)
            hull_vertices = np.array([vertices[i] for i in hull.vertices])  # Keep only the convex hull vertices
            shape_vertices_list.append(hull_vertices.tolist())
            particle_vols.append(hull.volume)
            
        min_vol = min(particle_vols)
        modified_vols = [particle_vols[t]/min_vol for t in range(len(particle_vols))]

        # modified_vols = particle_vols[:] # No need to scale the shapes to have the same volume, as we are already scaling them to have the target volumes specified in the input JSON file during the shape generation step.

        shape_vertices_list_vol_scaled = []
        for v in range(len(shape_vertices_list)):
            shape_vertices = np.array(shape_vertices_list[v]) * (modified_vols[v]/particle_vols[v])**(1/3)
            hull = ConvexHull(shape_vertices)
            shape_vertices = np.array([shape_vertices[t] for t in hull.vertices])  # Keep only the convex hull vertices
            shape_vertices_list_vol_scaled.append(shape_vertices.tolist())
            print(types[v], round(modified_vols[v], 4))

        self.shape_vertices_list = shape_vertices_list_vol_scaled[:] # Overwrite the shape vertices list with volume scaled shapes
        self.directory = self.directory + "temp_files/"

        # Write JSON file of the shape
        counter = 0
        for vertices in shape_vertices_list:
            hull = ConvexHull(vertices)
            vertices = [vertices[t] for t in hull.vertices]  # Keep only the convex hull vertices
            JSON_writer_obj = json_writer.JSON_writer()
            JSON_writer_obj.JSON_writer_func(vertices, self.shape_id, self.directory, types, counter)
            counter += 1

    def run_simulations(self, shape_verts: List, types: List, sys_prep_pf: float, target_pf: float):
        """Executes the HOOMD-HPMC compression and NVT simulation pipeline.

        Prepares initial configuration snapshots (GSD format), executes quick compression
        up to the target packing fraction, and starts NVT simulations with HOOMD-blue if 
        compression runs succeed without overlaps.

        Args:
            shape_verts (List[List[float]]): Vertices defining each particle type's polyhedron shape.
            types (List[str]): List of particle/atom type designations.
            sys_prep_pf (float): Initial packing fraction for scaling the unit cell box volume.
            target_pf (float): Target packing fraction to compress to and run NVT under.
        """
        if str(self.config.get("create_unitcell")) != "True":
            return

        logger.info("Starting HOOMD-HPMC Simulation Pipeline...")
        
        # Prepare GSD data
        uc_pos = [self.system.positions[t] for t in self.uc_particle_indices]
        uc_orient = [rowan.random.rand(1) for _ in range(len(uc_pos))]
        typeids = [types.index(self.system.typeids[i]) for i in self.uc_particle_indices]

        shape_meta, shape_vol = [], []
        total_vol = 0.0
        for i, ty in enumerate(types):
            v = shape_verts[i]
            shape_meta.append({"type": "ConvexPolyhedron", "rounding_radius": 0.0, "vertices": v})
            shape_vol.append(ConvexHull(v).volume)
            total_vol += len([j for j in self.uc_particle_indices if self.system.typeids[j] == ty]) * ConvexHull(v).volume

        # Scale Box to initial packing fraction
        scale = ((total_vol / sys_prep_pf) / self.unitcell.box.volume)**(1/3)
        l_params = self.unitcell.box.to_box_lengths_and_angles()
        new_params = [p * scale if idx < 3 else p for idx, p in enumerate(l_params)]
        uc_box = freud.box.Box.from_box_lengths_and_angles(*new_params)
        uc_pos = [p * scale for p in uc_pos]

          # Prepare shape data for HOOMD-HPMC
        shape_data = {} # Custom shape information
        for i in range(len(types)):
            N_particles = len([j for j in self.uc_particle_indices if self.system.typeids[j] == types[i]])*np.prod((1, 1, 1))  # Total number of particles in the system
            shape_data[types[i]] = {}
            shape_data[types[i]]["Num_particles"] = int(N_particles)
            shape_data[types[i]]["Vertices"] = shape_verts[i]
            shape_data[types[i]]["Volume"] = shape_vol[i]
            shape_data[types[i]]["typeid"] = types[i]
            # shape_data[types[i]]["types"] = types[i*2:(i+1)*2]  # Mobile and fixed types for this basis type
            shape_data[types[i]]["types"] = types[:]


        # Initial GSD Write
        sys_prep_pf_str = str(sys_prep_pf).replace(".", "p")
        gsd_path = os.path.join(self.temp_dir, f"{self.shape_id}_uc_pf{sys_prep_pf_str}.gsd")
        writer = gsd_writer.GSDWriter(filename=gsd_path)
        writer.write_GSD(uc_box, uc_pos, uc_orient, typeids, types, shape_meta, repeat=(1, 1, 1))

        # HPMC Compression
        comm = hoomd.communicator.Communicator()
        overlaps, num_parts = 0, len(uc_pos)
        num_parts, shape_data, overlaps = hoomd_quickcompress.hoomd_quickcompress_func(
            self.temp_dir, len(uc_pos), shape_data, comm, self.shape_id, quickCompress=False,
            target_pf=target_pf, sys_prep_pf=sys_prep_pf, patches="False", translational_spring="True"
        )

        if overlaps == 0:
            logger.info(f"Compression successful. Proceeding to NVT at PF={target_pf}")
            hpmc_nvt.nvt(self.temp_dir, num_parts, shape_data, comm, self.shape_id, floppy_nvt=False, target_pf=target_pf,
                          translational_spring="True", end_step=50_000, write_period=500)
        else:
            logger.error("Simulation aborted: Overlaps detected after compression.")

    
