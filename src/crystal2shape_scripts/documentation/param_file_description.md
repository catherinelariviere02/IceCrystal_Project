# Parameter Configuration Guide (`param_file.json`)

This document describes all parameters available in the configuration file `param_file.json`, which governs crystal structure generation, symmetry analysis, information polyhedron calculation, and HOOMD-blue HPMC simulations.

## File and Path Settings

*   **`directory`** *(string)*: Relative or absolute path to the directory containing input CIF files.
    *   *Example:* `"../input_files/input_cif/orthorhombic/"`
*   **`input_CIF`** *(string)*: Filename of the target input CIF file.
    *   *Example:* `"64_Ga_0.cif"`
*   **`space_group_number`** *(integer)*: International Space Group number representing the crystal structure symmetry.
    *   *Example:* `64`
*   **`sg_chiral`** *(string boolean: `"True"` or `"False"`)*: Indicates whether the space group is chiral.
    *   *Example:* `"False"`

## Particle and Type Definitions

*   **`type_arr`** *(list of strings)*: Chemical symbols of the atomic species present in the crystal unit cell.
    *   *Example:* `["Ga"]`
*   **`radii_arr`** *(list of integers/floats)*: Atomic sizes corresponding to species in `type_arr` (often in pm), used to determine relative pairwise truncation ratios.
    *   *Example:* `[127]`
*   **`atom_type_selection`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, restricts local neighborhood shell calculations to only include particles of the same atomic type as the reference particle.
    *   *Example:* `"True"`

## System Replication and Initialization

*   **`num_replicas`** *(integer)*: Multiplier for replicating the unit cell box along the Cartesian axes (producing a $N \times N \times N$ system) to avoid self-interaction in bulk calculations.
    *   *Example:* `5`
*   **`repeat`** *(list of three integers)*: Explicit replication factor for building/visualizing lattices.
    *   *Example:* `[1, 1, 1]`
*   **`create_struc`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, generates the crystal structure from space group generators instead of reading a CIF file.
    *   *Example:* `"False"`
*   **`use_CIF`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, parses `input_CIF` to initialize the coordinates and lattice parameters.
    *   *Example:* `"True"`

## Radial Distribution Function (RDF) Settings

*   **`show_rdf`** *(string boolean: `"True"` or `"False"`)*: Enables generating and saving the Radial Distribution Function plots.
    *   *Example:* `"True"`
*   **`rmax`** *(float)*: Maximum search radius for neighbor queries in RDF computation.
    *   *Example:* `3`
*   **`bins`** *(integer)*: Number of bins used to discretize distance intervals in the RDF histogram.
    *   *Example:* `500`

## Tolerances and Rounding Factors

*   **`rd_factor`** *(integer)*: Decimal rounding precision for float distance comparisons and candidate identification (e.g. `3` rounds values to 3 decimal places).
    *   *Example:* `3`
*   **`pg_tolerance`** *(float)*: Point group symmetry matching tolerance passed to Pymatgen `PointGroupAnalyzer`.
    *   *Example:* `1e-2`
*   **`hierarchy_tol`** *(float)*: Distance tolerance for grouping neighbors into radial shells/hierarchy layers. Neighbors with distance differences smaller than this tolerance are considered to belong to the same shell.
    *   *Example:* `0.0`
*   **`d_tol`** *(float)*: Distance offset threshold used for matching equivalent atom distances.
    *   *Example:* `0.05`
*   **`length_contraction`** *(float)*: Length scale contraction factor used to adjust boundary distances.
    *   *Example:* `1.0`

## Polyhedron Truncation Settings

*   **`tr_pt`** *(list of floats)*: Self-truncation ratios defining where plane cuts are made along neighbor normal vectors. Ratios range from 0.0 (cut at the reference particle) to 1.0 (cut at the neighbor center).
    *   *Example:* `[0.5]`
*   **`target_volume`** *(list of floats)*: Scaling factor for final polyhedral volumes for each type. A value of `1.0` indicates scaling to unit volume.
    *   *Example:* `[1.0]`
*   **`use_hierarchy_to_construct_shape`** *(string boolean: `"True"` or `"False"`)*: Enables utilizing hierarchical coordination shells for constructing the final truncated shapes.
    *   *Example:* `"False"`

## HOOMD Simulation Parameters

*   **`create_unitcell`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, writes the initial configuration GSD file and triggers HPMC simulations.
    *   *Example:* `"False"`
*   **`sys_prep_pf`** *(float)*: Starting packing fraction of the simulated box.
    *   *Example:* `0.6`
*   **`target_pf`** *(float)*: Target packing fraction for HPMC compression and NVT runs.
    *   *Example:* `0.63`

## Visualization Flags

*   **`show_unitcell`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, saves a PyVista rendering of the crystal unit cell.
    *   *Example:* `"False"`
*   **`show_hierarchy`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, renders and saves the shell coordination bonds (hierarchical connections).
    *   *Example:* `"False"`
*   **`show_voronoi`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, renders the Voronoi polyhedron cells.
    *   *Example:* `"False"`
*   **`show_truncation`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, opens or saves interactive step-wise plane-cut rendering.
    *   *Example:* `"False"`
*   **`show_basis`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, highlights the chosen asymmetric unit basis points.
    *   *Example:* `"False"`
*   **`replicated_asym_unit_show`** *(string boolean: `"True"` or `"False"`)*: If `"True"`, plots the replicated asymmetric units to verify coverage check.
    *   *Example:* `"False"`
