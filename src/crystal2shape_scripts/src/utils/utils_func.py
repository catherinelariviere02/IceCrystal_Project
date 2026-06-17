"""
Utility Functions Module
========================

This module provides common geometric, algebraic, and crystallographic helper functions used across 
the Crystal2Shape pipeline. It includes utilities for point group analysis, particle coordination, 
moment of inertia calculations, and dual polyhedron construction.
"""

import math
import freud
from mpld3 import show
import numpy as np
import coxeter, rowan
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from ..operations.pointgroup_sym import PointGroup

def _get_point_group(positions, 
                    species_label, 
                    pg_tolerance=0.1):
    """Calculates the point group symmetry operations map of a set of coordinates.

    Uses the internal PointGroup analyzer class.

    Args:
        positions (np.ndarray or List[List[float]]): 3D coordinates of the points.
        species_label (List[str]): Chemical species labels matched with positions.
        pg_tolerance (float): Tolerance for point group symmetry detection. Defaults to 0.1.

    Returns:
        Dict[float, List[List[float]]]: Map of symmetry operations (e.g., angles to list of axes).
    """
    pointgroup_obj = PointGroup()
    pointgroup_obj.analyze(coords=positions, 
                                   rtol=0.1, 
                                   atol=1.0)
    
    symm_dic = pointgroup_obj.symm_dic

    return symm_dic


def _get_particles_uc(system, 
                      unitcell, 
                      uc_particle_indices, 
                      particle_ids):
    """Maps arbitrary system particle IDs to equivalent particle IDs within the reference unit cell.

    Uses periodic boundary box wrapping to compute closest matches.

    Args:
        system (gsd.hoomd.Snapshot): The system snapshot.
        unitcell (freud.data.UnitCell): The unit cell object.
        uc_particle_indices (List[int]): Particle indices representing the base unit cell.
        particle_ids (List[int] or np.ndarray): Target particle IDs to map.

    Returns:
        Tuple[List[int], List[int]]: A tuple containing:
            - List of corresponding unit cell particle indices.
            - List of the original unique particle IDs.
    """
    particle_ids_uc_wrapped = np.array([unitcell.box.wrap(system.positions[t]) for t in particle_ids])
    uc_particle_positions = np.array([system.box.wrap(system.positions[t]) for t in uc_particle_indices])
    unique_particle_ids = []
    particle_ids_uc = []
    for k in range(len(particle_ids_uc_wrapped)):
        dist_arr = []
        for (i, e) in enumerate(uc_particle_positions):
            dist_arr.append(np.linalg.norm(unitcell.box.wrap(e - particle_ids_uc_wrapped[k])))

        min_index = np.argmin(dist_arr)
        if uc_particle_indices[min_index] not in particle_ids_uc:
            particle_ids_uc.append(uc_particle_indices[min_index])
            unique_particle_ids.append(particle_ids[k])

    return particle_ids_uc, unique_particle_ids

def _get_pointgroup_neighbors_of_neighbors(system, 
                              unitcell, 
                              rcut,
                              ref_particle, 
                              uc_particle_indices, 
                              atom_type=None,
                              rtol=1e-2,
                              pg_tolerance=0.1,
                              extra_positions=None, 
                              extra_particles_types=None,
                              extra_particles_wyckoffs=None,
                              show=False,
                              directory=None,
                              shape_id=None,
                              dist_map=None,
                              length_contraction=1.0) -> str:
    """Calculates the point group symbol of the neighborhood formed by neighbors of neighbor particles.

    Used to assess point group recovery around a reference particle.

    Args:
        system (gsd.hoomd.Snapshot): The system snapshot.
        unitcell (freud.data.UnitCell): The unit cell object.
        rcut (float): Active cutoff distance.
        ref_particle (int): Reference particle ID.
        uc_particle_indices (List[int]): Particle indices in the unit cell.
        atom_type (Optional[str]): Atom type filter. Defaults to None.
        rtol (float): Relative tolerance for floating-point checks. Defaults to 1e-2.
        pg_tolerance (float): Point group analysis tolerance. Defaults to 0.1.
        extra_positions (Optional[List[List[float]]]): Additional coords outside the unit cell.
        extra_particles_types (Optional[List[str]]): Atom types of extra positions.
        extra_particles_wyckoffs (Optional[List[str]]): Wyckoff positions of extra positions.
        show (bool): If True, plots the partial BOD and polyhedron. Defaults to False.
        directory (Optional[str]): Output directory path for plots. Defaults to None.
        shape_id (Optional[str]): Identifier prefix for saving files. Defaults to None.
        dist_map (Optional[Dict[str, float]]): Map of maximum distances. Defaults to None.
        length_contraction (float): Length contraction factor. Defaults to 1.0.

    Returns:
        str: Schoenflies point group symbol.
    """
    # Neighbors of neighnbor particles within rcut
    from skspatial.objects import Points
    from scipy.spatial import ConvexHull
    rd_factor = int(math.log10(1/rtol))

    max_dist_ref_particle = dist_map.get(f'{system.equiv_atoms[ref_particle]}_{system.typeids[ref_particle]}', 0.0)
    # Get invariant and equivalent points  within max_len from the reference particle
    equivalent_invariant_points = _get_equivalent_invariant_points(system, unitcell, ref_particle, uc_particle_indices, atom_type, 
                                                                       round(max_dist_ref_particle, rd_factor), rtol, pg_tolerance=pg_tolerance)
    
    pg_symm_inv_points = equivalent_invariant_points[1]
    pg_symm_inv_points, _ = _get_particles_uc(system, unitcell, uc_particle_indices, pg_symm_inv_points)

    neighbor_coords, particle_ids, transformed_poly_vertices = [], [], []
    # Get neighbor of neighbors coordinates of the reference particle within rcut
    for i in uc_particle_indices:
        if system.equiv_atoms[i] == system.equiv_atoms[ref_particle] and system.typeids[i] == system.typeids[ref_particle]: # Only consider those particles that are in the same Wyckoff position as the reference particle
            dist_ = dist_map.get(f'{system.equiv_atoms[i]}_{system.typeids[i]}', 0.0)
            neigh_, neigh_coords = _get_particle_ids_coords_rcut(
                        system, unitcell, i, uc_particle_indices, rcut, rtol=rtol
                    )

            for (j, coord) in enumerate(neigh_coords):
                nei_coord = np.round(system.box.wrap(coord), decimals=rd_factor).tolist()
                transformed_poly_vertices.append(neigh_coords + system.positions[i])
                if nei_coord not in neighbor_coords: 
                    neighbor_coords.append(nei_coord)
                    particle_ids.append(neigh_[j])

    try:        
        neighbor_coords = np.array(neighbor_coords)
        neighbor_ids = particle_ids[:]
        species_levels = [system.typeids[i] for i in neighbor_ids]

        # Calculate PG
        mol = Molecule(species=species_levels, coords=neighbor_coords)
        pga = PointGroupAnalyzer(mol, tolerance=pg_tolerance)

        # BOD Visualization
        if show == True:
            # See rotated polyhedron
            outfilename = directory + "temp_files/" + shape_id + "_final_shape.png"
            from ..visualization.pyvista_plot import pyvista_plot_class
            poly_vertices = np.array(transformed_poly_vertices)
            pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_particle_indices],
                                types=[system.typeids[j] for j in uc_particle_indices], 
                                wyckoff_uc=[system.wyckoffs[j] for j in uc_particle_indices],
                                poly_vertices=poly_vertices,
                                line_points=None,
                                text="Final Shape Polyhedron",
                                filename=None,
                                color="slate_grey").pyvista_plot_func()
            
            # Partial BOD visualization
            filename=directory + "temp_files/" + shape_id + "_partial_bod.png"
            import pyvista as pv
            pl = pv.Plotter()
            pl.add_mesh(pv.Sphere(radius=rcut, center=(0, 0, 0)), color="black", lighting=True, roughness=0.0, metallic=1.0, opacity=0.8)
            pl.add_points(np.array(neighbor_coords), render_points_as_spheres=True, point_size=20, color="red", lighting=True, roughness=0.0, metallic=1.0)
            light = pv.Light(color='white', light_type='headlight')
            pl.add_text("pBOD", position='upper_edge', font_size=14, color='black', shadow=True)
            pl.add_light(light)
            pl.enable_ssao()
            pl.set_background("white")
            pl.show()
            pl.close()

        return pga.sch_symbol
    
    except Exception:
        return "None", "None"
    
def _get_pointgroup_neighbors_of_neighbors_v1(system, 
                              unitcell, 
                              rcut,
                              ref_particle, 
                              uc_particle_indices, 
                              atom_type=None,
                              rtol=1e-2,
                              pg_tolerance=0.1,
                              extra_positions=None, 
                              extra_particles_types=None,
                              extra_particles_wyckoffs=None,
                              show=False,
                              directory=None,
                              shape_id=None,
                              dist_map=None,
                              length_contraction=1.0) -> str:
    """Alternative version of `_get_pointgroup_neighbors_of_neighbors`.

    Applies crystallographic PG symmetry operations on the local neighbors of
    the reference particle to calculate point group symbol of the environment.

    Args:
        system (gsd.hoomd.Snapshot): The system snapshot.
        unitcell (freud.data.UnitCell): The unit cell object.
        rcut (float): Cutoff radius.
        ref_particle (int): Reference particle ID.
        uc_particle_indices (List[int]): Particle indices in the unit cell.
        atom_type (Optional[str]): Atom type filter. Defaults to None.
        rtol (float): Relative tolerance. Defaults to 1e-2.
        pg_tolerance (float): Point group analysis tolerance. Defaults to 0.1.
        extra_positions (Optional[List[List[float]]]): Additional coords.
        extra_particles_types (Optional[List[str]]): Atom types of extra positions.
        extra_particles_wyckoffs (Optional[List[str]]): Wyckoff positions of extra positions.
        show (bool): If True, plots partial BOD and polyhedron. Defaults to False.
        directory (Optional[str]): Output directory path. Defaults to None.
        shape_id (Optional[str]): Identifier prefix. Defaults to None.
        dist_map (Optional[Dict[str, float]]): Max distances map. Defaults to None.
        length_contraction (float): Length contraction factor. Defaults to 1.0.

    Returns:
        str: Schoenflies point group symbol.
    """
    # Neighbors of neighnbor particles within rcut
    from skspatial.objects import Points
    from scipy.spatial import ConvexHull
    rd_factor = int(math.log10(1/rtol))

    max_dist_ref_particle = dist_map.get(f'{system.equiv_atoms[ref_particle]}_{system.typeids[ref_particle]}', 0.0)
    neigh_, neigh_coords = _get_particle_ids_coords_rcut(
                    system, unitcell, ref_particle, uc_particle_indices, rcut, rtol=rtol
                )
    
    neigh_.append(ref_particle)

    rotated_coords, transformed_poly_vertices = [], []
    for op in unitcell.pg_symmetry:
        rotation_matrix = op
        rotated_coord = [np.round(np.dot(rotation_matrix, t), decimals=rd_factor).tolist() for t in neigh_coords]
        transformed_poly_vertices.append(rotated_coord)
        for rc in rotated_coord:
            if rc not in rotated_coords:
                rotated_coords.append(rc)

    try:        
        neighbor_coords = np.array(rotated_coords)
        species_levels = ["C" for i in range(len(rotated_coords))]

        # Calculate PG
        mol = Molecule(species=species_levels, coords=neighbor_coords)
        pga = PointGroupAnalyzer(mol, tolerance=pg_tolerance)

        # BOD Visualization
        if show == True:
            # See rotated polyhedron
            outfilename = directory + "temp_files/" + shape_id + "_final_shape.png"
            from ..visualization.pyvista_plot import pyvista_plot_class
            poly_vertices = transformed_poly_vertices + system.positions[ref_particle]
            pyvista_plot_class(uc_box=unitcell.box, positions=[system.positions[j] for j in uc_particle_indices],
                                types=[system.typeids[j] for j in uc_particle_indices], 
                                wyckoff_uc=[system.wyckoffs[j] for j in uc_particle_indices],
                                poly_vertices=poly_vertices,
                                line_points=None,
                                text="Final Shape Polyhedron",
                                filename=None,
                                color="teal").pyvista_plot_func()

            filename=directory + "temp_files/" + shape_id + "_partial_bod.png"
            import pyvista as pv
            pl = pv.Plotter()
            pl.add_mesh(pv.Sphere(radius=rcut, center=(0, 0, 0)), color="black", lighting=True, roughness=0.0, metallic=1.0, opacity=0.8)
            pl.add_points(np.array(neighbor_coords), render_points_as_spheres=True, point_size=20, color="red", lighting=True, roughness=0.0, metallic=1.0)
            light = pv.Light(color='white', light_type='headlight')
            pl.add_text("pBOD", position='upper_edge', font_size=14, color='black', shadow=True)
            pl.add_light(light)
            pl.enable_ssao()
            pl.set_background("white")
            pl.show()
            pl.close()

        return pga.sch_symbol
    
    except Exception:
        return "None"


def _get_particle_ids_coords_rcut(system,
                                  unitcell,
                                  particle,
                                  uc_particle_indices, 
                                  rcut,
                                  rtol=1e-2):
    """Retrieves particle IDs and their relative coordinates within a specified cutoff distance.

    Enforces lattice-based boundary checks to ensure points are correctly bounded.

    Args:
        system (gsd.hoomd.Snapshot): The system snapshot.
        unitcell (freud.data.UnitCell): The unit cell object.
        particle (int): Central reference particle index.
        uc_particle_indices (List[int]): Particle indices in the unit cell.
        rcut (float): Cutoff radius.
        rtol (float): Relative tolerance for floating-point comparisons. Defaults to 1e-2.

    Returns:
        Tuple[List[int], List[np.ndarray]]: A tuple containing:
            - List of particle indices within the cutoff.
            - List of relative coordinate vectors wrapped within the system box.
    """
    rd_factor = int(math.log10(1/rtol))
    lattice_parameters = unitcell.box.to_box_lengths_and_angles()
    uc_lattice_vectors_unit = [t/np.linalg.norm(t) for t in unitcell.lattice_vectors]

    particle_ids = [int(j) for j in range(len(system.positions)) if (rcut - round(float(np.linalg.norm(system.box.wrap(system.positions[j] - system.positions[particle]))), rd_factor)) >= 0 and
                            round(float(lattice_parameters[0]) - max([float((np.dot((system.positions[j] - system.positions[particle]), uc_lattice_vectors_unit[0]))), float((np.dot((system.positions[j] - system.positions[particle]), -uc_lattice_vectors_unit[0])))]), rd_factor) >= 0 and
                            round(float(lattice_parameters[1]) - max([float((np.dot((system.positions[j] - system.positions[particle]), uc_lattice_vectors_unit[1]))), float((np.dot((system.positions[j] - system.positions[particle]), -uc_lattice_vectors_unit[1])))]), rd_factor) >= 0 and
                            round(float(lattice_parameters[2]) - max([float((np.dot((system.positions[j] - system.positions[particle]), uc_lattice_vectors_unit[2]))), float((np.dot((system.positions[j] - system.positions[particle]), -uc_lattice_vectors_unit[2])))]) , rd_factor) >= 0 and
                            j != particle]
    
    particle_coords =([system.box.wrap(system.positions[j] - system.positions[particle]) for j in particle_ids])

    return particle_ids, particle_coords

def _get_pointgroup_symmetry_equivalent_invariant_points(coordinates, 
                                                         particle_ids,
                                                         species_labels, 
                                                         pg_tolerance=0.1):
    """Identifies symmetry-equivalent and invariant points using Pymatgen PointGroupAnalyzer.

    Args:
        coordinates (np.ndarray or List[List[float]]): Relative coordinates of the points.
        particle_ids (List[int]): Corresponding particle IDs.
        species_labels (List[str]): Species labels of each point.
        pg_tolerance (float): Symmetry matching tolerance. Defaults to 0.1.

    Returns:
        Tuple[List[int], List[int], List[Any], PointGroupAnalyzer]: A tuple containing:
            - List of symmetry-equivalent particle IDs.
            - List of symmetry-invariant particle IDs.
            - List of symmetry operations.
            - The PointGroupAnalyzer object instance.
    """
    mol = Molecule(species=species_labels, coords=coordinates)
    pga = PointGroupAnalyzer(mol, tolerance=pg_tolerance)
    equivalent_atoms_info = pga.get_equivalent_atoms()
    equivalent_atoms_indices = equivalent_atoms_info["eq_sets"]
    equivalent_atoms_src_indices = list(equivalent_atoms_indices.keys())
    equivalent_atoms_dst_indices = list(equivalent_atoms_indices.values())
    symm_equiv_indices = equivalent_atoms_src_indices[:]
    symm_inv_indices = [equivalent_atoms_src_indices[i] for i in range(len(equivalent_atoms_src_indices)) if len(equivalent_atoms_dst_indices[i]) == 1]

    symm_equiv_ids = [particle_ids[i] for i in symm_equiv_indices]
    symm_inv_ids = [particle_ids[i] for i in symm_inv_indices]
    symm_ops = pga.get_symmetry_operations()

    return symm_equiv_ids, symm_inv_ids, symm_ops, pga

def _get_equivalent_invariant_points(system, unitcell, particle, uc_indices, atom_type, max_len, rtol, pg_tolerance=0.1):
    """Identifies symmetry-equivalent and invariant points within a maximum distance scale.

    Args:
        system (gsd.hoomd.Snapshot): The system snapshot.
        unitcell (freud.data.UnitCell): The unit cell object.
        particle (int): Central reference particle ID.
        uc_indices (List[int]): Particle indices in the unit cell.
        atom_type (str or List[str]): Species name(s) to filter by.
        max_len (float): Maximum cutoff distance.
        rtol (float): Relative distance tolerance.
        pg_tolerance (float): Point group analyzer tolerance. Defaults to 0.1.

    Returns:
        List[List[int]]: A list containing:
            - List of equivalent particle IDs.
            - List of invariant particle IDs.
    """
    particle_ids_all, particle_coords_all = _get_particle_ids_coords_rcut(
                system, unitcell, particle, uc_indices, max_len, rtol=rtol
    )
    particle_ids_all = [j for j in particle_ids_all if system.typeids[j] in atom_type]

    particle_ids_all.append(particle)
    particle_coords_all = np.array([system.box.wrap(system.positions[j] - system.positions[particle]) for j in particle_ids_all])

    # Get invariant particles under symmetry operations
    equiv_particle_ids, inv_particle_ids, symm_ops, pga = _get_pointgroup_symmetry_equivalent_invariant_points(
        coordinates=particle_coords_all,
        particle_ids=particle_ids_all,
        species_labels=[system.typeids[j] for j in particle_ids_all],
        pg_tolerance=pg_tolerance
    )
    
    equivalent_invariant_points = [equiv_particle_ids, inv_particle_ids]

    return equivalent_invariant_points

def _get_trans_orien_equivalent_invariant_points(system,
                                                unitcell,
                                                coordinates, 
                                                particle_ids,
                                                species_labels, 
                                                pg_tolerance=0.1):
    """Identifies translationally and orientationally equivalent points.

    Checks if translationally equivalent points have identical orientation alignments.

    Args:
        system (gsd.hoomd.Snapshot): The system snapshot.
        unitcell (freud.data.UnitCell): The unit cell object.
        coordinates (np.ndarray): Particle coordinates.
        particle_ids (List[int]): Particle indices.
        species_labels (List[str]): Species labels.
        pg_tolerance (float): Symmetry tolerance. Defaults to 0.1.

    Returns:
        List[int]: List of translationally and orientationally equivalent particle IDs.
    """
    mol = Molecule(species=species_labels, coords=coordinates)
    pga = PointGroupAnalyzer(mol, tolerance=pg_tolerance)
    equivalent_atoms_info = pga.get_equivalent_atoms()
    equivalent_atoms_indices = equivalent_atoms_info["eq_sets"]
    equivalent_atoms_src_indices = list(equivalent_atoms_indices.keys())
    equivalent_atoms_dst_indices = list(equivalent_atoms_indices.values())
    equivalent_atoms_dst_indices = [list(t) for t in equivalent_atoms_dst_indices]

    orientation_equiv_pts = []
    for ori in range(len(equivalent_atoms_dst_indices)):
        orien_list = [system.local_env_cluster_ids[particle_ids[i]] for i in equivalent_atoms_dst_indices[ori]]
        unique_orien = []
        for o in orien_list:
            if o not in unique_orien:
                unique_orien.append(o)
        
        for uo in unique_orien:
            orientation_equiv_pts.append(equivalent_atoms_dst_indices[ori][orien_list.index(uo)])

    symm_equiv_indices = orientation_equiv_pts[:]
    symm_equiv_ids = list(set([particle_ids[i] for i in symm_equiv_indices]))

    return symm_equiv_ids

def _moment_of_inertia(shape_poly):
    """Calculates the diagonalized principal moments of inertia of a polyhedron.

    Args:
        shape_poly (np.ndarray): 3D vertices defining the polyhedron.

    Returns:
        List[float]: A list of three diagonal components [Ixx, Iyy, Izz].
    """
    import coxeter
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(shape_poly)
    shape_poly = np.array([shape_poly[u] for u in hull.vertices])
    poly = coxeter.shapes.ConvexPolyhedron(shape_poly)
    poly.diagonalize_inertia()
    moi = poly.inertia_tensor
    moi_arr = [moi[0][0], moi[1][1], moi[2][2]]
    return moi_arr

def _covariance_matrix(points, vecs=None):
    """Computes the covariance matrix of a point cloud and extracts its eigenvalues.

    If reference eigenvectors are provided, projects the covariance onto those vectors.

    Args:
        points (np.ndarray): 3D coordinates.
        vecs (Optional[np.ndarray]): Pre-defined projection directions. Defaults to None.

    Returns:
        Tuple[List[float], np.ndarray]: A tuple containing:
            - List of eigenvalues (or projected values).
            - The eigenvector matrix.
    """
    import coxeter
    from scipy.spatial import ConvexHull

    cov_matrix = np.cov(points.T)
    if vecs is None:
        eig_val, eig_vecs = np.linalg.eig(cov_matrix)
    else:
        eig_val, eig_vecs = [], []
        for v in vecs.T:
            unit_v = v / np.linalg.norm(v)
            eig_val.append(np.dot(unit_v.T, np.dot(cov_matrix, unit_v)))

    return eig_val, eig_vecs

def _get_dual_polyhedron(shape_poly) -> np.ndarray:
    """Constructs the dual polyhedron from a given polyhedron shape.

    The dual vertices are calculated as face midpoints, centered at the origin,
    and scaled to match the volume of the original polyhedron.

    Args:
        shape_poly (np.ndarray or List[List[float]]): Original polyhedron vertices.

    Returns:
        np.ndarray: Vertices of the scaled dual polyhedron.
    """
    import coxeter
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(shape_poly)
    shape_poly = [shape_poly[u] for u in hull.vertices]
    poly = coxeter.shapes.ConvexPolyhedron(shape_poly)
    faces = poly.faces
    face_midpoints = [np.mean([shape_poly[v] for v in face], axis=0) for face in faces]
    dual_poly_vertices = face_midpoints

    target_volume = poly.volume
    dual_poly = coxeter.shapes.ConvexPolyhedron(dual_poly_vertices)
    dual_poly_vertices = np.array(dual_poly_vertices) * (target_volume / dual_poly.volume) ** (1/3)
    dual_poly_vertices = np.array(dual_poly_vertices - np.mean(dual_poly_vertices, axis=0))

    return dual_poly_vertices

def _plot(x, y, xlabel, ylabel, title):
    """Utility function to render and display a basic metric validation plot.

    Args:
        x (List[float]): X-axis values.
        y (List[float]): Y-axis values.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Plot title.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color='b', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()

def safe_input(prompt: str, expected_type: type = str, valid_choices: list = None, range_bounds: tuple = None):
    """Prompts user for input, validates type and choices/bounds, and handles interrupts gracefully.

    Args:
        prompt (str): The prompt string to display to the user.
        expected_type (type): The type to cast the input to (e.g., int, float, str). Defaults to str.
        valid_choices (list, optional): A list of valid values. Defaults to None.
        range_bounds (tuple, optional): A tuple of (min, max) bounds. Defaults to None.

    Returns:
        Any: Casted and validated user input.
    """
    import sys
    while True:
        try:
            val_str = input(prompt)
            if expected_type is int:
                val = int(val_str)
            elif expected_type is float:
                val = float(val_str)
            else:
                val = val_str

            if valid_choices is not None and val not in valid_choices:
                print(f"[ERROR] Invalid choice. Must be one of: {valid_choices}")
                continue

            if range_bounds is not None:
                min_v, max_v = range_bounds
                if min_v is not None and val < min_v:
                    print(f"[ERROR] Input must be at least {min_v}.")
                    continue
                if max_v is not None and val > max_v:
                    print(f"[ERROR] Input must be at most {max_v}.")
                    continue

            return val
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Execution interrupted by user. Exiting gracefully...")
            sys.exit(0)
        except ValueError:
            expected_name = expected_type.__name__
            print(f"[ERROR] Invalid input. Please enter a valid {expected_name}.")