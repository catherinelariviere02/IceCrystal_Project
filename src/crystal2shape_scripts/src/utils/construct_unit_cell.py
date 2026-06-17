import numpy as np
import rowan, freud, coxeter, math
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from scipy.linalg import orth
from scipy.spatial.transform import Rotation as Rot
from itertools import permutations
from itertools import chain


from ..symmetry import detect_pg
from ..visualization.pyvista_plot import pyvista_plot_class
from .utils_func import safe_input

class ConstructUnitCell():
    def __init__(self, positions, all_positions, hierarchy_coords, box_arr, box, target_pf, hierarchy, particle_arr, vertices_arr,
                 uc_ids, type_arr, basis_type, sys_types, frac_positions, target_volume):
        
        """
        Initialize the ConstructUnitCell class.

        Parameters
        ----------
        positions : np.ndarray
            The positions of the particles in the unit cell.

        all_positions : np.ndarray
            The positions of all particles in the system.

        hierarchy_coords: list
            A list of lists containing the coordinates of the hierarchy of particles.

        box : freud.Box
            The simulation box containing the unit cell.

        target_pf : float
            The target packing fraction for the unit cell.

        hierarchy : list
            A list of lists containing the hierarchy of particles.

        particle_arr : list
            A list of particle indices corresponding to the hierarchy.

        vertices_arr : list
            A list of vertices for each particle in the hierarchy.

        uc_ids : list
            A list of unit cell IDs corresponding to the particles in the hierarchy.

        type_arr : list
            A list of particle types in the unit cell.

        basis_type : str
            The type of basis used for the unit cell.

        sys_types : list
            A list of system types corresponding to the particles in the unit cell.

        frac_positions : np.ndarray     
            The fractional positions of the particles in the unit cell.

        target_volume : list
            The target volume for the unit cell, typically defined as [Lx, Ly, L

        Attributes
        ----------
        vertices : list
            The vertices of the convex polyhedron formed by the particles in the unit cell.

        orientations : list
            The orientations of the particles in the unit cell, represented as quaternions.

        box : freud.Box
            The simulation box containing the unit cell, adjusted to the target packing fraction.

        positions : np.ndarray
            The positions of the particles in the unit cell, adjusted to the target packing fraction.

        types : list
            A list of particle types in the unit cell.

        typeids : list
            A list of type IDs corresponding to the particle types in the unit cell.

        shape_dic : dict
            A dictionary containing the shape information of the convex polyhedron formed by the particles in the unit cell.

        type_arr: list
                List of atom types in the system

        Notes
        -----
        This class is used to construct a unit cell from a set of particles, taking into account
        their positions, orientations, and the target packing fraction. It generates the necessary
        data structures to represent the unit cell, including the vertices of the convex polyhedron,
        the orientations of the particles, and the simulation box. The class also provides a method
        to generate all combinations of particles in the hierarchy based on their distances.
        The `construct_unit_cell_func` method is the main function that performs the construction
        of the unit cell, including the generation of neighbor points and their combinations.
        The `combinations_hierarchy` method generates all possible combinations of particles in the hierarchy
        based on their distances from a reference particle. It ensures that the combinations are unique
        and considers all possible arrangements of particles at each distance. This is crucial for constructing
        the rotational matrix that can lead the set of points to another one, as the correct arrangement
        of particles is necessary for achieving the desired packing fraction and orientation.
        The class also includes functionality to plot the hierarchical level-1 polyhedra using the `pyvista_plot_class`.
    
        """

        self.positions = positions
        self.all_positions = all_positions
        self.hierarchy_coords = hierarchy_coords
        self.box_arr = box_arr
        self.box = box
        self.target_pf = target_pf
        self.hierarchy = hierarchy
        self.particle_arr = particle_arr
        self.vertices_arr = vertices_arr
        self.uc_ids = uc_ids
        self.type_arr = type_arr
        self.basis_type = basis_type
        self.sys_types = sys_types
        self.frac_positions = frac_positions
        self.target_volume = target_volume

    def combinations_hierarchy(self, neighbor_points):
        """
        Generate all combinations of particles in the hierarchy.
        This function takes a list of neighbor points and generates all possible combinations

        1. First sort all points based on the distances
        2. For the points at a specific distances from the reference particle, generate all possible combinations
           as we need to create a rotational matrix that can lead the set of points to another one.
        3. We have to consider which point of set-1 will coincide which point of set-2 in the two point sets, that's why sorting based on the distances is important.
        4. As we have to create a rotational matrix, leading to the all possible combinations of the row/column of the matrix
           we don't know which one would work
        5. If a particular rotational matrix doesn't work, possibly the alteration of row1/2 would work, that's why all possible combinations would be required
        6. For which rotational matrix, the distance metric is zero, that's the right one

        Parameters
        ----------
        neighbor_points : list
            A list of lists containing the neighbor points for each particle in the hierarchy.

        Returns
        -------
        neighbor_points : list
            A modified list of neighbor points containing all possible combinations of points at each distance.

        """
        neighbor_points_modified = []
        for p in range(len(neighbor_points)):
            cons_points_dist_ = [round(float(np.linalg.norm(neighbor_points[p][s])), 3) for s in range(len(neighbor_points[p]))]
            original_dist_arr = np.array(cons_points_dist_)
            cons_points_dist_.sort()
            indices = [[e for e, d in enumerate(original_dist_arr) if d == cons_points_dist_[s]] for s in range(len(cons_points_dist_))]
            points_temp = []
            track_arr = []
            for j in range(len(indices)):
                if indices[j] not in track_arr:
                    if len(indices[j]) > 1:
                        permutation_list = []
                        for k in range(len(indices[j])):
                            permutation_list.append(np.array(neighbor_points[p][indices[j][k]]))

                        points_temp.append(permutation_list)

                    else:
                        points_temp.append([np.array(neighbor_points[p][indices[j][0]])])

                    track_arr.append(indices[j])

            points_ = []
            for k in range(len(points_temp)):
                if len(points_temp[k]) > 1:
                    points_sub = []
                    for m in permutations(points_temp[k], len(points_temp[k])):
                        points_sub.append(list(m))

                    points_.append(points_sub)
                else:
                    points_.append([points_temp[k]])

            # Concatenate all possible combinations of points
            conc_points = []
            if len(points_) == 1:
                conc_points = points_[0]
            else:
                for k in range(len(points_)):
                    for l in range(len(points_)):
                        if k != l:
                            for m in range(len(points_[k])):
                                for n in range(len(points_[l])):
                                    conc_points.append(np.concatenate((points_[k][m], points_[l][n])))
            

            neighbor_points_modified.append(conc_points)

        neighbor_points = neighbor_points_modified[:]

        return neighbor_points

    def construct_unit_cell_func(self):
        uc_positions, uc_orientations, vertices_arr = [], [], []
        for t in range(len(self.type_arr)):
            particle_ids = self.particle_arr[t]  
            hierarchy_cons = self.hierarchy[t]
            for p in particle_ids:
                uc_positions.append((self.all_positions[p]))

            # Level-1 of the hierarchy
            lvl = safe_input("Enter level : ", expected_type=int, range_bounds=(1, None))
            neighbor_points = ([[(self.hierarchy_coords[t][p][s] - self.hierarchy_coords[t][p][hierarchy_cons[p][0][0]]) for s in list(set([g[lvl-1] for g in hierarchy_cons[p] if len(g) >= lvl]))] for p in range(len(particle_ids))])
            
            # Modify the neighbor hierarchy so that it does not contain any extra points
            '''rcut = 1.2
            neighbor_points = [[neighbor_points[p][s] for s in range(len(neighbor_points[p])) if round(float(np.linalg.norm(neighbor_points[p][s])), 2) <= rcut] for p in range(len(neighbor_points))]
            '''
            neighbor_points = self.combinations_hierarchy(neighbor_points)

            index = 0 # reference particle index to compare with
            hull = ConvexHull(neighbor_points[index][0])
            ref_verts = [neighbor_points[index][0][i] for i in hull.vertices]  # reference vertices of the polyhedron
            poly = coxeter.shapes.ConvexPolyhedron(ref_verts)
            detect_pg_obj = detect_pg.detect_pg_class()
            detect_pg_obj.detect_pg_func(neighbor_points[index][0], len(poly.edges), len(poly.faces))
            invariant_quaternions = detect_pg_obj.invariant_quaternions

            # First sort out the vertices order where the transformation would be
            # It is important to finalize the order of the eigen vectors
            # as the order of the eigen vectors would lead to the correct rotational matrix
            for v in range(len(neighbor_points)):
                s1 = (neighbor_points[index][0])
                s2_val, q_arr = [], []
                for s2 in neighbor_points[v]:
                    if len(s2) == len(s1):
                        rot, rssd = Rot.align_vectors(s2, s1)
                        R = rot.as_matrix()
                        q = rowan.from_matrix(R)
                        s2_val.append(rssd)
                        q_arr.append(q)

                if len(s2_val) > 0:
                    min_rssd = min(s2_val)
                    index_temp = [i for i, j in enumerate(s2_val) if j == min_rssd]
                    # print("Index of minimum RSSD: ", index_temp)
                    s2 = neighbor_points[v][index_temp[0]]

                    # Once the final order of the vertices is determined, apply the quaternion to the vertices 
                    # to find the minimum angle and the corresponding quaternion to rotate one set of vertices to the other
                    s1_eqv = [rowan.rotate(t, s1) for t in invariant_quaternions]
                    angle_arr, q_arr = [], []
                    for s1_p in s1_eqv:
                        if len(s2) == len(s1_p):
                            rot, rssd = Rot.align_vectors(s2, s1_p)
                            R = rot.as_matrix()
                            q = rowan.from_matrix(R)
                            axis_angle = rowan.to_axis_angle(q)
                            axis, angle = axis_angle[0][0], round(float((np.rad2deg(axis_angle[1][0]))), 2)
                            angle_arr.append(angle)
                            q_arr.append(q)

                    if len(angle_arr) > 0:
                        min_angle = min(angle_arr)
                        index_temp = [i for i, j in enumerate(angle_arr) if j == min_angle]
                        # print("Index of minimum angle: ", index_temp)
                        # print("Minimum angle: ", min_angle)
                        q = q_arr[index_temp[0]] # The final quaternion to rotate the vertices

                        axis_angle = rowan.to_axis_angle(q)
                        axis, angle = axis_angle[0][0], round(float((np.rad2deg(axis_angle[1][0]))), 2)
                        # print(axis, angle, round(min_rssd, 2))
                        uc_orientations.append(q.tolist())

                        # Plotting hierarchical level-1 polyhedra
                        '''uc_box_list = self.box.to_matrix() # Convert freud.Box to numpy array
                        # line_points = [[self.all_positions[self.uc_ids[v]].tolist(), (axis + self.all_positions[self.uc_ids[v]]).tolist()]]
                        posi_uc = [self.all_positions[j] for j in self.uc_ids]
                        poly_vertices = [rowan.rotate(q, s1)+ self.all_positions[particle_ids[v]], s2 + self.all_positions[particle_ids[v]]]
                        pyvista_plot_obj = pyvista_plot_class(uc_box_list=uc_box_list, positions=posi_uc, lattice_sites=None, poly_vertices=poly_vertices, extra_positions=None, color_arr=None, line_points=None)
                        pyvista_plot_obj.pyvista_plot_func()'''

        uc = freud.data.UnitCell(self.box, basis_positions=self.frac_positions)
        n_repeats = 1
        (uc_box, uc_positions) = uc.generate_system(num_replicas=n_repeats, scale=1.0)
        N = np.prod(n_repeats)
        indices_uc = np.repeat(np.arange(len(uc.basis_positions)), N)
        
        # Set positions, box at target_pf
        '''uc_box_matrix = uc_box.to_matrix()
        uc_box_volume = uc_box.volume
        uc_box_arr = [uc_box.Lx, uc_box.Ly, uc_box.Lz, uc_box.xy, uc_box.xz, uc_box.yz]
        particle_volume = np.sum(np.array([len(self.particle_arr[t])*self.target_volume[t] for t in range(len(self.particle_arr))]))
        target_volume = particle_volume/self.target_pf
        uc_box = freud.box.Box(Lx=uc_box_arr[0]* (target_volume/uc_box_volume) ** (1/3),
                               Ly=uc_box_arr[1]* (target_volume/uc_box_volume) ** (1/3),
                               Lz=uc_box_arr[2]* (target_volume/uc_box_volume) ** (1/3),
                               xy=uc_box_arr[3],
                               xz=uc_box_arr[4],
                               yz=uc_box_arr[5])
        
        # modified_box_matrix = uc_box_matrix * (target_volume/uc_box_volume) ** (1/3)
        # uc_box = freud.box.Box.from_matrix(modified_box_matrix)
        uc_positions = uc_box.wrap(np.array(uc_positions)* (target_volume/uc_box_volume) ** (1/3))'''

        if self.vertices_arr != None:
            for t in range(len(self.type_arr)):
                shape_vertices = self.vertices_arr[t]
                # print("Shape vertices:", shape_vertices)
                vertices = shape_vertices[index]
                hull = ConvexHull(vertices)
                vertices = [vertices[i] for i in hull.vertices]  # reference vertices of the polyhedron
                vertices_arr.append(vertices)

            types = self.type_arr[:]
            typeids = np.repeat(np.array([0 for _ in range(len(uc_positions))]), np.prod(n_repeats), axis=0).tolist()  # Assuming two types of particles, adjust as necessary

            shape_dic_arr = []
            for v in range(len(vertices_arr)):
                verts = [(t - np.average(vertices_arr[v], axis=0)).tolist() for t in vertices_arr[v]]
                shape_dic = {
                    "type": "ConvexPolyhedron",
                    "rounding_radius": 0.0,
                    "vertices": verts
                }
                shape_dic_arr.append(shape_dic)

            self.vertices_arr = vertices_arr
            self.types = types
            self.typeids = typeids
            self.shape_dic = shape_dic_arr


        self.orientations = uc_orientations
        self.box = uc_box
        self.positions = uc_positions
        

    