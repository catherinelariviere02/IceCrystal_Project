import numpy as np
import os
import rowan, math
from . import face_edge_from_vertex

class point_group:
    """
    Class to calculate the point group symmetry of a crystal structure.
    The class uses the `face_edge_from_vertex` module to compute the edges and faces of the crystal structure
    based on the vertices provided. It then calculates the symmetry operations and invariant quantities of the structure.
    The class has a method `_point_group_func` that takes in vertices, number of edges, and number of faces
    and computes the point group symmetry.

    Parameters
    ----------
    vertices : list
        List of vertices of the crystal structure.

    num_edges : int
        Number of edges in the crystal structure.

    num_faces : int
        Number of faces in the crystal structure.

    Attributes
    ----------
    vertices : list
        List of vertices of the crystal structure.

    edges : list
        List of edges of the crystal structure.

    faces : list
        List of faces of the crystal structure.

    dic_data : dict
        Dictionary containing the symmetry operations and invariant quantities.

    invariant_quantity : list
        List of invariant quantities of the crystal structure.

    Methods
    -------
    _point_group_func(vertices, num_edges, num_faces)
        Compute the point group symmetry of the crystal structure.

    Notes
    -----
    The class uses the `rowan` library for quaternion calculations and the `numpy` library for numerical operations.
    The class also uses the `face_edge_from_vertex` module to compute the edges and faces of the crystal structure.
    The class is designed to be used as part of a larger program for analyzing crystal structures.
    The class is initialized with no parameters and the `_point_group_func` method is called with the vertices,
    number of edges, and number of faces of the crystal structure.
    The class is designed to be used as part of a larger program for analyzing crystal structures.

    """
    def __init__(self):
        pass

    def _point_group_func(self, vertices, num_edges, num_faces):
        edges, faces, vertices = face_edge_from_vertex.face_edge_from_vertex_func(vertices, num_edges, num_faces)

        tol = 3
        atol = 1e-3
        tolerance = 1
        center = np.mean(np.array([vertices[i] for i in range(len(vertices))]), axis=0).tolist()
        positions = []
        # Vertices
        for i in range(len(vertices)):
            vv = np.array(vertices[i]) - np.array(center)
            positions.append(vv.tolist())

        for i in range(len(faces)):
            vertice_arr = [vertices[faces[i][j]] for j in range(len(faces[i]))]
            face_midpt = np.mean(np.array(vertice_arr), axis=0)
            vv = face_midpt - np.array(center)
            positions.append(vv.tolist())

        # Center to face normal
        '''center_to_face_normal = []
        for i in range(len(faces)):
            polygon_vertices = [np.array(vertices[faces[i][j]]) for j in range(len(faces[i]))]
            n_cap = calc_normal.calc_normal_func(polygon_vertices)
            p_vec = vertices[faces[i][0]]
            d = -p_vec[0]*n_cap[0] - p_vec[1]*n_cap[1] - p_vec[2]*n_cap[2]
            dist_to_plane = np.dot(np.array(n_cap), np.array(center)) + d
            pt = np.array(center) - dist_to_plane*np.array(n_cap)
            center_to_face_normal.append(pt.tolist())

        for i in range(len(center_to_face_normal)):
            vv = np.array(center_to_face_normal[i])
            positions.append(vv.tolist())'''

        edges_arr = []
        for e in range(len(list(edges))):
            edges_arr.append(list(list(edges)[e]))

        # center to edges midpoint
        edge_midpt_arr = [np.mean(np.array([vertices[edges_arr[i][j]] for j in range(len(edges_arr[i]))]), axis=0) for i in range(len(edges_arr))]
        edges_len = [np.linalg.norm(np.array(vertices[edges_arr[i][0]]) - np.array(vertices[edges_arr[i][1]])) for i in range(len(edges_arr))]
        # print(edges_len)

        for i in range(len(edge_midpt_arr)):
            vv = np.array(edge_midpt_arr[i]) - np.array(center)
            positions.append(vv.tolist())

        #***************************************************************************

        mod_positions_all = [np.array(positions[i]).tolist() for i in range(len(positions))]
        rot_arr = [np.deg2rad(180), np.deg2rad(120), np.deg2rad(90), np.deg2rad(60)]

        ref_vertices = vertices[:]
        ref_vertices_rounded = []
        for i in range(len(vertices)):
            ref_vertices_rounded.append(np.array(ref_vertices[i]).tolist())

        final_q_arr, mod_positions_sym = [], []
        Arr = []
        dic_data = {}
        temp_data = {}

        for i in range(len(positions)):
            vecvec = positions[i]
            unit_vec = positions[i]/np.linalg.norm(positions[i])
            for j in range(len(rot_arr)):
                q = rowan.normalize(rowan.from_axis_angle(unit_vec, rot_arr[j]))
                q_rounded = [round(m, tol) for m in q]  # rounding off to 4 decimal places
                count = 0
                for k in range(len(ref_vertices)):
                    vec = rowan.rotate(q, ref_vertices[k])
                    vec_rounded = vec.tolist()
                    for t in range(len(ref_vertices_rounded)):
                        a = np.isclose(vec_rounded, ref_vertices_rounded[t], atol=atol)
                        a_true = [t for t in a if t==True]
                        if np.linalg.norm(vec_rounded) != 0 and np.linalg.norm(ref_vertices_rounded[t]) != 0:
                            angle = np.rad2deg(np.arccos(round(np.dot(vec_rounded/np.linalg.norm(vec_rounded), ref_vertices_rounded[t]/np.linalg.norm(ref_vertices_rounded[t])), tol)))
                            if angle <= tolerance:
                                count = count + 1
                                break

                if count == len(ref_vertices) and q_rounded not in final_q_arr:
                    # print(q_rounded, np.round(unit_vec, 4), round(np.rad2deg(rot_arr[j]), 2))
                    if round(np.rad2deg(rot_arr[j]),0) not in list(dic_data.keys()):
                        dic_data[int(round(np.rad2deg(rot_arr[j]),0))] = []
                        temp_data[int(round(np.rad2deg(rot_arr[j]),0))] = []

                    unit_vec_rnd = [round(unit_vec[0], tol), round(unit_vec[1], tol), round(unit_vec[2], tol)]
                    minus_unit_vec_rnd = [-round(unit_vec[0], tol), -round(unit_vec[1], tol), -round(unit_vec[2], tol)]

                    if round(np.rad2deg(rot_arr[j]),0) == 180:
                        if unit_vec_rnd not in temp_data[round(np.rad2deg(rot_arr[j]),0)] and minus_unit_vec_rnd not in temp_data[round(np.rad2deg(rot_arr[j]),0)]:
                            dic_data[round(np.rad2deg(rot_arr[j]),0)].append(vecvec)
                            temp_data[round(np.rad2deg(rot_arr[j]),0)].append(unit_vec_rnd)
                        else:
                            pass

                    else:
                        dic_data[round(np.rad2deg(rot_arr[j]),0)].append(vecvec)
                        temp_data[round(np.rad2deg(rot_arr[j]),0)].append(unit_vec_rnd)

                    Arr.append(math.gcd(360, int(round(np.rad2deg(rot_arr[j]), tol))))
                    final_q_arr.append(q_rounded)
                    mod_positions_sym.append(vecvec)

        sorted_mod_positions_sym = mod_positions_sym[:]
        sorted_quat_arr = final_q_arr[:]

        invariant_quantity = []
        for i in final_q_arr:
            invariant_quantity.append(i)

        invariant_quantity.append([1, 0, 0, 0])
        # invariant_quantity.append([-1, 0, 0, 0])

        # Collect the data as attributes
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.dic_data = dic_data
        self.invariant_quantity = invariant_quantity
        
