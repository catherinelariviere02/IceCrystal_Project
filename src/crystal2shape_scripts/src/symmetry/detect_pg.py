"""Point group detection module.

This module provides classes to detect the Schoenflies point group symmetry
for a regular coordinate point set / body based on vertices, edges, and faces.
"""

from . import face_edge_from_vertex
from . import calc_normal
import numpy as np
import rowan
import math
import coxeter

class detect_pg_class:
    """Class to detect point group symmetry of a coordinate shape."""

    def __init__(self):
        pass

    def detect_pg_func(self, vertices, num_edges, num_faces):
        """Detect the point group of a regular body based on its vertices, edges, and faces.

        Args:
            vertices (List[List[float]] or np.ndarray): List of vertices of the regular body.
            num_edges (int): Number of edges in the regular body.
            num_faces (int): Number of faces in the regular body.

        Returns:
            Tuple[Dict[int, List], List[int], List[float]]: A tuple containing:
                - Dictionary containing symmetry operations grouped by rotation angles.
                - List of occurrences of each symmetry operation.
                - List of rotation angles corresponding to the symmetry operations.
        """
        
        edges, faces, vertices = face_edge_from_vertex.face_edge_from_vertex_func(vertices, num_edges, num_faces)

        tol = 5
        atol = 1e-5
        tolerance = 0.01
        center = np.mean(np.array([vertices[i] for i in range(len(vertices))]), axis=0).tolist()
        # center = np.array([0, 0, 0])
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
            vecvec_rounded = [round(float(t), tol) for t in vecvec]
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
                        '''if len(a_true) == len(vec_rounded):
                            count = count + 1
                            break'''
                        if np.linalg.norm(vec_rounded) != 0 and np.linalg.norm(ref_vertices_rounded[t]) != 0:
                            angle = np.rad2deg(np.arccos(round(np.dot(vec_rounded/np.linalg.norm(vec_rounded), ref_vertices_rounded[t]/np.linalg.norm(ref_vertices_rounded[t])), tol)))
                            if angle <= tolerance:
                                count = count + 1
                                break

                if count == len(ref_vertices) and q_rounded not in final_q_arr:
                    # print(q_rounded, np.round(unit_vec, 4), round(np.rad2deg(rot_arr[j]), 2))
                    if round(np.rad2deg(rot_arr[j]),0) not in list(dic_data.keys()):
                        dic_data[int(round(np.rad2deg(rot_arr[j]),2))] = []
                        temp_data[int(round(np.rad2deg(rot_arr[j]),2))] = []

                    unit_vec_rnd = [round(float(unit_vec[0]), tol), round(float(unit_vec[1]), tol), round(float(unit_vec[2]), tol)]
                    minus_unit_vec_rnd = [-round(float(unit_vec[0]), tol), -round(float(unit_vec[1]), tol), -round(float(unit_vec[2]), tol)]

                    '''dic_data[int(round(np.rad2deg(rot_arr[j]), 2))].append(vecvec)
                    temp_data[int(round(np.rad2deg(rot_arr[j]), 2))].append(unit_vec_rnd)'''

                    if int(round(np.rad2deg(rot_arr[j]), 2)) == 180:
                        if unit_vec_rnd not in temp_data[int(round(np.rad2deg(rot_arr[j]), 2))] and minus_unit_vec_rnd not in temp_data[int(round(np.rad2deg(rot_arr[j]), 2))]:
                            dic_data[int(round(np.rad2deg(rot_arr[j]), 2))].append(vecvec_rounded)
                            temp_data[int(round(np.rad2deg(rot_arr[j]), 2))].append(unit_vec_rnd)
                        else:
                            pass

                    else:
                        dic_data[round(np.rad2deg(rot_arr[j]),0)].append(vecvec_rounded)
                        temp_data[round(np.rad2deg(rot_arr[j]),0)].append(unit_vec_rnd)

                    Arr.append(math.gcd(360, int(round(np.rad2deg(rot_arr[j]), tol))))
                    final_q_arr.append(q_rounded)
                    mod_positions_sym.append(vecvec_rounded)

        invariant_quantity = []
        for i in final_q_arr:
            invariant_quantity.append(i)

        invariant_quantity.append([1, 0, 0, 0])
        invariant_quantity.append([-1, 0, 0, 0])
        
        self.invariant_quaternions = final_q_arr
        self.symmetry_dict = dic_data.copy()
