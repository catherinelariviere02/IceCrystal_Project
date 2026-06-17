import numpy as np
import coxeter

from collections import Counter, defaultdict, deque, namedtuple
from scipy.spatial import ConvexHull
import rowan
import math
from sklearn.decomposition import PCA

# This function is copied from Euclid
def convexHull(vertices, tol):
    """Compute the 3D convex hull of a set of vertices and merge coplanar faces.

    Args:
        vertices (list): List of (x, y, z) coordinates
        tol (float): Floating point tolerance for merging coplanar faces


    Returns an array of vertices and a list of faces (vertex
    indices) for the convex hull of the given set of vertice.

    .. note::
        This method uses scipy's quickhull wrapper and therefore requires scipy.

    """
    from scipy.spatial import cKDTree, ConvexHull;
    from scipy.sparse.csgraph import connected_components;

    hull = ConvexHull(vertices);
    # Triangles in the same face will be defined by the same linear equalities
    dist = cKDTree(hull.equations);
    trianglePairs = dist.query_pairs(tol);

    connectivity = np.zeros((len(hull.simplices), len(hull.simplices)), dtype=np.int32);

    for (i, j) in trianglePairs:
        connectivity[i, j] = connectivity[j, i] = 1;

    # connected_components returns (number of faces, cluster index for each input)
    (_, joinTarget) = connected_components(connectivity, directed=False);
    faces = defaultdict(list);
    norms = defaultdict(list);
    for (idx, target) in enumerate(joinTarget):
        faces[target].append(idx);
        norms[target] = hull.equations[idx][:3];

    # a list of sets of all vertex indices in each face
    faceVerts = [set(hull.simplices[faces[faceIndex]].flat) for faceIndex in sorted(faces)];
    # normal vector for each face
    faceNorms = [norms[faceIndex] for faceIndex in sorted(faces)];

    # polygonal faces
    polyFaces = [];
    for (norm, faceIndices) in zip(faceNorms, faceVerts):
        face = np.array(list(faceIndices), dtype=np.uint32);
        N = len(faceIndices);

        r = hull.points[face];
        rcom = np.mean(r, axis=0);

        # plane_{a, b}: basis vectors in the plane
        plane_a = r[0] - rcom;
        plane_a /= np.sqrt(np.sum(plane_a**2));
        plane_b = np.cross(norm, plane_a);

        dr = r - rcom[np.newaxis, :];

        thetas = np.arctan2(dr.dot(plane_b), dr.dot(plane_a));

        sortidx = np.argsort(thetas);

        face = face[sortidx];
        polyFaces.append(face);

    return (hull.points, polyFaces);

ConvexDecomposition = namedtuple('ConvexDecomposition', ['vertices', 'edges', 'faces'])

# This function is copied from Euclid
def convexDecomposition(vertices, tol):
    """Decompose a convex polyhedron specified by a list of vertices into
    vertices, faces, and edges. Returns a ConvexDecomposition object.
    """
    (vertices, faces) = convexHull(vertices, tol)
    edges = set()

    for face in faces:
        for (i, j) in zip(face, np.roll(face, -1)):
            edges.add((min(i, j), max(i, j)))

    return ConvexDecomposition(vertices, edges, faces)

def face_edge_from_vertex_func(vertices, num_edges, num_faces):
    """
    Function to decompose a convex polyhedron into its edges and faces.
    
    Args:
    --------
    vertices : list
        List of vertices of the polyhedron.
    num_edges : int     
        Number of edges in the polyhedron.
    num_faces : int
        Number of faces in the polyhedron.

    Returns:
    --------
    edges : list
        List of edges of the polyhedron.
    faces : list
        List of faces of the polyhedron.
    vertices : list
        List of vertices of the polyhedron.
    """
    n_arr = [i for i in range(12, 3, -1)]
    for z in n_arr:
        tol = 1/(10**(z))
        convex_decomp = convexDecomposition(vertices, tol)
        edges = convex_decomp[1]
        faces = convex_decomp[2]
        # print(len(edges), len(faces), len(vertices))
        if len(edges) == num_edges and len(faces) == num_faces:
            break

    return edges, faces, vertices

class PointGroup:
    def __init__(self):
        pass

    def analyze(self, coords, rtol=0.1, atol=1):
        poly = coxeter.shapes.ConvexPolyhedron(coords)
        vertices = coords.tolist()
        faces = poly.faces
        edges = poly.edges
        edges = [edge.tolist() for edge in edges]

        tol = 4
        center = np.mean(np.array([vertices[i] for i in range(len(vertices))]), axis=0).tolist()
        # center = np.array([0, 0, 0])
        vecs = []
        # Vertices
        for i in range(len(vertices)):
            vv = np.array(vertices[i]) - np.array(center)
            vecs.append(vv.tolist())

        for i in range(len(faces)):
            vertice_arr = [vertices[faces[i][j]] for j in range(len(faces[i]))]
            face_midpt = np.mean(np.array(vertice_arr), axis=0)
            vv = face_midpt - np.array(center)
            vecs.append(vv.tolist())

        # Center to face normal
        '''center_to_face_normal = []
        for i in range(len(faces)):
            polygon_vertices = [np.array(vertices[faces[i][j]]) for j in range(len(faces[i]))]
            n_cap = calc_normal_func(polygon_vertices)
            p_vec = vertices[faces[i][0]]
            d = -p_vec[0]*n_cap[0] - p_vec[1]*n_cap[1] - p_vec[2]*n_cap[2]
            dist_to_plane = np.dot(np.array(n_cap), np.array(center)) + d
            pt = np.array(center) - dist_to_plane*np.array(n_cap)
            center_to_face_normal.append(pt.tolist())

        for i in range(len(center_to_face_normal)):
            vv = np.array(center_to_face_normal[i])
            vecs.append(vv.tolist())'''

        edges_arr = []
        for e in range(len(list(edges))):
            edges_arr.append(list(list(edges)[e]))

        # center to edges midpoint
        edge_midpt_arr = [np.mean(np.array([vertices[edges_arr[i][j]] for j in range(len(edges_arr[i]))]), axis=0) for i in range(len(edges_arr))]
        edges_len = [np.linalg.norm(np.array(vertices[edges_arr[i][0]]) - np.array(vertices[edges_arr[i][1]])) for i in range(len(edges_arr))]
        # print(edges_len)

        for i in range(len(edge_midpt_arr)):
            vv = np.array(edge_midpt_arr[i]) - np.array(center)
            vecs.append(vv.tolist())

        #****************************************************************************

        '''rot_arr = [np.deg2rad(180), np.deg2rad(120), np.deg2rad(240), np.deg2rad(90), np.deg2rad(270),
            np.deg2rad(72), np.deg2rad(144), np.deg2rad(216), np.deg2rad(288), np.deg2rad(60),
            np.deg2rad(300),np.deg2rad(45), np.deg2rad(135), np.deg2rad(225), np.deg2rad(315), np.deg2rad(252), np.deg2rad(324),
            np.deg2rad(36), np.deg2rad(108), np.deg2rad(-120), np.deg2rad(-240), np.deg2rad(-90), np.deg2rad(-270),
            np.deg2rad(-72), np.deg2rad(-144), np.deg2rad(-216), np.deg2rad(-288), np.deg2rad(-60),
            np.deg2rad(-300),np.deg2rad(-45), np.deg2rad(-135), np.deg2rad(-225), np.deg2rad(-315),
            np.deg2rad(-36), np.deg2rad(-108), np.deg2rad(-252), np.deg2rad(-324)]'''
        
        rot_arr = [np.deg2rad(180), np.deg2rad(120), np.deg2rad(90), np.deg2rad(60)]

        ref_vertices = vertices[:]
        ref_vertices_rounded = []
        for i in range(len(vertices)):
            ref_vertices_rounded.append(np.array(ref_vertices[i]).tolist())

        final_q_arr, mod_positions_sym = [], []
        Arr = []
        dic_data = {}
        for t in rot_arr:
            dic_data[round(np.rad2deg(t),0)] = []

        for i in range(len(vecs)):
            vecvec = vecs[i]
            unit_vec = vecs[i]/np.linalg.norm(vecs[i])
            for j in range(len(rot_arr)):
                q = rowan.normalize(rowan.from_axis_angle(unit_vec, rot_arr[j]))
                q_rounded = [round(m, tol) for m in q]  # rounding off to 4 decimal places
                count = 0
                for k in range(len(ref_vertices)):
                    vec = rowan.rotate(q, ref_vertices[k])
                    vec_rounded = vec.tolist()
                    for t in range(len(ref_vertices_rounded)):
                        a = np.isclose(vec_rounded, ref_vertices_rounded[t], atol=rtol)
                        a_true = [t for t in a if t==True]
                        if np.linalg.norm(vec_rounded) != 0 and np.linalg.norm(ref_vertices_rounded[t]) != 0:
                            angle = np.rad2deg(np.arccos(round(np.dot(vec_rounded/np.linalg.norm(vec_rounded), ref_vertices_rounded[t]/np.linalg.norm(ref_vertices_rounded[t])), tol)))
                            if angle <= atol:
                                count = count + 1
                                break

                if count == len(ref_vertices) and q_rounded not in final_q_arr:
                    # print(q_rounded, np.round(unit_vec, 4), round(np.rad2deg(rot_arr[j]), 2))
                    dic_data[round(np.rad2deg(rot_arr[j]),0)].append(np.array(vecvec).tolist()) 
                    Arr.append(math.gcd(360, int(round(np.rad2deg(rot_arr[j]), tol))))
                    final_q_arr.append(q_rounded)
                    mod_positions_sym.append(np.array(vecvec).tolist())

        # If no symmetry found, define principal axis using PCA
        if np.sum([len(v) for v in list(dic_data.values())]) == 0:
            X = np.array(coords)
            pca = PCA(n_components=1)
            pca.fit(X)
            principal_axis = pca.components_[0]
            dic_data[360.0] = [principal_axis.tolist(), (-1*principal_axis).tolist()]

        invariant_quantity = []
        for i in final_q_arr:
            invariant_quantity.append(i)

        invariant_quantity.append([1, 0, 0, 0])

        self.symm_dic = dic_data
        self.invariant_quats = invariant_quantity