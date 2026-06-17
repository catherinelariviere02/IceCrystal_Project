from collections import Counter, defaultdict, deque, namedtuple
from scipy.spatial import ConvexHull
import numpy as np

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
