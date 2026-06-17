from Geometry3D import *

def convexpolyhedron_from_geo3d_func(vertices, edges, faces):
    """
    Create a ConvexPolyhedron from vertices, edges, and faces using Geometry3D library.

    Parameters
    ----------
    vertices : list of tuples
        List of vertex coordinates, where each vertex is represented as a tuple (x, y, z).

    edges : list of tuples
        List of edges, where each edge is represented as a tuple of vertex indices.

    faces : list of tuples
        List of faces, where each face is represented as a tuple of vertex indices.

    Returns
    -------
    ConvexPolyhedron
        A ConvexPolyhedron object constructed from the provided vertices, edges, and faces.

    Notes
    -----
    The function rounds the coordinates of the vertices to 3 decimal places for precision.
    The edges are not used in the construction of the ConvexPolyhedron, as Geometry3D
    constructs the polyhedron based on the vertices and faces alone.
    The function assumes that the input vertices, edges, and faces are valid and correctly defined.
    The rounding of coordinates is done to avoid floating-point precision issues.
    The function uses the Geometry3D library to create the ConvexPolyhedron object.
    """
    r = 3
    # Vertices
    geo_vertices = []
    for v in vertices:
        # geo_vertices.append(header.Point(round(v[0], r), round(v[1], r), round(v[2], r)))
        geo_vertices.append(Point(v[0], v[1], v[2]))

    # faces
    polygon_arr = []
    for f in faces:
        geo_poly_arr = [geo_vertices[f[t]] for t in range(len(f))]
        polygon_arr.append(ConvexPolygon(tuple(geo_poly_arr)))

    cph0 = ConvexPolyhedron(tuple(polygon_arr))

    return cph0

def convexpolygon_from_geo3d_func(vertices):
    """
    Create a ConvexPolygon from a list of 3D vertices.

    Parameters
    ----------
    vertices : list of tuples
        List of vertex coordinates, where each vertex is represented as a tuple (x, y, z).

    Returns
    -------
    ConvexPolygon
        A ConvexPolygon object constructed from the provided vertices.

    """
    r=5
    # Vertices
    geo_vertices = []
    for v in vertices:
        geo_vertices.append(Point(round(v[0], r), round(v[1], r), round(v[2], r)))
        # geo_vertices.append(Point(v[0], v[1], v[2]))

    
    cph0 = ConvexPolygon(tuple([p for p in geo_vertices]))

    return cph0
