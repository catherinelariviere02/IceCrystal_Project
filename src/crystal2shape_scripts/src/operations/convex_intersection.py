import trimesh
import numpy as np
from scipy.spatial import ConvexHull

def intersect_convex_bodies_3d(poly1_vertices, poly2_vertices):
    """
    Finds the intersection of two convex polytopes in 3D using trimesh.

    Args:
        poly1_vertices: (n1, 3) numpy array of vertices for polytope 1.
        poly2_vertices: (n2, 3) numpy array of vertices for polytope 2.

    Returns:
        A trimesh.Trimesh object representing the intersection, or None if no intersection 
        or an error occurs.  Returns an empty trimesh.Trimesh() if there's no intersection.
    """

    if poly1_vertices.shape[1] != 3 or poly2_vertices.shape[1] != 3:
        return []  # Incompatible dimensions

    try:
        poly1_mesh = trimesh.convex.convex_hull(poly1_vertices) # Create meshes
        poly2_mesh = trimesh.convex.convex_hull(poly2_vertices)

        # Use boolean intersection
        intersection_mesh = trimesh.boolean.intersection([poly1_mesh, poly2_mesh], engine='manifold')

        if intersection_mesh.is_empty:
            return []  # Return empty mesh if no intersection

        intersection_verts = intersection_mesh.vertices
        pts = intersection_verts[:]
        updated_pts = [pts[0]]
        for i in range(1, len(pts)):
            # Compare two list for item similarity
            if not np.allclose(pts[i], updated_pts[-1], atol=1e-2, rtol=1e-2):
                updated_pts.append(pts[i])

        pts = updated_pts[:]
        # hull = ConvexHull(pts)
        # intersection_verts = [pts[t] for t in hull.vertices]
        intersection_verts = pts[:]

        return intersection_verts

    except Exception as e:
        print(f"Error during 3D intersection: {e}")
        return []