"""Polyhedron vertices cleanup module.

This module provides tools for simplifying polyhedron shapes by removing
non-essential or duplicate vertices while preserving moments of inertia.
"""

import numpy as np
import rowan, coxeter
from scipy.spatial import ConvexHull
import meshlib.mrmeshpy as mrmeshpy
import meshlib.mrmeshnumpy as mrmeshnumpy

class CleanupPolyVertices:
    """Simplifies polyhedral shapes by pruning redundant vertices.

    Attributes:
        vertices_arr (List[np.ndarray]): Array of polyhedral vertices to process.
        cleaned_vertices (List[np.ndarray]): Pruned vertices of same shape.
        orientation_arr (List[np.ndarray]): Rowan quaternions mapping local frame to global.
        rotational_matrices (List[np.ndarray]): Matrix transformations from diagonalization.
    """

    def __init__(self, vertices_arr):
        self.vertices_arr = vertices_arr

    def SaveMesh(self, vertices, filename="output_mesh.stl"):
        """Saves the convex hull of the vertices as an STL mesh.

        Args:
            vertices (np.ndarray): Polyhedron vertices.
            filename (str): Target STL output filename. Defaults to "output_mesh.stl".
        """
        hull = ConvexHull(vertices)
        faces = np.ndarray(
                        shape=(len(hull.simplices), 3),
                        dtype=np.int32,
                        buffer=np.array(
                            hull.simplices,
                            dtype=np.int32,
                        ),
                    )
        
        verts = np.ndarray(
                            shape=(len([vertices[i] for i in hull.vertices]), 3),
                            dtype=np.float32,
                            buffer=np.array(
                                [vertices[i] for i in hull.vertices],
                                dtype=np.float32,
                            ),
                        )

        mesh = mrmeshnumpy.meshFromFacesVerts(faces, verts)
        mrmeshpy.saveMesh(mesh, filename)

    def RemoveExtraVertices(self, vertices, min_num_vertices):
        """Removes extra non-essential vertices from a polyhedron.

        Prunes vertices systematically while verifying that the moments of
        inertia (diagonalized inertia tensor) match the original polyhedron.

        Args:
            vertices (np.ndarray): The starting polyhedral vertices.
            min_num_vertices (int): The target minimum number of vertices.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The cleaned global-frame vertices.
                - Rowan quaternion mapping local to global coordinates.
                - The rotational matrix from diagonalization.
        """
        hull = ConvexHull(vertices)
        vertices = [vertices[v] for v in hull.vertices]
        poly = coxeter.shapes.ConvexPolyhedron(vertices)
        inertia_tensor = poly.inertia_tensor
        eigvals, eigvecs = np.linalg.eig(inertia_tensor)
        rotational_matrix = np.linalg.inv(eigvecs)
        # print(rotational_matrix)

        poly.diagonalize_inertia()
        moi_tensor = poly.inertia_tensor.round(3)
        moi_vals = [float(moi_tensor[0][0]), float(moi_tensor[1][1]), float(moi_tensor[2][2])]

        if len(vertices) > min_num_vertices:  # Need to remove some vertices
            removed_vertices_tracked = []
            while len(vertices) > min_num_vertices:
                for i in range(len(vertices)):
                    other_vertices_expect_i = [vertices[j] for j in range(len(vertices)) if j != i]
                    temp_poly = coxeter.shapes.ConvexPolyhedron(other_vertices_expect_i)
                    temp_poly.diagonalize_inertia()
                    temp_moi_tensor = temp_poly.inertia_tensor.round(3)
                    temp_moi_vals = [temp_moi_tensor[0][0], temp_moi_tensor[1][1], temp_moi_tensor[2][2]]
                    if moi_vals == temp_moi_vals:
                        removed_vertices_tracked.append(vertices[i])
                        vertices = other_vertices_expect_i[:]
                        break

        else:
            removed_vertices_tracked = []

        vertices_global_frame = np.array(vertices[:])
        poly = coxeter.shapes.ConvexPolyhedron(vertices_global_frame)
        eigvals, eigvecs = np.linalg.eig(inertia_tensor)
        rotational_matrix = np.linalg.inv(eigvecs)

        poly.diagonalize_inertia()
        vertices_local_frame = poly.vertices
        R, t, indices = rowan.mapping.icp(vertices_local_frame, vertices_global_frame, return_indices=True)
        q = rowan.from_matrix(R)

        return vertices_global_frame, q, rotational_matrix

    def clean(self):
        """Cleans and prunes all polyhedra in `vertices_arr` to a common size.

        Prunes all shapes to match the vertex count of the smallest polyhedron
        in the input array, and updates the class instance results.
        """
        # Reference vertices based on the minimum number of vertices
        min_vertices_len = min([len(v) for v in self.vertices_arr])
        indices = [g for g, e in enumerate(self.vertices_arr) if len(e) == min_vertices_len][0]

        cleanedup_vertices_arr = []
        orientation_arr = []
        rotational_matrices = []
        for vertices in self.vertices_arr:
            cleaned_vertices, orientation, rotational_matrix = self.RemoveExtraVertices(vertices, min_vertices_len)
            cleanedup_vertices_arr.append(cleaned_vertices)
            orientation_arr.append(orientation)
            rotational_matrices.append(rotational_matrix)
            # self.SaveMesh(cleaned_vertices, filename=f"output_mesh_{i}.stl")

        self.cleaned_vertices = cleanedup_vertices_arr[:]
        self.orientation_arr = orientation_arr[:]
        self.rotational_matrices = rotational_matrices[:]