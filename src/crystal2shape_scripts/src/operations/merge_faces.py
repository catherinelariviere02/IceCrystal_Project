import numpy as np
import freud
import coxeter
from scipy.spatial import ConvexHull
from itertools import chain

from ..visualization.plot_poly import plot_poly_func
import matplotlib.pyplot as plt

class merge_faces_class():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def calc_normal_func(self, points):
        a = np.array(points[1]) - np.array(points[0])
        b = np.array(points[2]) - np.array(points[0])
        n_cap = np.cross(a, b)
        n_unit = (n_cap/np.linalg.norm(n_cap))

        return n_unit

    def merge_faces_func(self):
        """
        Merges faces of the polyhedron 
        
        """
        # ConvexHull
        hull = ConvexHull(self.vertices)
        vertices = [self.vertices[vertex] for vertex in hull.vertices]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        env = [[round(float(m[0]), 2), round(float(m[1]), 2), round(float(m[2]), 2)] for m in vertices]
        env_arr = [env]
        for env in env_arr:
            hull = ConvexHull(env)
            env = [env[u] for u in hull.vertices]
            poly = coxeter.shapes.ConvexPolyhedron(env)
            fig, ax = plot_poly_func(self, poly, fig, ax, env)
        plt.show()
        plt.close()

        # Make Coxeter object
        poly = coxeter.shapes.ConvexPolyhedron(vertices)
        faces = poly.faces
        print(len(faces), "faces before merging")
        normals = poly.normals

        # Assuring the normals to point outside from the center of the polyhedron
        normals_mod = []
        for n in range(len(normals)):
            vec = - np.array(vertices[faces[n][0]])
            test_val = np.dot(normals[n], vec)
            if test_val <= 0:
                normals_mod.append(normals[n])
            else:
                normals_mod.append(-normals[n])
        normals = normals_mod[:]

        # Get unique normals 
        unique_normals = []
        for n in normals:
            if not any(np.allclose(n, un, atol=self.tol) for un in unique_normals):
                unique_normals.append(n)

        # print(len(unique_normals), len(normals))
        # Categorize faces based on normals
        face_category = []
        for un in unique_normals:
            faces_with_normal = [f.tolist() for f, n in zip(faces, normals) if np.allclose(n, un, atol=self.tol)]
            face_category.append(faces_with_normal)

        print(face_category)
        # Merge coplannar faces
        merged_faces = []
        for faces in face_category:
            if len(faces) > 1:
                # Find the shared edges
                common_elements = []
                for f in range(len(faces)-1):
                    for g in range(f + 1, len(faces)):
                        shared_edges = list(set(faces[f]).intersection(faces[g]))
                        if len(shared_edges) > 0:
                            for c in shared_edges:
                                if c not in common_elements:
                                    common_elements.append(c)

                # Eliminate common edges
                remaining_vertices = [v for v in chain.from_iterable(faces) if v not in common_elements]
                merged_faces.append(remaining_vertices)
            else:
                merged_faces.append(faces[0])

        print(merged_faces)
        self.merged_faces = merged_faces[:]
        self.vertices_after_merging = [vertices[i] for i in list(set(chain.from_iterable(merged_faces)))]
