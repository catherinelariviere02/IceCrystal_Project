from scipy.spatial import ConvexHull
import coxeter
import numpy as np
from itertools import chain
import pyvista as pv
import os
import json

class shape_class():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def shape_func(self):
        pl = pv.Plotter()

        verts = self.poly_vertices[:]
        hull = ConvexHull(verts)
        verts = [verts[t] for t in hull.vertices]
        poly = coxeter.shapes.ConvexPolyhedron(verts)
        v_arr = verts[:]
        faces = poly.faces
        # points = list(chain.from_iterable(v_arr))
        points = [t.tolist() for t in v_arr]
        polyhedron_connectivity = []
        num_particles = 1
        for i in range(num_particles):
            sub_arr = []
            sub_arr.append(len(faces))
            for j in range(len(faces)):
                sub_arr.append(len(faces[j]))
                for k in range(len(faces[j])):
                    sub_arr.append(faces[j][k]+i*len(v_arr[i]))

            sub_arr = [len(sub_arr)] + sub_arr
            for s in sub_arr:
                polyhedron_connectivity.append(s)
        
        # Creating unstructured grid from pyvista
        cells = polyhedron_connectivity[:]
        celltypes = [pv.CellType.POLYHEDRON for _ in range(num_particles)]
        ungrid = pv.UnstructuredGrid(cells, celltypes, points)

        pl.add_mesh(ungrid, show_edges=True, line_width=1, color=pv.Color('darkslateblue', opacity=0.5), lighting=True, specular=1.0, specular_power=1.0, ambient=0.5, opacity=0.5)

        centroid, largest_face_centroids, largest_face_dists = self.face_centroid_dist()
        line1 = pv.Line(centroid, largest_face_centroids[0])
        line1_len = np.linalg.norm(centroid-largest_face_centroids[0])
        line2 = pv.Line(centroid, largest_face_centroids[1])
        line2_len = np.linalg.norm(centroid-largest_face_centroids[1])
        line3 = pv.Line(centroid, largest_face_centroids[2])
        line3_len = np.linalg.norm(centroid-largest_face_centroids[2])
        line4 = pv.Line(centroid, largest_face_centroids[3])
        line4_len = np.linalg.norm(centroid-largest_face_centroids[3])

        pl.add_mesh(line1, color='blue', line_width=5, label = f"{line1_len}")
        pl.add_mesh(line2, color='red', line_width=5, label = f"{line2_len}")
        pl.add_mesh(line3, color='green', line_width=5, label = f"{line3_len}")
        pl.add_mesh(line4, color='purple', line_width=5, label = f"{line4_len}")

        pl.add_legend()
        pl.show()
        #pl.export_html(self.filename)

    def face_centroid_dist(self):
        pl = pv.Plotter()
        
        verts = self.poly_vertices[:]
        hull = ConvexHull(verts)
        verts = [verts[t] for t in hull.vertices]
        poly = coxeter.shapes.ConvexPolyhedron(verts)
        v_arr = verts[:]
        faces = poly.faces

        centroid = poly.centroid 
        print(f"centroid is {centroid}")

        #4 largest faces 
        face_area = np.sort(poly.get_face_area(), descending=True)
        face_inds = np.argsort(poly.get_face_area(), descending=True)
        largest_faces = face_area[0:4]
        largest_face_inds = face_inds[0:4]

        print(f"face area is {face_area}, largest {len(largest_faces)} are {largest_faces}")

        centroids = poly.face_centroids
        largest_face_centroids = centroids[largest_face_inds]

        print(largest_face_centroids)

        largest_face_dists = np.abs(largest_face_centroids - centroid)

        return centroid, largest_face_centroids, largest_face_dists 


dir = "../../inputs/92_H2O_3/"
for i, shape_file in enumerate(os.listdir(dir)):
    if shape_file.endswith(".json"):
        with open(dir + shape_file) as file:
            print(file, " ", i)
            shape = json.load(file)
            verts = shape_class(poly_vertices = np.array(shape["8_vertices"]), filename = f"{dir}shape{i}_withlines.html")
            verts.shape_func()
            verts.face_centroid_dist()