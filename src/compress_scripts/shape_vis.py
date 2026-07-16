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
        pl.export_html(self.filename)

dir = "../../inputs/141_H2O_0/"
for i, shape_file in enumerate(os.listdir("../../inputs/141_H2O_0/")):
    if shape_file.endswith(".json"):
        with open(dir + shape_file) as file:
            print(file, " ", i)
            shape = json.load(file)
            verts = shape_class(poly_vertices = np.array(shape["8_vertices"]), filename = f"shape{i}.html")
            verts.shape_func()