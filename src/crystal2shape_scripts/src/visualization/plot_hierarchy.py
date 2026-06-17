from scipy.spatial import ConvexHull
import coxeter
import numpy as np
from itertools import chain
import pyvista as pv

class plot_hierarchy_class():
    def __init__(self):
        pass

    def plot_box(self, pl, uc_box):
        """
        Plot the unit cell box using PyVista.
        This function constructs the vertices of the unit cell box based on the dimensions provided in `uc_box_list`.
        It creates a polydata object representing the box and extracts its feature edges to visualize the box
        in the PyVista plot. The box is displayed with a specified color and line width.
        """

        # Box
        uc_box_list = np.array([uc_box.get_box_vector(i) for i in range(3)])
        b1, b2, b3 = uc_box_list[:,0], uc_box_list[:,1], uc_box_list[:,2]
        # print(b1, b2, b3)

        # Box vertices
        boxvert1 = (-b1/2.0) + (-b2/2.0) + (-b3/2.0)
        boxvert2 = (-b1/2.0) + (b2/2.0) + (-b3/2.0)
        boxvert3 = (b1/2.0) + (-b2/2.0) + (-b3/2.0)
        boxvert4 = (b1/2.0) + (b2/2.0) + (-b3/2.0)
        boxvert5 = (-b1/2.0) + (-b2/2.0) + (b3/2.0)
        boxvert6 = (-b1/2.0) + (b2/2.0) + (b3/2.0)
        boxvert7 = (b1/2.0) + (-b2/2.0) + (b3/2.0)
        boxvert8 = (b1/2.0) + (b2/2.0) + (b3/2.0)

        box_points = np.array([boxvert1, boxvert3, boxvert2, boxvert5, boxvert7, boxvert6, boxvert4, boxvert8])
        # print(box_points)
        box_faces = [[0, 1, 4, 3], [1, 6, 7, 4], [6, 2, 5, 7], [2, 0, 3, 5], [0, 1, 6, 2], [3, 4, 7, 5]]
        modified_box_faces = []
        for b in box_faces:
            modified_box_faces.append(len(b))
            for a in b:
                modified_box_faces.append(a)

        box_mesh = pv.PolyData(box_points, modified_box_faces)
        visibleEdges = box_mesh.extract_feature_edges()
        sim_added_box = pl.add_mesh(visibleEdges, color='k', line_width=5)

    def plot_hierarchy_func(self, 
                            posi, 
                            uc_box_list,  
                            line_points=None, 
                            **kwargs):

        color_types = ["navy", "purple", "green", "yellow", "pink", "cyan", "lime", "brown", "teal", "maroon", "magenta", "aquamarine", "coral"]
        for key, value in kwargs.items():
            setattr(self, key, value)

        # PyVista plot
        pl = pv.Plotter()
        sp_mesh = pl.add_points(np.array(posi), render_points_as_spheres=True, point_size=10, color=pv.Color('navy', opacity=0.5), lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)

        # Extra positions if exists
        # if self.extra_positions is not None:
        #     pl.add_points(np.array(self.extra_positions), render_points_as_spheres=True, point_size=10, color=pv.Color('red', opacity=0.5), lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)

        unique_types = []
        counter = 0
        if self.types is not None:
            unique_types = list(set(self.types))
            for u in range(len(unique_types)):
                indices = [i for i, x in enumerate(self.types) if x == unique_types[u]]
                positions_type = np.array(self.positions)[indices]
                if self.wyckoff_uc is not None:
                    wyckoffs_type = np.array(self.wyckoff_uc)[indices]
                    unique_wyckoffs = list(set(wyckoffs_type))
                    for w in range(len(unique_wyckoffs)):
                        positions_wyckoff = [positions_type[i] for i in range(len(positions_type)) if wyckoffs_type[i] == unique_wyckoffs[w]]
                        sp_mesh = pl.add_points(np.array(positions_wyckoff), render_points_as_spheres=True, point_size=15, color=color_types[counter], lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
                        counter += 1

                else:
                    sp_mesh = pl.add_points(positions_type, render_points_as_spheres=True, point_size=15, color=color_types[counter], lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
                    counter += 1

        # Extra positions if exists
        counter = 0
        if self.extra_positions is not None:
            for u in range(len(unique_types)):
                positions_type = [self.extra_positions[i] for i in range(len(self.extra_positions)) if self.extra_particles_types[i] == unique_types[u]]
                if self.extra_particles_wyckoffs is not None:
                    wyckoffs_type = [self.extra_particles_wyckoffs[i] for i in range(len(self.extra_particles_wyckoffs)) if self.extra_particles_types[i] == unique_types[u]]
                    unique_wyckoffs = list(set(wyckoffs_type))
                    for w in range(len(unique_wyckoffs)):
                        positions_wyckoff = [positions_type[i] for i in range(len(positions_type)) if wyckoffs_type[i] == unique_wyckoffs[w]]
                        sp_mesh = pl.add_points(np.array(positions_wyckoff), render_points_as_spheres=True, point_size=15, color=color_types[counter], lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
                        counter += 1
                else:
                    pl.add_points(np.array(positions_type), render_points_as_spheres=True, point_size=15, color=pv.Color(color_types[counter], opacity=0.2), lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
                    # pl.add_points(np.array(positions_type), render_points_as_spheres=True, point_size=15, color="teal", lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
                    counter += 1

        if line_points is not None:
            for line in line_points:
                points = np.array(line)
                pl.add_lines(points, color='blue', width=2)

        # Plot Box
        self.plot_box(pl, uc_box_list)

        pl.camera.elevation = -30.0 # Vertical angle
        pl.camera.azimuth = -135.0   # Horizontal angle

        if self.filename is not None:
            from pathlib import Path
            root, extension = Path(self.filename).stem, Path(self.filename).suffix
            if extension == '.png':
                pl.show()
                pl.screenshot(self.filename)
            elif extension == '.html':
                pl.show()
                pl.export_html(self.filename)

        pl.close()