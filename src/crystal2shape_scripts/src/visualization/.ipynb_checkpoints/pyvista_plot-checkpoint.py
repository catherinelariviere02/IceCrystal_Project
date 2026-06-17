from scipy.spatial import ConvexHull
import coxeter
import numpy as np
from itertools import chain
import pyvista as pv
import pyista
pyvista.set_jupyter_backend('trame')

class pyvista_plot_class():
    """
    Plotting class for PyVista.
    This class is used to visualize the system using PyVista.
    It creates a 3D plot of the particles and the unit cell box.
    The class takes the following parameters:

    Parameters
    ----------
    uc_box_list : list
        List of unit cell box dimensions.
    positions : list
        List of particle positions.
    lattice_sites : list
        List of lattice sites.
    poly_vertices : list
        List of polyhedron vertices.    

    
    Attributes
    ----------  
    pl : Plotter
        PyVista plotter object.
    
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def plot_poly(self, poly_vertices, color):
        """
        Plot a polyhedron using PyVista.

        Parameters
        ----------
        poly_vertices : list
            List of vertices of the polyhedron to be plotted.

        color : str
            Color of the polyhedron.

        Returns
        -------
        None

        Notes
        -----
        This function uses the ConvexHull from scipy.spatial to create a convex hull from the
        provided vertices. It then constructs a ConvexPolyhedron object from the vertices of the hull
        and creates an unstructured grid in PyVista. The mesh is then added to the PyVista plotter with specified
        visual properties such as color, opacity, and lighting.
        The function assumes that the input vertices are in a format compatible with ConvexHull and PyVista.
        The vertices should be a list of tuples or numpy arrays representing the coordinates of the polyhedron vertices.
        The function does not return any value, but it modifies the PyVista plotter object by adding the polyhedron mesh to it.
        The polyhedron is visualized with edges shown, a specified line width, and a color of 'darkslateblue' with some opacity.
        The lighting and specular properties are also set to enhance the visual appearance of the polyhedron in the plot.

        """
        verts = poly_vertices[:]
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

        # Plot polyhedron vertices in different colors
        # self.pl.add_points(poly_vertices, render_points_as_spheres=True, point_size=30, color='red', lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)

        added_mesh = self.pl.add_mesh(ungrid, show_edges=True, line_width=1, color=pv.Color("darkslateblue", opacity=0.5), lighting=True, specular=1.0, specular_power=1.0, ambient=0.5, opacity=0.5)
        # added_mesh = self.pl.add_mesh(ungrid, show_edges=True, line_width=1.5, color=pv.Color(color, opacity=0.5), lighting=True, specular=1.0, specular_power=1.0, ambient=0.5, opacity=0.5)
        # Save the mesh
        # self.pl.export_obj('polyhedron.obj')

    def plot_box(self):
        """
        Plot the unit cell box using PyVista.
        This function constructs the vertices of the unit cell box based on the dimensions provided in `uc_box_list`.
        It creates a polydata object representing the box and extracts its feature edges to visualize the box
        in the PyVista plot. The box is displayed with a specified color and line width.
        """
        # Box
        uc_box_list = np.transpose(self.uc_box.to_matrix())
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
        sim_added_box = self.pl.add_mesh(visibleEdges, color='k', line_width=5)

    def pyvista_plot_func(self):
        # PyVista plot
        self.pl = pv.Plotter()
        color_types = ["olive", "purple", "green", "yellow", "pink", "cyan", "navy", "lime", "brown", "teal", "maroon", "magenta", "aquamarine", "coral"]
        unique_types = []
        if self.types is not None:
            unique_types = list(set(self.types))
            for u in range(len(unique_types)):
                indices = [i for i, x in enumerate(self.types) if x == unique_types[u]]
                positions_type = np.array(self.positions)[indices]
                sp_mesh = self.pl.add_points(positions_type, render_points_as_spheres=True, point_size=20, color="teal", lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
        # else:
        #     self.pl.add_points(np.array(self.positions), render_points_as_spheres=True, point_size=20, color=pv.Color('teal'), lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)

        # Extra positions if exists
        if self.extra_positions is not None:
            for u in range(len(unique_types)):
                positions_type = [self.extra_positions[i] for i in range(len(self.extra_positions)) if self.extra_particles_types[i] == unique_types[u]]
                # self.pl.add_points(np.array(positions_type), render_points_as_spheres=True, point_size=20, color=color_types[u], lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)
                self.pl.add_points(np.array(positions_type), render_points_as_spheres=True, point_size=20, color="teal", lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)

        '''if self.extra_positions is not None:
            for p in range(len(self.extra_positions)):
                self.pl.add_points(np.array(self.extra_positions[p]), render_points_as_spheres=True, point_size=10, color=pv.Color(self.color_arr[p], opacity=0.5), lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)'''

        if self.line_points is not None:
            for line in self.line_points:
                points = np.array(line)
                self.pl.add_lines(points, color='blue', width=2)

        # Plot polyhedron
        if len(self.poly_vertices) != 0:
            if self.types is not None:
                unique_types = list(set(self.types))
                counter = 0
                for poly in self.poly_vertices:
                    # self.plot_poly(poly, color_types[unique_types.index(self.types[counter])])
                    self.plot_poly(poly, "darkslateblue")
                    counter += 1

            else:
                for poly in self.poly_vertices:
                    self.plot_poly(poly, 'teal')
        # Plot Box
        if self.uc_box is not None:
            self.plot_box()

        light = pv.Light(color='white', light_type='headlight')
        self.pl.add_light(light)
        self.pl.enable_ssao()

        self.pl.show()
        self.pl.close()
