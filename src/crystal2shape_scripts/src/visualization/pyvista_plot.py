from scipy.spatial import ConvexHull
import coxeter
import numpy as np
from itertools import chain
import pyvista as pv

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
    def __init__(self, uc_box, positions, types, wyckoff_uc,
                 poly_vertices, line_points=None, special_points=None, extra_positions=None, 
                 extra_particles_types=None, extra_particles_wyckoffs=None,
                 center=None, direction=None, text=None, filename=None, color="darkslateblue", 
                 line_colors=None):
        
        self.uc_box = uc_box
        self.positions = positions
        self.types = types
        self.wyckoff_uc = wyckoff_uc
        self.poly_vertices = poly_vertices
        self.line_points = line_points
        self.special_points = special_points
        self.extra_positions = extra_positions
        self.extra_particles_types = extra_particles_types
        self.extra_particles_wyckoffs = extra_particles_wyckoffs
        self.center = center
        self.direction = direction
        self.text = text
        self.filename = filename
        self.color = color
        self.line_colors = line_colors
        
    def plot_poly(self, pl, poly_vertices, color="darkslateblue", alpha=0.5, line_width=1.0):
        """
        Plot a polyhedron using PyVista.

        Parameters
        ----------
        poly_vertices : list
            List of vertices of the polyhedron to be plotted.

        color : str
            Color of the polyhedron.

        alpha : float
            Opacity of the polyhedron.

        line_width : float
            Line width for the edges of the polyhedron.

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
        # pl.add_points(poly_vertices, render_points_as_spheres=True, point_size=30, color='red', lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)

        added_mesh = pl.add_mesh(ungrid, show_edges=True, line_width=line_width, color=pv.Color(color, opacity=alpha), lighting=True, specular=0.0, diffuse=1.0, smooth_shading=False, opacity=0.5)
        # Save the mesh
        # pl.export_obj('polyhedron.obj')

    def plot_box(self, pl):
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
        sim_added_box = pl.add_mesh(visibleEdges, color='k', line_width=5)

    def pyvista_plot_func(self):
        # PyVista plot
        pl = pv.Plotter()
        color_types = ["navy", "purple", "green", "yellow", "pink", "cyan", "lime", "brown", "teal", "maroon", "magenta", "aquamarine", "coral"]
        unique_types = []
        counter = 0
        if self.types is not None:
            unique_types = np.unique(self.types).tolist()
            for u in range(len(unique_types)):
                indices = [i for i, x in enumerate(self.types) if x == unique_types[u]]
                positions_type = np.array(self.positions)[indices]
                if self.wyckoff_uc is not None:
                    wyckoffs_type = np.array(self.wyckoff_uc)[indices]
                    unique_wyckoffs = np.unique(wyckoffs_type).tolist()
                    for w in range(len(unique_wyckoffs)):
                        positions_wyckoff = [positions_type[i] for i in range(len(positions_type)) if wyckoffs_type[i] == unique_wyckoffs[w]]
                        pl.add_points(np.array(positions_wyckoff), render_points_as_spheres=True, point_size=20, color=color_types[counter], lighting=True, roughness=0.0, metallic=1.0)
                        counter += 1

                else:
                    pl.add_points(positions_type, render_points_as_spheres=True, point_size=20, color=color_types[counter], lighting=True, roughness=0.0, metallic=1.0)
                    counter += 1

        # Add special points as highlighted points
        if self.special_points is not None:
            pl.add_points(self.special_points, render_points_as_spheres=True, point_size=30, color='maroon', lighting=True, roughness=0.0, metallic=1.0)

        counter = 0
        # Extra positions if exists
        if self.extra_positions is not None:
            for u in range(len(unique_types)):
                indices = [i for i, x in enumerate(self.extra_particles_types) if x == unique_types[u]]
                positions_type = [self.extra_positions[i] for i in range(len(self.extra_positions)) if self.extra_particles_types[i] == unique_types[u]]
                if self.extra_particles_wyckoffs is not None:
                    wyckoffs_type = np.array(self.extra_particles_wyckoffs)[indices]
                    unique_wyckoffs = np.unique(wyckoffs_type).tolist()
                    for w in range(len(unique_wyckoffs)):
                        positions_wyckoff = [positions_type[i] for i in range(len(positions_type)) if wyckoffs_type[i] == unique_wyckoffs[w]]
                        pl.add_points(np.array(positions_wyckoff), render_points_as_spheres=True, point_size=20, color=color_types[counter], lighting=True, roughness=0.0, metallic=1.0)
                        counter += 1
                else:
                    pl.add_points(np.array(positions_type), render_points_as_spheres=True, point_size=20, color=pv.Color(color_types[counter], opacity=0.2), lighting=True, roughness=0.0, metallic=1.0)
                    counter += 1

        '''if self.extra_positions is not None:
            for p in range(len(self.extra_positions)):
                pl.add_points(np.array(self.extra_positions[p]), render_points_as_spheres=True, point_size=20, color=pv.Color(self.color_arr[p], opacity=0.5), lighting=True, specular=0.7, specular_power=0.7, ambient=0.8)'''

        if self.line_points is not None:
            for i, line in enumerate(self.line_points):
                points = np.array(line)
                pl.add_lines(points, color=self.line_colors[i], width=3)

        # Plot polyhedron
        if self.poly_vertices is not None:
            if self.types is not None:
                unique_types = list(set(self.types))
                counter = 0
                for poly in self.poly_vertices:
                    # plot_poly(poly, color_types[unique_types.index(self.types[counter])])
                    self.plot_poly(pl, poly, self.color, alpha=0.5, line_width=2.0)
                    counter += 1

            else:
                for poly in self.poly_vertices:
                    self.plot_poly(pl, poly, self.color, alpha=0.5, line_width=2.0)

        # Plot Box
        if self.uc_box is not None:
            self.plot_box(pl)

        # Planes
        if self.center is not None and self.direction is not None:
            for i in range(len(self.direction)):
                plane = pv.Plane(center=self.center[i], direction=self.direction[i], i_size=5.0, j_size=5.0)
                pl.add_mesh(plane, color='tomato', opacity=0.3)

        light = pv.Light(color='white', light_type='headlight')
        pl.add_light(light)
        pl.enable_ssao()

        if self.text is not None:
            pl.add_text(self.text, position='upper_edge', font_size=14, color='black', shadow=True)

        pl.camera.elevation = -30.0 # Vertical angle
        pl.camera.azimuth = -135.0   # Horizontal angle

        # Save the plot
        from pathlib import Path
        if self.filename is not None:
            root, extension = Path(self.filename).stem, Path(self.filename).suffix
            if extension == '.html':
                pl.export_html(self.filename)
                pl.show()
            elif extension == '.gltf' or extension == '.glb':
                pl.export_gltf(self.filename)
                pl.show()
            elif extension == '.png':
                pl.show()
                pl.screenshot(self.filename)
            elif extension == '.obj':
                pl.export_obj(self.filename)
                pl.show()
            elif extension == '.vtk':
                pl.export_vtksz(self.filename)
                pl.show()

        else:
            pl.show()

        pl.close()
        
