import numpy as np
import coxeter
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull
from scipy.linalg import solve
from pointgroup import PointGroup
from itertools import chain

from ..visualization.plot_poly import plot_poly_func
import matplotlib.pyplot as plt
from ..visualization.pyvista_plot import pyvista_plot_class
from ..symmetry.detect_pg import detect_pg_class

class truncation_class():
    """
    Class to handle the truncation of a convex hull with a plane.
    This class computes the intersection of a convex hull with a plane
    and truncates the convex hull based on the intersection points.
    It also provides a method to visualize the truncated convex hull.

    """
    def __init__(self):
        '''for key, value in kwargs.items():
            setattr(self, key, value)'''
        
        pass

    # Function to check intersection of a line segment and a plane
    def line_plane_intersection(self, p1, p2, plane_normal, d):
        v = p2 - p1
        dot_product = np.dot(plane_normal, v)
        if abs(dot_product) < 1e-6:  # Line is parallel to plane
            return None
        t = (-d - np.dot(plane_normal, p1)) / dot_product
        if 0 <= t <= 1:
            return p1 + t * v
        else:
            return None

    def intersect_convex_hull_with_plane(self, points, plane_normal, plane_point):
        """
        Computes the intersection of a convex hull with a plane.

        Args:
            points: A numpy array of points representing the convex hull.
            plane_normal: A numpy array representing the normal vector of the plane.
            plane_point: A numpy array representing a point on the plane.

        Returns:
            A numpy array of points representing the intersection polygon, or None if no intersection.
        """
        hull = ConvexHull(points)
        # points = [points[t] for t in hull.vertices]
        # poly = coxeter.shapes.ConvexPolyhedron(points)
        
        d = -np.dot(plane_normal, plane_point)
        intersection_points = []
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    p1 = points[simplex[i]]
                    p2 = points[simplex[j]]
                    intersection = self.line_plane_intersection(p1, p2, plane_normal, d)
                    if intersection is not None:
                        '''intersection = [round(float(intersection[0]), r), round(float(intersection[1]), r), round(float(intersection[2]), r)]
                        if intersection not in intersection_points:
                            intersection_points.append(intersection)'''
                        
                        # Remove the very close vertices for precision issues
                        k = 0
                        if len(intersection_points) == 0:
                            intersection_points.append(intersection)
                        else:
                            for k in range(len(intersection_points)):
                                # Compare two list for item similarity
                                if np.allclose(intersection, intersection_points[k], atol=1e-2):
                                    pass
                            if k == len(intersection_points)-1:
                                intersection_points.append(intersection)

        return intersection_points


    def truncation_func(self, ref_point, poly_vertices, normals, trn_pts, counter, **kwargs):
        """
        Parameters
        ----------

        poly_vertices : list
            List of vertices representing the convex polyhedron.

        normal : list
            List of normal vectors for the truncation planes.

        trn_pts : list
            List of truncation points where the planes intersect the polyhedron.

        counter : int
            Counter to keep track of the number of truncations performed.

        Attributes
        ----------
        poly_vertices : list
            Updated list of vertices after truncation.
            
        """

        dic_data = {}
        for key, value in kwargs.items():
            dic_data[str(key)] = value

        r = 2 # Rounding precision
        hull = ConvexHull(poly_vertices)
        poly_vertices = [poly_vertices[t] for t in hull.vertices]
        poly = coxeter.shapes.ConvexPolyhedron(poly_vertices)
        volume = poly.volume
        edges = poly.edges
        faces = poly.faces
        pointgroup = None # Dummy variable for point group symmetry

        vertices_rnd = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in poly_vertices]
        truncation_pts = []
        pts_all = []
        req_pts = []
        poly_vertices_g = [t.tolist() for t in poly_vertices]
        original_pts = poly_vertices_g[:]
        trunc_vol_sum = 0
        for t in range(len(trn_pts)):
            intersection_point = trn_pts[t]
            normal = normals[t]
            d = -np.dot(normal, intersection_point)
            pts = self.intersect_convex_hull_with_plane(np.array(poly_vertices_g), normal, intersection_point)

            if pts != None:
                truncation_pts = pts[:]

            req_pts = []
            for p in truncation_pts:
                req_pts.append(p)

            for p in poly_vertices_g:
                vec =  np.array(p) - np.array(intersection_point)
                test_val = round(float(np.dot(normal, vec)), 2)
                if test_val <= 0:
                    req_pts.append(p)

            # Visulaize each truncated faces in matplotlib
            '''fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            env = [[round(float(m[0]), r), round(float(m[1]), r), round(float(m[2]), r)] for m in req_pts]
            env_arr = [env, original_pts]
            for env in env_arr:
                hull = ConvexHull(env)
                env = [env[u] for u in hull.vertices]
                poly = coxeter.shapes.ConvexPolyhedron(env)
                original_pts = np.array(original_pts)
                ax.plot([0, intersection_point[0]], [0, intersection_point[1]], [0, intersection_point[2]], c='r', linewidth=1.0)
                fig, ax = plot_poly_func(self, poly, fig, ax, env)
            plt.show()
            plt.close()'''

            # Dictionary with the extra parameters
            dic_data = {}
            for key, value in kwargs.items():
                dic_data[str(key)] = value

            keys = list(dic_data.keys())
            # Calculate Point group symmetry
            if "calc_pointgroup" in keys and dic_data["calc_pointgroup"] == True:
                req_pts_temp = [[round(float(p[0]), r), round(float(p[1]), r), round(float(p[2]), r)] for p in req_pts]
                hull = ConvexHull(req_pts_temp)
                hull_verts = [req_pts_temp[p] for p in hull.vertices]

                # Symmetry axes and angles detection -- Sumitava's method
                '''poly = coxeter.shapes.ConvexPolyhedron(hull_verts)
                detect_pg_obj = detect_pg_class()
                detect_pg_obj.detect_pg_func(np.array(hull_verts), len(poly.edges), len(poly.faces))
                info_symmetry_dict = detect_pg_obj.symmetry_dict
                pointgroup = info_symmetry_dict'''

                # Point group symmetry detection -- PointGroup package
                # https://pointgroup.readthedocs.io/en/latest/
                vertices = np.array([(p - np.average(hull_verts, axis=0)) for p in hull_verts])
                pg = PointGroup(positions=vertices, symbols= ['C']*len(vertices), tolerance_eig=0.01, tolerance_ang=5)
                pointgroup = pg.get_point_group()
                print(pointgroup)

                self.pointgroup = pointgroup

            # Visualize the truncated polyhedron
            if "show_poly" in keys and dic_data["show_poly"] == "True":
                if counter < 10:
                    counter_str = "0" + str(counter)
                else:
                    counter_str = str(counter)

                outfilename = dic_data["directory"] + "temp_files/" + dic_data["shape_id"] +  "_truncated_polyhedron_" + counter_str + ".png"
                posi_uc = [dic_data["positions"][j] for j in dic_data["uc_ids"]]
                poly_vertices_temp = [np.array(req_pts) + ref_point]
                pyvista_plot_obj = pyvista_plot_class(uc_box=dic_data["uc_box"], 
                                                      positions=posi_uc, 
                                                      types=dic_data["type_uc"], 
                                                      wyckoff_uc=dic_data["wyckoff_uc"],
                                                      poly_vertices=poly_vertices_temp, 
                                                      extra_positions=dic_data["extra_positions"], 
                                                      line_points=dic_data["line_points"], 
                                                      line_colors="red",
                                                      extra_particles_types=dic_data["extra_particles_types"],
                                                      extra_particles_wyckoffs=dic_data["extra_particles_wyckoffs"],
                                                      center=[trn_pts[t]+ref_point],
                                                      direction=[normals[t]],
                                                      text=None,
                                                      filename=outfilename,
                                                      color="teal")
                pyvista_plot_obj.pyvista_plot_func()

            if len(req_pts) > 3:
                poly_vertices_g = req_pts[:]
            

        pts = req_pts[:]
        if len(pts) > 3:
            updated_pts = [pts[0]]
            for i in range(1, len(pts)):
                # Compare two list for item similarity
                if not np.allclose(pts[i], updated_pts[-1], atol=1e-2):
                    updated_pts.append(pts[i])

            pts = updated_pts[:]
            hull = ConvexHull(pts)
            pts = [pts[p] for p in hull.vertices]


        # Attribute the updated vertices to the class variable
        self.updated_vertices = pts[:]
    
    