"""JSON writer module.

This module provides classes to export polyhedral shape vertices, moments of
inertia, and metadata to formatted JSON files.
"""

import json
import numpy as np
from scipy.spatial import ConvexHull
import coxeter

class JSON_writer:
    """Exports polyhedral shapes and metadata to standard JSON format files."""

    def __init__(self):
        pass

    def JSON_writer_func(self, shape_poly, shape_id, directory, basis_type_, counter):
        """Writes shape polyhedron coordinate and moment datasets to a JSON file.

        Computes the convex hull of shape vertices, centers and normalizes them,
        diagonalizes the inertia tensor, and writes the results to a file named
        with the shape ID and active basis type.

        Args:
            shape_poly (np.ndarray): Array of shape vertices.
            shape_id (str): Unique shape identifier (typically CIF filename base).
            directory (str): Path of directory to output files into.
            basis_type_ (List[str]): Active basis type names.
            counter (int): Index of active basis type to use for file naming.
        """

        r = 5  # Number of decimal places to round the coordinates

        hull = ConvexHull(shape_poly)
        shape_poly = [shape_poly[u] for u in hull.vertices]
        shape_poly = [(t - np.average(shape_poly, axis=0)) for t in shape_poly]
        shape_poly = [t.tolist() for t in shape_poly]
        shape_poly = [[round(float(t[0]), r), round(float(t[1]), r), round(float(t[2]), r)] for t in shape_poly]
        
        # Moment of Inertia
        hull = ConvexHull(shape_poly)
        shape_poly = [shape_poly[u] for u in hull.vertices]
        poly = coxeter.shapes.ConvexPolyhedron(shape_poly)
        poly.diagonalize_inertia()
        moi = poly.inertia_tensor.round(3)
        moi_arr = [moi[0][0], moi[0][1], moi[0][2], moi[1][1], moi[1][2], moi[2][2]]

        # Write JSON file
        data = {}
        data["0_Id"] = shape_id
        data["1_Name"] = shape_id
        data["2_ShortName"] = shape_id
        data["3_Commentt"] = "Vertices are in principal frame,centered, and correspond to unit volume"
        data["4_Volume"] = round(poly.volume, 5)
        data["5_center_of_mass"] = [0.0, 0.0, 0.0]
        data["6_moment_of_inertia"] = moi_arr[:]
        data["7_comment_about_moment_of_inertia"] = "The components are stored as xx, xy, xz, yy, yz, zz"
        data["8_vertices"] = shape_poly[:]

        output_json = directory +  "shape_" + shape_id + "_" + basis_type_[counter] + "_unit_volume_principal_frame.json"
        with open(output_json, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print(f"JSON file written successfully.")
