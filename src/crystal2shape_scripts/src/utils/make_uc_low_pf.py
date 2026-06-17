import numpy as np
import rowan, freud, coxeter, math
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from scipy.linalg import orth
from scipy.spatial.transform import Rotation as Rot
from itertools import permutations
from itertools import chain


from ..visualization.pyvista_plot import pyvista_plot_class

class ConstructUnitCell():
    def __init__(self):
        pass
    def make_uc(self, system, unitcell, target_pf, vertices_arr, uc_particle_indices, basis_type, n_repeats):
        uc_positions = [system.positions[i] for i in uc_particle_indices]
        box = system.box

        # At target pf
        target_volume = len(uc_positions)/target_pf
        scale_factor = (target_volume/box.volume)**(1/3)
        new_box = freud.box.Box(Lx=box.Lx*scale_factor, Ly=box.Ly*scale_factor, Lz=box.Lz*scale_factor,
                                xy=box.xy, xz=box.xz, yz=box.yz)
        new_positions = np.array(uc_positions)*scale_factor
        new_system = freud.AABBQuery(new_box, new_positions)

        uc = freud.data.UnitCell(new_box, basis_positions=unitcell.fractional_positions)
        n_repeats = (n_repeats, n_repeats, n_repeats)
        (box, positions) = uc.generate_system(num_replicas=n_repeats, scale=1.0)
        orientations = [rowan.random.rand(1) for i in range(len(positions))]
        N = np.prod(n_repeats)
        indices_uc = np.repeat(np.arange(len(uc.basis_positions)), N)
        types = basis_type
        typeids = np.repeat(np.array([0 for _ in range(len(uc_positions))]), np.prod(n_repeats), axis=0).tolist()

        shape_dic_arr = []
        for v in range(len(vertices_arr)):
            verts = [(t - np.average(vertices_arr[v], axis=0)).tolist() for t in vertices_arr[v]]
            shape_dic = {
                "type": "ConvexPolyhedron",
                "rounding_radius": 0.0,
                "vertices": verts
            }
            shape_dic_arr.append(shape_dic)

        self.vertices_arr = vertices_arr
        self.types = types
        self.typeids = typeids
        self.shape_dic = shape_dic_arr


        self.orientations = orientations
        self.box = box
        self.positions = positions
        
