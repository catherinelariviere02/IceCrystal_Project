"""
Make Hierarchy Module
=====================

This module defines the :class:`Hierarchy` class, which constructs a hierarchical tree-like
representation of neighbor shells around a reference particle. The hierarchy groups neighboring
particles based on their distances and projections along bond directions.
"""

import numpy as np
import freud
from scipy.spatial import Delaunay, ConvexHull

class Hierarchy:
    """Creates a hierarchical representation of a particle's local coordinate environment.

    This representation allows systematic plane-cut truncations along coordinate shells
    from the innermost to the outermost neighbors.

    Attributes:
        hierarchy_arr (List[List[int]]): Hierarchical arrangement of particle IDs based on
            their distances and spatial projections from the reference particle.
        hierarchical_points_coords (List[np.ndarray]): Coordinates of the particles sorted
            hierarchically (by descending distance).
        hierarchical_particle_ids (List[int]): Particle IDs corresponding to the sorted coordinates.
    """
    def __init__(self):
        """Initializes the Hierarchy object."""
        pass

    def make_hierarchy(self, system, particle_coords, max_distance, hierarchy_tol, r=2):
        """Constructs the neighbor shell hierarchy based on distance and normal projections.

        Groups neighbor particles into radial lineages starting from the closest coordination shell
        and proceeding outwards.

        Args:
            system (gsd.hoomd.Snapshot): System object containing particle information.
            particle_coords (List[np.ndarray] or np.ndarray): Relative coordinates of the neighbors.
            max_distance (float): Maximum cutoff distance from the reference particle.
            hierarchy_tol (float): Tolerance for grouping projection distances along normal vectors.
            r (int, optional): Rounding factor for distance calculations. Defaults to 2.
        """
        # Sorting nearest neighbors according to the descending order of distance from the reference particle
        dist_ = [round(float(np.linalg.norm(j)), r) for j in particle_coords]
        original_dist_arr = dist_[:]
        dist_ = list(set(dist_))  # Unique distances
        dist_.sort(reverse=True)
        point_ids = [i for i in range(len(particle_coords))]  # Assuming particle IDs are their indices
        sorted_info_ids, sorted_info_poly_vertices = [], []
        for t in range(len(dist_)):
            indices = [k for k, e in enumerate(original_dist_arr) if e == dist_[t]]
            for i in indices:
                if point_ids[i] not in sorted_info_ids:
                    sorted_info_ids.append(point_ids[i])
                    sorted_info_poly_vertices.append(particle_coords[i])

        hierarchical_points_coords = (sorted_info_poly_vertices)
        hierarchical_particle_ids = (sorted_info_ids)

        # Construction of the hierarchy
        hierarchy_arr = []
        track_points, sorted_vertices_larger_dist_ids = [], []
        for n in range(len(hierarchical_points_coords)):
            # Finding points further along the normal direction from the current point
            if np.linalg.norm(hierarchical_points_coords[n]) > 0:  # Beyond closest neighbors of the reference particle
                normal = np.array(hierarchical_points_coords[n]) / np.linalg.norm(hierarchical_points_coords[n])
                sorted_vertices_larger_dist_ids = [int(hierarchical_particle_ids[t]) for t in range(len(hierarchical_points_coords)) if t != n and round(float(np.dot(np.array(hierarchical_points_coords[t]), normal)) - float(np.linalg.norm(np.array(hierarchical_points_coords[n])) * (1 - hierarchy_tol)), r) >= 0.0 and hierarchical_points_coords[t].tolist() != hierarchical_points_coords[n].tolist() and hierarchical_points_coords[t].tolist() != [0, 0, 0]]
                
                for t in sorted_vertices_larger_dist_ids:
                    if t not in track_points:
                        # Check if the point is already considered for truncation
                        track_points.append(int(t))  # Track the points which are already considered for truncation

            else: # Closest neighbors of ref_particle
                sorted_vertices_larger_dist_ids = []
                for h in range(len(hierarchical_particle_ids)):
                    if int(hierarchical_particle_ids[h]) not in track_points and hierarchical_particle_ids[h] != sorted_info_ids[-1]:  # Check if the point is already considered for truncation
                        sorted_vertices_larger_dist_ids.append(int(hierarchical_particle_ids[h]))
                        track_points.append(int(hierarchical_particle_ids[h]))

            if len(sorted_vertices_larger_dist_ids) > 0: 
                for t in sorted_vertices_larger_dist_ids:
                    counter = 0
                    for g in range(len(hierarchy_arr)):
                        if t == hierarchy_arr[g][-1]:
                            hierarchy_arr[g].append(int(hierarchical_particle_ids[n]))
                        else:
                            counter = counter + 1

                    if counter == len(hierarchy_arr):
                        hierarchy_arr.append([int(t), int(hierarchical_particle_ids[n])])

            else:
                hierarchy_arr.append([int(hierarchical_particle_ids[n])])

        hierarchy_arr = self._remove_subset(hierarchy_arr)

        # Reverse each subarr of the hierarchical array --> from innermost to outermost direction from reference particle
        reverse_hierarchical_arr = [np.array(hierarchy_arr[j][::-1]).tolist() for j in range(len(hierarchy_arr))]
        hierarchy_arr = reverse_hierarchical_arr[:]

        self.hierarchy_arr = hierarchy_arr
        self.hierarchical_points_coords = hierarchical_points_coords
        self.hierarchical_particle_ids = hierarchical_particle_ids

    def _remove_subset(self, dummy_hierarchy_arr):
        """Filters out redundancy in the hierarchy by removing subset lineages.

        If a lineage is fully contained within a longer lineage, it is discarded.

        Args:
            dummy_hierarchy_arr (List[List[int]]): Unfiltered hierarchy lineages.

        Returns:
            List[List[int]]: Cleaned list of unique, non-overlapping maximal lineages.
        """
        hierarchy_arr = []
        for h in range(len(dummy_hierarchy_arr)):
            cntr = 0
            for k in range(len(dummy_hierarchy_arr)):
                if k != h:
                    intersection_ = np.intersect1d(dummy_hierarchy_arr[h], dummy_hierarchy_arr[k]).tolist()
                    if intersection_ is not None:
                        if np.sort(intersection_).tolist() != np.sort(dummy_hierarchy_arr[h]).tolist():
                            cntr = cntr + 1

            if cntr == len(dummy_hierarchy_arr) - 1 and dummy_hierarchy_arr[h] not in hierarchy_arr:
                hierarchy_arr.append(dummy_hierarchy_arr[h])

        return hierarchy_arr