"""Bond Order Diagram (BOD) calculation module.

This module provides classes to calculate neighbor vectors, bond separations,
and neighbor lists for building bond order diagrams.
"""

import numpy as np
import freud
import matplotlib.pyplot as plt

class bod_class:
    """Calculates bond order connections and relative neighbor vectors.

    Attributes:
        positions (np.ndarray or List[List[float]]): List of particle positions.
        box_arr (freud.box.Box): Freud box object defining the simulation box.
        rcut (float): Cutoff radius.
    """

    def __init__(self, positions, box_arr, rcut):
        """Initializes the BOD class with particle coordinates and box vectors.

        Args:
            positions (np.ndarray or List[List[float]]): List of positions.
            box_arr (freud.box.Box): Freud box object.
            rcut (float): Cutoff radius.
        """
        self.positions = positions
        self.box_arr = box_arr
        self.rcut = rcut

    def bod_func(self):
        """Computes relative neighbor position vectors and separation arrays.

        Returns:
            Tuple[List[List[float]], List[List[List[float]]], List[List[int]]]: A tuple containing:
                - List of relative displacement vectors for all neighbors.
                - Particle-wise grouped lists of displacement vectors.
                - Particle-wise lists of neighbor indices.
        """
        position_arr = []
        aq = freud.AABBQuery(self.box_arr, self.positions)
        query_result = aq.query(self.positions, dict(r_min=0, r_max=self.rcut, exclude_ii=True))
        nlist = query_result.toNeighborList()
        separation_arr = [[] for _ in range(len(self.positions))]
        neighbor_li = [[] for _ in range(len(self.positions))]
        for (i, j) in nlist:
            vec = self.box_arr.wrap(self.positions[j] - self.positions[i])
            d = np.linalg.norm(vec)
            position_arr.append(vec)
            separation_arr[i].append(vec)
            neighbor_li[i].append(j)

        return position_arr, separation_arr, neighbor_li
