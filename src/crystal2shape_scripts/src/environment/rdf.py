"""Radial Distribution Function (RDF) module.

This module provides classes to calculate and plot the radial distribution function
for particle configurations using the Freud analysis library.
"""

import numpy as np
import freud

from ..visualization.plot_xy import PlotXY  

class rdf_class:
    """Computes the radial distribution function (RDF) using the freud library."""
    
    def __init__(self):
        pass

    def rdf_func(self, positions, box_arr, rmax=3.0, bins=100, show_rdf="False"):
        """Computes the RDF and optionally plots it.

        Args:
            positions (np.ndarray or List[List[float]]): List of positions of the particles.
            box_arr (freud.box.Box): Freud box object defining the simulation box.
            rmax (float): Maximum cutoff distance. Defaults to 3.0.
            bins (int): Number of bins in the RDF histogram. Defaults to 100.
            show_rdf (str): 'True' to plot the RDF, 'False' otherwise. Defaults to "False".
        """
        RDF = freud.density.RDF(bins, rmax)  # RDF
        aq = freud.AABBQuery(box_arr, positions)
        RDF.compute(system=aq, reset=False)

        if show_rdf == "True":
            PlotXY_obj = PlotXY()
            PlotXY_obj.plot_XY_func(RDF.bin_centers, RDF.rdf, xlabel='r', ylabel='g(r)', title='RDF')
