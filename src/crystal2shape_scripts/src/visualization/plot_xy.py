"""Plotting module for XY data.

This module provides basic XY plotting utilities using matplotlib.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class PlotXY:
    """Class to plot simple 2D XY data plots."""
    
    def __init__(self):
        pass

    def plot_XY_func(self, x_data, y_data, xlabel='r', ylabel='g(r)', title='RDF'):
        """Generates and displays a 2D line plot.

        Args:
            x_data (List or np.ndarray): X data for the plot.
            y_data (List or np.ndarray): Y data for the plot.
            xlabel (str): Label for the X-axis. Defaults to 'r'.
            ylabel (str): Label for the Y-axis. Defaults to 'g(r)'.
            title (str): Title of the plot. Defaults to 'RDF'.
        """
        figure_rdf = plt.figure(figsize=(8, 6))
        ax_rdf = figure_rdf.add_subplot(111)
        ax_rdf.plot(x_data, y_data)
        ax_rdf.set_xlabel(xlabel)
        ax_rdf.set_ylabel(ylabel)
        ax_rdf.set_title(title, fontsize=6)
        figure_rdf.tight_layout()
        plt.show()
        plt.close(figure_rdf)
