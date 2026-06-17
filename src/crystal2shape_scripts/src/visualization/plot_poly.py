import numpy as np
import matplotlib.pyplot as plt
import coxeter

def plot_poly_func(self, poly, fig, ax, shape_vertices, include_tensor=False, length_scale=3):
    """Plot a polyhedron a provided set of matplotlib axes.

    The include_tensor parameter controls whether or not the axes
    of the inertia tensor are plotted. If they are, then the
    length_scale controls how much the axis vectors are extended,
    which is purely for visualization purposes.

    Parameters
    ----------
    poly : coxeter.shapes.ConvexPolyhedron
        The polyhedron to plot.

    fig : matplotlib.figure.Figure
        The figure to plot the polyhedron in.

    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axes to plot the polyhedron in.

    shape_vertices : list of tuples
        List of vertices of the shape to be plotted, where each vertex is represented as a tuple


    Returns
    -------
    fig : matplotlib.figure.Figure

    ax : matplotlib.axes._subplots.Axes3DSubplot
        The figure and axes with the polyhedron plotted.

    Notes
    -----   
    This function generates a triangulation of the polyhedron's surface and plots it using
    `plot_trisurf`. It also manually adds lines for the edges of the polyhedron
    to ensure that the edges are clearly visible. If `include_tensor` is set to True, it plots
    the axes of the inertia tensor as arrows originating from the center of the polyhedron.

    """
    # Generate a triangulation for plot_trisurf.
    vertex_to_index = {tuple(v): i for i, v in enumerate(poly.vertices)}
    triangles = [
        [vertex_to_index[tuple(v)] for v in triangle]
        for triangle in poly._surface_triangulation()
    ]

    # Plot the triangulation to get faces, but without any outlines because outlining
    # the triangulation will include lines along faces where coplanar simplices intersect.
    verts = poly.vertices
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=triangles,
        # Make the triangles partly transparent.
        color=tuple([*plt.get_cmap("tab10").colors[4], 0.3]),
    )

    # Add lines manually.
    for face in poly.faces:
        verts = poly.vertices[face]
        verts = np.concatenate((verts, verts[[0]]))
        ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], c="k", lw=0.4)

    # If requested, plot the axes of the inertia tensor.
    if include_tensor:
        centers = np.repeat(poly.center[np.newaxis, :], axis=0, repeats=3)
        arrows = poly.inertia_tensor * length_scale
        ax.quiver3D(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            arrows[:, 0],
            arrows[:, 1],
            arrows[:, 2],
            color="k",
            lw=3,
        )
    
    ax.scatter(np.array(shape_vertices)[:,0], np.array(shape_vertices)[:,1], np.array(shape_vertices)[:,2], s=20, marker='o', facecolors='none', edgecolors='red')
    ax.view_init(elev=30, azim=-90)
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    ax.set_axis_off()
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim([center[0] - radius, center[0] + radius])
    ax.set_ylim([center[1] - radius, center[1] + radius])
    ax.set_zlim([center[2] - radius, center[2] + radius])
    ax.tick_params(which="both", axis="both", labelsize=0)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.tight_layout()
    # plt.show()
    # plt.close()

    return fig, ax