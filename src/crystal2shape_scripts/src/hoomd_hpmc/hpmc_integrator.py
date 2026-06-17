import hoomd
import coxeter
import numpy as np

def create_simulation(communicator, shape_vertices, types, move_sizes):
    """
    Create a HOOMD simulation with a HPMC integrator for the given shape vertices.

    Parameters:
    -----------

    communicator: Object
           HOOMD communicator for parallel execution.

    shape_vertices: list of list of lists 
           Vertices of the convex polyhedron shapes ofr each particle type

    types: list of list of str
           Types of the shapes corresponding to the vertices for each particle type.

    move_sizes: list of list of tuples
           Move sizes for the shapes, where each tuple contains (a_val, d_val) for each particle type.

    Returns:
    --------
    simulation: hoomd.Simulation
           A HOOMD simulation object with the HPMC integrator set up.

    """

    cpu = hoomd.device.CPU(communicator=communicator)
    simulation = hoomd.Simulation(device=cpu, seed=np.random.randint(1, 1e4))
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    
    for i in range(len(shape_vertices)):
       poly = coxeter.shapes.ConvexPolyhedron(shape_vertices[i])
       mc.shape[types[i]] = dict(vertices=poly.vertices)
       mc.a[types[i]] = move_sizes[i][0]
       mc.d[types[i]] = move_sizes[i][1]
        
    simulation.operations.integrator = mc

    return simulation