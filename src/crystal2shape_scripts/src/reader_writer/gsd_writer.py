"""GSD file writer module.

This module provides tools to export HOOMD-blue snapshot frames and configuration
states into standard binary GSD (General Simulation Data) format files.
"""

import numpy as np
import gsd
import gsd.hoomd
import os
import freud

class GSDWriter():
    """Writes system configuration frames and trajectories to GSD files.

    Attributes:
        filename (str): Output GSD file name.
    """

    def __init__(self, filename):
        """Initializes the GSDWriter with a target filename.

        Args:
            filename (str): The output GSD file path.
        """
        self.filename = filename

    def write_frame(self, i, box,  posi, orien, typeids, types, shape_dic):
        """Creates a single HOOMD-blue snapshot frame configuration.

        Args:
            i (int): Frame index step number.
            box (freud.box.Box): The simulation box dimension specifications.
            posi (np.ndarray or List[List[float]]): Particle position coordinates.
            orien (np.ndarray or List[List[float]]): Particle quaternions or orientations.
            typeids (List[int]): List of particle type index identifiers.
            types (List[str]): List of active type names in the simulation.
            shape_dic (Dict[str, Any] or List[Any]): Metadata dict for particle shapes.

        Returns:
            gsd.hoomd.Frame: The constructed simulation frame object.
        """
        frame = gsd.hoomd.Frame()
        frame.configuration.box = [box.Lx, box.Ly, box.Lz, box.xy,  box.xz, box.yz]
        frame.configuration.step = i
        frame.particles.N = len(posi)
        frame.particles.position = posi[:]
        if orien is not None:
            frame.particles.orientation = orien
        frame.particles.typeid = typeids[:]
        frame.particles.types = types[:]
        frame.particles.type_shapes = tuple(shape_dic)
        
        return frame

    def write_GSD(self, box, positions, orientations, type_ids, types, shape_dic, repeat=None):
        """Generates and writes a system configuration to a GSD file.

        Optionally replicates the unit cell box and positions before appending
        the first frame to the output file.

        Args:
            box (freud.box.Box): The unit cell box object.
            positions (np.ndarray): Particle positions.
            orientations (np.ndarray): Particle orientations.
            type_ids (np.ndarray): Particle type IDs.
            types (List[str]): Particle type names.
            shape_dic (Dict[str, Any] or List[Any]): Meta dictionary for shape visualization in OVITO/etc.
            repeat (Tuple[int, int, int]): Replication counts. Defaults to None.
        """
        # Create the system and replicate the unit cell properties
        uc = freud.data.UnitCell(box, box.make_fractional(positions))
        n_repeats = tuple(repeat)
        (box, points) = uc.generate_system(n_repeats)
        positions = points
        N = np.prod(n_repeats)
        # indices = np.repeat(np.arange(len(uc.basis_positions)), N)
        if orientations is not None:
            orientations = np.tile(orientations, (N, 1))
        else:
            orientations = None
        
        type_ids = np.repeat(type_ids, N)

        file = gsd.hoomd.open(name=self.filename, mode='w')
        file.append(self.write_frame(0, box, positions, orientations, type_ids, types, shape_dic))