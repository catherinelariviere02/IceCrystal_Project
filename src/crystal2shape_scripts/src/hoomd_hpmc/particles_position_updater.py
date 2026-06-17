"""HOOMD custom action updater module.

This module provides custom HOOMD-blue actions to update particle coordinates
dynamically during Monte Carlo compression or NVT steps.
"""

import hoomd
import numpy as np
import freud, rowan

class ParticlesPositionAction(hoomd.custom.Action):
    """Custom action to update particle positions based on current box vectors.

    This action is intended to be used with a HOOMD-blue simulation to adjust
    the particle positions dynamically during the simulation.

    Attributes:
        particles_position (np.ndarray): Particle positions mapping array.
    """

    def __init__(self, particles_position):
        """Initializes the action.

        Args:
            particles_position (np.ndarray): Particle positions to track.
        """
        super().__init__()
        self.particles_position = particles_position

    def act(self, timestep):
        """Updates particle coordinates and adjusts the wall potential bounds.

        Args:
            timestep (int): The current simulation timestep.
        """
        snapshot = self._state.get_snapshot()
        if snapshot.communicator.rank == 0:
            random_quat = rowan.random.rand(1)
            updated_positions = rowan.rotate(random_quat, snapshot.particles.position)
            box = self._state.box

            # Update wall positions 
            box_mat = freud.box.Box(Lx=box.Lx, Ly=box.Ly, Lz=box.Lz, xy=box.xy, xz=box.xz, yz=box.yz).to_matrix()
            lattice_vecs = np.transpose(box_mat) # Getting box vectors
            wall_directions = []
            wall_directions.append(lattice_vecs[2]*0.5) # Walls in +z direction
            wall_directions.append(-lattice_vecs[2]*0.5) # Walls in -z direction

            walls_arr = []
            for w in range(len(wall_directions)):
                wall = hoomd.wall.Plane(origin=tuple(wall_directions[w]), normal=tuple(-wall_directions[w]/np.linalg.norm(wall_directions[w])))
                walls_arr.append(wall)
                
            self.wall_potential = hoomd.hpmc.external.wall.WallPotential(walls_arr)
