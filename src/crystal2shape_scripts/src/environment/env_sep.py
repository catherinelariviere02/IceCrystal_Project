"""Local environment separation module.

This module provides tools for clustering and categorizing local environments 
of particles using Freud cluster analysis.
"""

import numpy as np
from itertools import chain
import freud

class env_sep_class():
    """Separates and clusters the local environment of the particles.

    Attributes:
        cluster_centers (np.ndarray): Coordinates of the centroids of the clusters.
        group_arr (List[List[int]]): Cluster label indices grouped per particle.
        unique_arr (List[List[int]]): Unique list of environment configurations.
        original_chosen_particle_arr (List[int]): IDs of particles sharing similar environments.
        brav_estim (float): Estimated percentage of particles matching the dominant environment.
    """

    def __init__(self):
        pass


    def env_sep_func(self, position_arr, separation_arr, particleids, original_particleids, rcut, positions):
        """Filters, clusters, and matches unique neighbor environments.

        Args:
            position_arr (np.ndarray or List[List[float]]): Array of neighbor coordinates relative to center particles.
            separation_arr (List[List[float]]): Particle-wise neighbor distance lists.
            particleids (List[int]): Replicated system particle IDs.
            original_particleids (List[int]): Original unit cell particle IDs.
            rcut (float): Radial cutoff distance.
            positions (np.ndarray or List[List[float]]): Coordinates of all system particles.
        """
        # Freud clustering
        cl = freud.cluster.Cluster()
        b = 2*rcut
        box = freud.box.Box(Lx=b, Ly=b, Lz=b,xy=0.0, xz=0.0, yz=0.0, is2D=False)
        system = freud.AABBQuery(box, position_arr)
        # rmax = float(input("Enter max distance for clsutering : "))
        rmax = 0.05 # Defined for ideal lattice
        cl.compute(system, neighbors={"r_max": rmax})
        n_clusters = cl.num_clusters
        labels = list(cl.cluster_idx)
        labels = np.array([int(t) for t in labels])

        cluster_ids_filtered = list(set(labels))
        cluster_centers, group_arr, unique_arr, choice, original_chosen_particle_arr = [], [], [], [], []
        print("Number of clusters : ", len(cluster_ids_filtered))
        cluster_centers = []
        for c in cluster_ids_filtered:
            indices = [i for i, e in enumerate(labels) if e == c]
            posi = np.average(np.array([position_arr[t] for t in indices]), axis=0)
            cluster_centers.append(posi)

        cluster_centers = np.array(cluster_centers)

        separation_arr_len = [len(t) for t in separation_arr]
        separataion_arr_flattened = list(chain.from_iterable(separation_arr))

        group_arr = []
        c = 0
        for i in range(len(separation_arr_len)):
            sorted_arr = labels[c:c+separation_arr_len[i]]
            sorted_arr.sort()
            # print(c, c+separation_arr_len[i], sorted_arr.tolist())
            # print(sorted_arr.tolist())
            group_arr.append(sorted_arr.tolist())
            c = c+separation_arr_len[i]

        #****************************************************************
        # Environment matching based on local neighbors 
        # Get unique environments i.e. cluster ids per particle
        c = 1.0
        unique_arr, count_arr, equiv_particle_arr = [], [], []
        particle_counter = 0
        for g in group_arr:
            if len(g) != 0:
                if len(unique_arr) == 0:
                    if len(g) != 0:
                        unique_arr.append(g)
                        count_arr.append(1)
                        equiv_particle_arr.append([particle_counter])
                else:
                    count = 0
                    perm_arr = ['n' for _ in range(len(unique_arr))]
                    for k in range(len(unique_arr)):
                        li = list(set(unique_arr[k]).intersection(list(set(g))))
                        if len(li)/len(g) < c:  # unique
                            count = count + 1
                        else:
                            perm_arr[k] = 'y'

                    if count == len(unique_arr):   # unique
                        if len(g) != 0:
                            unique_arr.append(g)
                            count_arr.append(1)
                            equiv_particle_arr.append([particle_counter])
                    else:
                        for k in range(len(unique_arr)):
                            if perm_arr[k] == 'y':
                                count_arr[k] = count_arr[k] + 1
                                equiv_particle_arr[k].append(particle_counter)

            else:
                if g not in unique_arr:
                    if len(g) != 0:
                        unique_arr.append(g)
                        count_arr.append(1)
                        equiv_particle_arr.append([particle_counter])
                else:
                    for k in range(len(unique_arr)):
                        if len(unique_arr[k]) == 0:
                            count_arr[k] = count_arr[k] + 1
                            equiv_particle_arr[k].append(particle_counter)

            particle_counter = particle_counter + 1

        unique_arr = unique_arr   # Unique local environment
        # print(unique_arr)
        max_count = np.max(count_arr)
        index = [i for i, e in enumerate(count_arr) if e == max_count]
        unique_arr_max = unique_arr[index[0]]
        equiv_particles = equiv_particle_arr[index[0]]
        chosen_particle_arr = equiv_particles[:]
        original_chosen_particle_arr = []   # Particle ids sharing similar environment
        for z in range(len(chosen_particle_arr)):
            indi = [i for i, e in enumerate(original_particleids) if e == particleids[chosen_particle_arr[z]]][0]
            original_chosen_particle_arr.append(original_particleids[indi])

        particle_arr_env = [[] for _ in range(len(unique_arr))]
        for j in range(len(group_arr)):
            for k in range(len(unique_arr)):
                if group_arr[j] == unique_arr[k]:
                    particle_arr_env[k].append(j)

        # print(len(chosen_particle_arr))
        # print(particle_arr_env)
        for u in range(len(unique_arr)):
            if len(unique_arr[u]) == 0:
                unique_arr.pop(u)

        # print("Environment percentage for one single type : ", len(original_chosen_particle_arr)/len(positions))
            
        brav_estim = round(len(original_chosen_particle_arr)*100/len(positions), 0)
        self.cluster_centers = cluster_centers
        self.group_arr = group_arr
        self.unique_arr = unique_arr
        self.original_chosen_particle_arr = original_chosen_particle_arr
        self.brav_estim = brav_estim