#! /usr/bin/env python3
# Andrey Zelenskiy, 2024-2026

import numpy as np

from samsa.structures import Lattice


def get_particle_data_detailed(
    dimension, space_group_index, positions=[[0, 0, 0]], crystal_parameters=None
):
    """
    Generate particle verex positions and face groups from a full space
    group input.

    Arguments:
    dimension          - int, dimension of the lattice;

    space_group_index  - int, space group number (as per International
                         Tables for Crystallography, volume A);

    positions          - list of ArrayType[3], (default=[[0, 0, 0]]),
                         vertex (Wyckoff) positions written in fractional
                         coordinates. Only inequivalent Wyckoff sites need
                         to be specified;

    crystal_parameters - (default=None), physical parameters of the lattice
                         (unit cell dimensions and angles):

                         if None, initializes the unconstrained parameters
                         randomly;

                         if dict with allowed keys
                         ['a', 'b', 'c', 'alpha', 'beta', 'gamma'],
                         initializes parameters from user input.

    Returns:
    vertex_list - list of float, vertices for all particles;
    face_groups - list of dict, particle face groups.
    """
    lattice = Lattice(
        dimension, space_group_index, positions, crystal_parameters
    )

    # Initialize lists for storing particle data
    vertex_list = []
    face_groups = []
    vor_list = lattice.site_voronoi
    n_neighbours = [len(n_list) for n_list in lattice.site_neighbours.values()]

    for i, vor in enumerate(vor_list.values()):
        # Identify correct Voronoi region
        region = vor.point_region[0]

        # Get indices of the vertices for the region
        vertex_idx = vor.regions[region]

        # Add vertices to the list
        particle_vertices = vor.vertices[vertex_idx]
        vertex_list += [particle_vertices]

        # Collect face groups
        particle_faces = []
        for n in range(1, n_neighbours[i] + 1):
            # Get vertex indices belonging to each face
            if (0, n) in vor.ridge_dict.keys():
                face_vertices = vor.vertices[vor.ridge_dict[(0, n)]]

            else:
                face_vertices = vor.vertices[vor.ridge_dict[(n, 0)]]

            # Get indices as in vertex_list
            face_indx = [
                np.where(np.linalg.norm(particle_vertices - v, axis=1) == 0.0)[
                    0
                ][0]
                for v in face_vertices
            ]
            particle_faces += [face_indx]

        face_groups += [particle_faces]

    return vertex_list, face_groups


def get_particle_data_simple(name, point_group):
    """
    Generate particle verex positions and face groups for one of the simple
    Bravais lattices.

    Arguments:
    name        - str, valid names are: chain, square, hexagonal, cubic,
                  bcc, fcc;

    point_group - Schoenflies symbol of the site point group symmetry.

    Returns:
    vertex_list - list of float, vertices for all particles;
    face_groups - list of dict, particle face groups.
    """
    lattice = Lattice.initialize_simple(name, point_group)

    # Initialize lists for storing particle data
    vertex_list = []
    face_groups = []
    vor_list = lattice.site_voronoi
    n_neighbours = [len(n_list) for n_list in lattice.site_neighbours.values()]

    for i, vor in enumerate(vor_list.values()):
        # Identify correct Voronoi region
        region = vor.point_region[0]

        # Get indices of the vertices for the region
        vertex_idx = vor.regions[region]

        # Add vertices to the list
        particle_vertices = vor.vertices[vertex_idx]
        vertex_list += [particle_vertices]

        # Collect face groups
        particle_faces = []
        for n in range(1, n_neighbours[i] + 1):
            # Get vertex indices belonging to each face
            if (0, n) in vor.ridge_dict.keys():
                face_vertices = vor.vertices[vor.ridge_dict[(0, n)]]

            else:
                face_vertices = vor.vertices[vor.ridge_dict[(n, 0)]]

            # Get indices as in vertex_list
            face_indx = [
                np.where(np.linalg.norm(particle_vertices - v, axis=1) == 0.0)[
                    0
                ][0]
                for v in face_vertices
            ]
            particle_faces += [face_indx]

        face_groups += [particle_faces]

    return vertex_list, face_groups
