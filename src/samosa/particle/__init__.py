#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
===============
samosa/particle
===============

This submodule defines Particle class, which constructs polyhedral particles
with prescribed point group symmetry. From the structure and symmetry of the
particle, anisotropic coupling matrix may also be calculated.
"""

import numpy as np

from samosa.symmetry.group_utils import Groupi, Orbit

from samosa.utils.type_checks import ArrayType, NoneType, is_None, not_None, \
                                     check_type, check_len, check_in_list

from samosa.utils.errors import custom_format_warning 

import warnings
warnings.formatwarning = custom_format_warning

"""
-------------------------------------------------------------------------------
Particle class
-------------------------------------------------------------------------------
"""

class Particle:
    """
    Defines a container for storing all relevant structural information,
    symmetry, and interactions about a polyhedral particle. 
    """

    # Particle intializers
    def __init__(self, 
                 symmetry_group, 
                 face_positions, 
                 calculate_faces=False):
        """
        Constructs a particle from point group symmetry and face coordinates 
        that match the basis of the group operators.  
        (not necessary to have all faces)

        Arguments:
        symmetry_group - Group, point group of the particle shape;

        face_positions - list, face coordinates written in the basis of the 
                         symmetry_group.

        """
        # Type checks
        check_type('symmetry_group', symmetry_group, Group)
        check_type('face_positions', face_positions, list)

        # Calculate the positions of the faces and store the symmetry
        # properties
        self.__face_groups = Orbit.sort_into_orbits(face_positions, 
                                                    symmetry_group)

        # Place patches on the particle


    @classmethod
    def from_lattice(cls, lattice):
        """
        Shortcut to initialize a Particle object from a Lattice object.
        
        Arguments:
        lattice - Lattice, periodic structure, for which the nearest-neighbour
                  connections define the shape of the particle.
        """
        # Type checks
        check_type('lattice', lattice, Lattice)
        
        # Calculate the list of positions of the nearest-neighbours
        face_positions = lattice.nearest_neighbours

        # Calculate the point group of the unit cell
        symmetry_group = lattice.point_group
        
        return cls(face_positions, symmetry_group)

    @clssmethod
    def simplex(cls, simplex_dimension):
        """
        Shortcut to initialize a Particle object from a simplex of some 
        dimensionality.

        Arguments:
        simplex_dimension - int, dimension of the simplex particle.
        """
        # Type checks
        check_type('simplex_dimension', simplex_dimension, int)

        pass
        
    # Particle methods
    

    # Particle properties 
    @property
    def face_groups():
        """
        A list of independent orbits of particle faces.
        """
        return self.__face_groups


    # Output summary functions
