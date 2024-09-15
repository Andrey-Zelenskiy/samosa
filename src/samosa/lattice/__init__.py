#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
==============
samosa/lattice
==============

This submodule defines Lattice class, which contains methods for working with
space group symmetries.
"""

import numpy as np

from samosa.symmetry.group_utils import Group
from samosa.symmetry.representations import SpaceGroupElement
from samosa.database import SpaceGroupDatabase


from samosa.api.api_utils import ArrayType, NoneType 
from samosa.api.api_utils import is_None, not_None
from samosa.api.api_utils import check_type, check_in_list
from samosa.api.api_utils import custom_format_warning 

import warnings
warnings.formatwarning = custom_format_warning
"""
-------------------------------------------------------------------------------
Lattice class
-------------------------------------------------------------------------------
"""
class Lattice:
    """
    Defines a container for storing all relevant spatial information about a
    lattice. 
    """

    def __init__(self, dimension, space_group_index):
        """
        Initialize all relevant space group information from the lattice 
        dimension and space group number. 

        Arguments:
        dimension         - int, dimension of the lattice;

        space_group_index - int, space group number (as per International 
                            Tables for Crystallography, volume A).
        """
        # Define a Database object to initialize space group properties
        database = SpaceGroupDatabase()

        # Type checks
        check_type('dimension', dimension, int)
        check_type('space_group_index', space_group_index, int)

        database.check_dimension(dimension)
        database.check_space_group_index(space_group_index, dimension)

        # Initialize symmetry properties
        self.__dimension = dimension
        self.__space_group_index = space_group_index
        sg_t = (self.dimension, self.space_group_index)

        labels = database.space_group_reference[sg_t]
        self.__point_group_symbol, self.__lattice_type = labels
        
        matrix_generators = database.space_group_generators[sg_t]
        self.__symmorphic = True
        generators = []
        for g in matrix_generators:
            G = SpaceGroupElement.from_matrix(g)
            self.__symmorphic *= G.symmorphic
            generators += [G]
        
        self.__space_group = Group(generators, 
                                   name=str(self.space_group_index))

        # Initialize structural properties

    # Lattice methods

    # Lattice properties
    @property
    def dimension(self):
        """
        Lattice dimension (number of basis vectors).
        """
        return self.__dimension

    @property
    def space_group_index(self):
        """
        Index of the space group of the lattice.
        """
        return self.__space_group_index

    @property
    def point_group_symbol(self):
        """
        Schoenflies symbol of the crystallographic point group.
        """
        return self.__point_group_symbol

    @property
    def lattice_type(self):
        """
        Bravais lattice type.
        """
        return self.__lattice_type

    @property
    def space_group(self):
        """
        Space group symmetry of the lattice.
        """
        return self.__space_group

    # Output summary functions
