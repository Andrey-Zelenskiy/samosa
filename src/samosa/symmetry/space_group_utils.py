#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
====================================
samosa/symmetry/space_group_utils.py
====================================

This program defines methods for defining and manipulating space groups.
"""

import numpy as np

from samosa.symmetry.group_utils import Group

from samosa.symmetry.representations import SpaceGroupElement

from samosa.symmetry.point_group_utils import point_group

from samosa.database import SpaceGroupDatabase

from samosa.utils.type_checks import is_None, not_None, check_type, \
                                     check_len, check_shape, ArrayType

def space_group(dimension, sg_index, return_info=False):
    """
    Returns a space group given spatial dimension, as well as the space group
    number.   
    
    Arguments:
    dimension   - int, dimension of the space (lattice);

    sg_index    - int, number of the space group according to the International
                  Tables for Crystallography, vol. A;

    return_info - bool, (default=False), if True, returns a dictionary with 
                  additional space group information.

    Returns: 
    (return_info=False) Group, space group object;
    (return_info=True) dict with keys 
                       ['space_group' : Group,
                 'point_group_symbol' : str,
                       'lattice_type' : str,
                         'symmorphic' : bool], space group information.
    """
    database = SpaceGroupDatabase()

    # Type checks
    check_type('dimension', dimension, int)
    check_type('sg_index', sg_index, int)

    database.check_dimension(dimension)
    database.check_space_group_index(sg_index, dimension)

    # Initialize symmetry properties
    sg_t = (dimension, sg_index)

    labels = database.space_group_reference[sg_t]
    point_group_symbol, lattice_type = labels
    
    order = point_group(point_group_symbol).order

    matrix_generators = database.space_group_generators[sg_t]
    symmorphic = True
    generators = []
    for g in matrix_generators:
        G = SpaceGroupElement.from_matrix(g)
        symmorphic *= G.symmorphic
        generators += [G]
    
    space_group = Group(generators, 
                        name=str(sg_index),
                        order=order,
                        filter_generators=True)

    if return_info:
        info = {'space_group' : space_group,
                'point_group_symbol' : point_group_symbol,
                'lattice_type' : lattice_type,
                'symmorphic' : symmorphic}
        return info

    else:
        return space_group
