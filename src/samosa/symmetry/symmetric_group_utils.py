#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
========================
symmetric_group_utils.py
========================

This program defines methods for defining and manipulating symmetric
permutation groups.
"""

import numpy as np

from samosa.symmetry.group_utils import Group
from samosa.symmetry.representations import PermutationGroupElement

from samosa.api.api_utils import not_None, check_type, check_len, ArrayType

def symmetric_group(n, as_matrices=False):
    """
    Returns the symmetric group of order n! given index n.   
    
    Arguments:
    n           - int, index of the symmetric group;

    as_matrices - bool (default=False), if True, initializes generators as
                  matrices, if False, initializes generators as permutation
                  cycles.

    Returns: 
    Group object - symmetric group of order n!.
    """

    # Type checks
    check_type('symmetric group index', n, int, np.int64)
    check_type('as_matrices', as_matrices, bool)

    # Initialize generators of the symmetric group
    g1 = tuple(i%n+1 for i in range(1, n+1))

    # For n <= 2, there is only one generator 
    if n > 2: 
        g2 = (2, 1) + tuple(i for i in range(3, n+1))
    else:
        g2 = tuple(i for i in range(1, n+1))

    G1 = PermutationGroupElement(g1)
    G2 = PermutationGroupElement(g2)

    if as_matrices:
        G1 = G1.permutation_matrix
        G2 = G2.permutation_matrix

    if n > 2:
        generators = [G1, G2]
    else:
        generators = [G1]
    
    return Group(generators, f'S{n}')
