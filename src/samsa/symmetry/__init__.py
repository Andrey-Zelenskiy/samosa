#! /usr/bin/env python3
# Andrey Zelenskiy, 2024-2026

"""
==============
samsa/symmetry
==============

This submodule provides methods for initializing most common types of groups:
point groups, space groups, and symmetric permutation groups.
"""

import numpy as np

from samsa.symmetry.group import Group

from samsa.symmetry.representations import (
    SpaceGroupElement,
    PermutationGroupElement,
)

from samsa.symmetry.point_groups import (
    _T_group,
    _Td_group,
    _Th_group,
    _O_group,
    _Oh_group,
    _I_group,
    _Ih_group,
    _Cn_group,
    _Cnv_group,
    _Cnh_group,
    _Sn_group,
    _Dn_group,
    _Dnd_group,
    _Dnh_group,
)

from samsa.database import SpaceGroupDatabase

from samsa.utils.math import _normalize_vector

from samsa.utils.type_checks import (
    is_None,
    not_None,
    NoneType,
    check_type,
)


def point_group(pg_symbol, *axes, basis=None):
    """
    Returns the 3D point group given its Schoenflies symbol.

    Arguments:
    pg_symbol     - str, Schoenflies symbol of the point group;

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth
                    basis vector. If None, assumes Cartesian basis;

    axes          - ArrayType, in the case of axial groups, provides the
                    primary and secondary transformation axes:
                    if a single axis is given, it defines the primary rotation
                    axis;
                    if two axes are given, the first one specifies the primary
                    rotation axis, while the second one gives the orientation
                    of the secondary rotation axis or the normal of the mirror
                    plane.

    Returns:
    group, point group object.
    """

    symbol_error = ValueError(f"Incorrect Schoenflies symbol: {pg_symbol}.")

    pg_type = pg_symbol[0]

    # Setup the coordinate system

    if not_None(basis):

        basis = [_normalize_vector(b) for b in basis]
        basis = np.array(basis)

        if len(basis.shape) != 2:
            raise TypeError(
                "Basis must consist of 3 vectors of length 3, "
                f"but has dimensions {basis.shape}."
            )

        if basis.shape[0] != 3 or basis.shape[1] != 3:
            raise TypeError(
                "Basis must consist of 3 vectors of length 3, "
                f"but has dimensions {basis.shape}."
            )

    # Test if the point group is polyhedral
    if pg_type == "T":
        # Tetrahedral point groups
        if len(pg_symbol) == 1:
            return _T_group(basis)

        elif len(pg_symbol) == 2 and pg_symbol[1] == "d":
            return _Td_group(basis)

        elif len(pg_symbol) == 2 and pg_symbol[1] == "h":
            return _Th_group(basis)

        else:
            raise symbol_error

    elif pg_type == "O":
        # Octahedral point groups
        if len(pg_symbol) == 1:
            return _O_group(basis)

        elif len(pg_symbol) == 2 and pg_symbol[1] == "h":
            return _Oh_group(basis)

        else:
            raise symbol_error

    elif pg_type == "I":
        # Icosahedral point groups
        if len(pg_symbol) == 1:
            return _I_group(basis)

        elif len(pg_symbol) == 2 and pg_symbol[1] == "h":
            return _Ih_group(basis)

        else:
            raise symbol_error

    # Test if the point group is axial
    else:
        if len(pg_symbol) == 1:
            raise symbol_error

        # Convert Cs and Ci symbols to C1h and S2 respectively
        if pg_symbol == "Cs":
            pg_symbol = "C1h"

        elif pg_symbol == "Ci":
            pg_symbol = "S2"

        # If given, assign the rotation axes
        if len(axes) == 0:
            axes = None

        else:
            axes = [a for a in axes]

        # Define the order of the primary rotation
        n_str = pg_symbol.lstrip("CSD").rstrip("vdh")
        try:
            n = int(n_str)
            if n < 1:
                raise Exception("n is smaller than 1!")
        except BaseException:
            raise symbol_error

        # Determine the secondary point group characteristic
        pg_symbol_tail = pg_symbol[(len(n_str) + 1) :]

        if pg_type == "C":
            # Cyclic proper rotation point groups
            if pg_symbol_tail == "":
                return _Cn_group(n, axes, basis)

            elif pg_symbol_tail == "v":
                return _Cnv_group(n, axes, basis)

            elif pg_symbol_tail == "h":
                return _Cnh_group(n, axes, basis)

            else:
                raise symbol_error

        elif pg_type == "S":
            # Cyclic improper rotation point groups
            if pg_symbol_tail == "":
                return _Sn_group(n, axes, basis)

            else:
                raise symbol_error

        elif pg_type == "D":
            # Dihedral point groups
            if pg_symbol_tail == "":
                return _Dn_group(n, axes, basis)

            elif pg_symbol_tail == "d":
                return _Dnd_group(n, axes, basis)

            elif pg_symbol_tail == "h":
                return _Dnh_group(n, axes, basis)

            else:
                raise symbol_error

        else:
            raise symbol_error


def space_group(dimension, sg_index, database=None, return_info=False):
    """
    Returns a space group given spatial dimension, as well as the space group
    number.

    Arguments:
    dimension   - int, dimension of the space (lattice);

    sg_index    - int, number of the space group according to the International
                  Tables for Crystallography, vol. A;

    database    - SpaceGroupDatabase object, (default=None) lookup data
                  structure;

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
    # Type checks
    check_type("dimension", dimension, int)
    check_type("sg_index", sg_index, int)
    check_type("database", database, SpaceGroupDatabase, NoneType)

    if is_None(database):
        database = SpaceGroupDatabase()

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

    space_group = Group(
        generators, name=str(sg_index), order=order, filter_generators=True
    )

    if return_info:
        info = {
            "space_group": space_group,
            "point_group_symbol": point_group_symbol,
            "lattice_type": lattice_type,
            "symmorphic": symmorphic,
        }
        return info

    else:
        return space_group


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
    check_type("symmetric group index", n, int, np.int64)
    check_type("as_matrices", as_matrices, bool)

    # Initialize generators of the symmetric group
    g1 = tuple(i % n + 1 for i in range(1, n + 1))

    # For n <= 2, there is only one generator
    if n > 2:
        g2 = (2, 1) + tuple(i for i in range(3, n + 1))
    else:
        g2 = tuple(i for i in range(1, n + 1))

    G1 = PermutationGroupElement(g1)
    G2 = PermutationGroupElement(g2)

    if as_matrices:
        G1 = G1.permutation_matrix
        G2 = G2.permutation_matrix

    if n > 2:
        generators = [G1, G2]
    else:
        generators = [G1]

    return Group(generators, f"S{n}")
