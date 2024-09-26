#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
====================================
samosa/symmetry/point_group_utils.py
====================================

This program defines methods for defining and manipulating 3D point groups.
"""

import numpy as np

from samosa.symmetry.group_utils import Group

from samosa.symmetry.representations import MatrixGroupElement

from samosa.symmetry.operations_3d import operator_C, operator_S, operator_M

from samosa.utils.type_checks import ArrayType, not_None, check_type

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
            raise TypeError("Basis must consist of 3 vectors of length 3, "
                           f"but has dimensions {basis.shape}.")

        if basis.shape[0] != 3 or basis.shape[1] != 3:
            raise TypeError("Basis must consist of 3 vectors of length 3, "
                           f"but has dimensions {basis.shape}.")
        

    # Test if the point group is polyhedral
    if pg_type == 'T':
        ## Tetrahedral point groups
        if len(pg_symbol) == 1:
            return _T_group(basis)
        
        elif len(pg_symbol) == 2 and pg_symbol[1] == 'd':
            return _Td_group(basis)
        
        elif len(pg_symbol) == 2 and pg_symbol[1] == 'h':
            return _Th_group(basis)
        
        else:
            raise symbol_error

    elif pg_type == 'O':
        ## Octahedral point groups
        if len(pg_symbol) == 1:
            return _O_group(basis)
        
        elif len(pg_symbol) == 2 and pg_symbol[1] == 'h':
            return _Oh_group(basis)
        
        else:
            raise symbol_error

    elif pg_type == 'I':
        ## Icosahedral point groups
        if len(pg_symbol) == 1:
            return _I_group(basis)
        
        elif len(pg_symbol) == 2 and pg_symbol[1] == 'h':
            return _Ih_group(basis)
        
        else:
            raise symbol_error


    # Test if the point group is axial
    else:
        if len(pg_symbol) == 1:
            raise symbol_error

        ## Convert Cs and Ci symbols to C1h and S2 respectively
        if pg_symbol == 'Cs':
            pg_symbol = 'C1h'

        elif pg_symbol == 'Ci':
            pg_symbol = 'S2'

        ## If given, assign the rotation axes
        if len(axes) == 0:
            axes = None
        
        else:
            axes = [a for a in axes]
    
        ## Define the order of the primary rotation
        n_str = pg_symbol.lstrip('CSD').rstrip('vdh') 
        try:
            n = int(n_str)
            if n < 1:
                raise Exception("n is smaller than 1!")
        except:
            raise symbol_error

        ## Determine the secondary point group characteristic
        pg_symbol_tail = pg_symbol[(len(n_str)+1):]

        if pg_type == 'C':
            ## Cyclic proper rotation point groups
            if pg_symbol_tail == '':
                return _Cn_group(n, axes, basis)
            
            elif pg_symbol_tail == 'v':
                return _Cnv_group(n, axes, basis)

            elif pg_symbol_tail == 'h':
                return _Cnh_group(n, axes, basis)

            else:
                raise symbol_error
            
        elif pg_type == 'S':
            ## Cyclic improper rotation point groups
            if pg_symbol_tail == '':
                return _Sn_group(n, axes, basis)

            else:
                raise symbol_error

        elif pg_type == 'D':
            ## Dihedral point groups
            if pg_symbol_tail == '':
                return _Dn_group(n, axes, basis)
            
            elif pg_symbol_tail == 'd':
                return _Dnd_group(n, axes, basis)
            
            elif pg_symbol_tail == 'h':
                return _Dnh_group(n, axes, basis)
            
            else:
                raise symbol_error

        else:
            raise symbol_error

"""
-------------------------------------------------------------------------------
3D point groups
-------------------------------------------------------------------------------
"""
# Polyhedral point groups

def _T_group(basis=None):
    """
    Tetrahedral rotational point group. 

    Selected generators are C3,[111], and C2,[001].
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [1, 1, 1]
    C3 = MatrixGroupElement.input_args(operator_C(axis_1, 3, basis),
                                       cycle_order = 3,
                                       operator_inverse = \
                                                operator_C(axis_1, -3, basis))

    axis_2 = [0, 0, 1]
    C2 = MatrixGroupElement.input_args(operator_C(axis_2, 2, basis),
                                       cycle_order = 2,
                                       operator_inverse = \
                                                operator_C(axis_2, 2, basis))
    
    generators = [C3, C2]

    return Group(generators, name='T', order=12) 


def _Td_group(basis=None):
    """
    Tetrahedral point group.

    Selected generators are S4,[001], and S4,[100].
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [0, 0, 1]
    S4_1 = MatrixGroupElement.input_args(operator_S(axis_1, 4, basis),
                                         cycle_order = 4,
                                         operator_inverse = \
                                                operator_S(axis_1, -4, basis))

    axis_2 = [1, 0, 0]
    S4_2 = MatrixGroupElement.input_args(operator_S(axis_2, 4, basis),
                                         cycle_order = 4,
                                         operator_inverse = \
                                                operator_S(axis_2, -4, basis))
    
    generators = [S4_1, S4_2]

    return Group(generators, name='Td', order=24) 


def _Th_group(basis=None):
    """
    Tetrahedral rotation-inversion point group.

    Selected generators are C3,[111], and Mh,[001]. 
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [1, 1, 1]
    C3 = MatrixGroupElement.input_args(operator_C(axis_1, 3, basis),
                                         cycle_order = 3,
                                         operator_inverse = \
                                                operator_C(axis_1, -3, basis))

    axis_2 = [0, 0, 1]
    Mh = MatrixGroupElement.input_args(operator_M(axis_2, basis),
                                       cycle_order = 2,
                                       operator_inverse = \
                                                operator_M(axis_2, basis))
    
    generators = [C3, Mh]

    return Group(generators, name='Th', order=24) 


def _O_group(basis=None):
    """
    Octahedral rotational point group. 

    Selected generators are C4,[001], and C4,[100].
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [0, 0, 1]
    C4_1 = MatrixGroupElement.input_args(operator_C(axis_1, 4, basis),
                                         cycle_order = 4,
                                         operator_inverse = \
                                                operator_C(axis_1, -4, basis))

    axis_2 = [1, 0, 0]
    C4_2 = MatrixGroupElement.input_args(operator_C(axis_2, 4, basis),
                                         cycle_order = 4,
                                         operator_inverse = \
                                                operator_C(axis_2, -4, basis))
    
    generators = [C4_1, C4_2]

    return Group(generators, name='O', order=24) 


def _Oh_group(basis=None):
    """
    Octahedral point group. 

    Selected generators are S6,[111], and S4,[001].
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [1, 1, 1]
    S6 = MatrixGroupElement.input_args(operator_S(axis_1, 6, basis),
                                       cycle_order = 6,
                                       operator_inverse = \
                                                operator_S(axis_1, -6, basis))

    axis_2 = [0, 0, 1]
    S4 = MatrixGroupElement.input_args(operator_S(axis_2, 4, basis),
                                       cycle_order = 4,
                                       operator_inverse = \
                                                operator_S(axis_2, -4, basis))
    
    generators = [S6, S4]

    return Group(generators, name='Oh', order=48) 


def _I_group(basis=None):
    """
    Icosahedral rotational point group. 

    Selected generators are C5,[01p], and C5,[p01], where

    p = (1 + sqrt(5))/2,
    
    is the golden ratio.
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    phi = (1 + np.sqrt(5))/2
    axis_1 = [0, 1, phi]
    C5_1 = MatrixGroupElement.input_args(operator_C(axis_1, 5, basis),
                                         cycle_order = 5,
                                         operator_inverse = \
                                                operator_C(axis_1, -5, basis))

    axis_2 = [phi, 0, 1]
    C5_2 = MatrixGroupElement.input_args(operator_C(axis_2, 5, basis),
                                         cycle_order = 5,
                                         operator_inverse = \
                                                operator_C(axis_2, -5, basis))
    
    generators = [C5_1, C5_2]

    return Group(generators, name='I', order=60) 


def _Ih_group(basis=None):
    """
    Icosahedral point group. 

    Selected generators are S10,[01p], and S10,[p01], where

    p = (1 + sqrt(5))/2,
    
    is the golden ratio.
    
    Arguments:
    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    phi = (1 + np.sqrt(5))/2
    axis_1 = [0, 1, phi]
    S10_1 = MatrixGroupElement.input_args(operator_S(axis_1, 10, basis),
                                         cycle_order = 10,
                                         operator_inverse = \
                                                operator_S(axis_1, -10, basis))

    axis_2 = [phi, 0, 1]
    S10_2 = MatrixGroupElement.input_args(operator_S(axis_2, 10, basis),
                                          cycle_order = 10,
                                          operator_inverse = \
                                                operator_S(axis_2, -10, basis))
    
    generators = [S10_1, S10_2]

    return Group(generators, name='Ih', order=120) 


# Axial point groups

def _Cn_group(n, axis = None, basis=None):
    """
    Cyclic group of n-fold rotations.

    Arguments:
    n             - int, cycle order of the rotations; 

    axis          - ArrayType, (default=None), defines the primary
                    rotation axis. If None, assumes [0, 0, 1]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Assign primary axis
    if not_None(axis):
        
        if len(axis) != 1:
            raise Exception(f"Only one axis is required for "
                            f"Cn point groups ({len(axis)} provided).")

        else:    
            axis = axis[0]

    else:
        ### By default, choose z-axis as the primary rotation axis
        axis = [0, 0, 1]

    ## Define generators
    Cn = MatrixGroupElement.input_args(operator_C(axis, n, basis),
                                         cycle_order = n,
                                         operator_inverse = \
                                                operator_C(axis, -n, basis))

    generators = [Cn]
    name = f'C{n}'
    order = n

    return Group(generators, name=name, order=order) 


def _Cnv_group(n, axes = None, basis=None):
    """
    Cyclic group of n-fold rotations and reflections with mirror planes 
    parallel to the n-fold axis.

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 

    axes          - ArrayType, (default=None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0, 0, 1] and [0, 1, 0]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
    
    ## Check that n > 1
    if n == 1:
        raise ValueError("C1v point group is not defined!")
    
    ## Assign primary axes
    if not_None(axes):
        if len(axes) != 2:
            raise Exception(f"Two axes are required for "
                            f"Cnv point groups ({len(axes)} provided).")
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the mirror plane normal
        axis_1 = [0, 0, 1]
        axis_2 = [0, 1, 0]

    ## Define generators
    Cn = MatrixGroupElement.input_args(operator_C(axis_1, n, basis),
                                         cycle_order = n,
                                         operator_inverse = \
                                                operator_C(axis_1, -n, basis))

    Mv = MatrixGroupElement.input_args(operator_M(axis_2, basis),
                                       cycle_order = 2,
                                       operator_inverse = \
                                                operator_M(axis_2, basis))

    generators = [Cn, Mv]
    name = f'C{n}v'
    order = 2*n

    return Group(generators, name=name, order=order) 


def _Cnh_group(n, axis = None, basis=None):
    """
    Cyclic group of n-fold rotations and a reflection with mirror plane 
    perpendicular to the n-fold axis.

    Arguments:
    n             - int, cycle order of the rotations; 

    axis          - ArrayType, (default=None), defines the primary
                    rotation axis. If None, assumes [0, 0, 1]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
    
    ## Assign primary axis
    if not_None(axis):
        
        if len(axis) != 1:
            raise Exception(f"Only one axis is required for "
                            f"Cnh point groups ({len(axes)} provided).")

        else:    
            axis = axis[0]

    else:
        # By default, choose z-axis as the primary rotation axis
        axis = [0, 0, 1]
    
    ## Define generators
    Cn = MatrixGroupElement.input_args(operator_C(axis, n, basis),
                                       cycle_order = n,
                                       operator_inverse = \
                                                operator_C(axis, -n, basis))

    Mh = MatrixGroupElement.input_args(operator_M(axis, basis),
                                       cycle_order = 2,
                                       operator_inverse = \
                                                operator_M(axis, basis))

    generators = [Cn, Mh]
    name = f'C{n}h'
    order = 2*n

    return Group(generators, name=name, order=order) 


def _Sn_group(n, axis = None, basis=None):
    """
    Group of n-fold improper rotations.

    Arguments:
    n             - int, cycle order of the improper rotations, must be an even
                    integer; 

    axis          - ArrayType, (default=None), defines the primary
                    rotation axis. If None, assumes [0, 0, 1]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Check that n is even
    if n%2 != 0:
        raise ValueError(f"Sn point group requires n to be an even interger!\n"
                         f"n = {n}")
    
    ## Assign primary axis
    if not_None(axis):
        
        if len(axis) != 1:
            raise Exception(f"Only one axis is required for "
                            f"Sn point groups ({len(axes)} provided).")

        else:    
            axis = axis[0]

    else:
        # By default, choose z-axis as the primary rotation axis
        axis = [0, 0, 1]

    ## Define generators
    Sn = MatrixGroupElement.input_args(operator_S(axis, n, basis),
                                       cycle_order = n,
                                       operator_inverse = \
                                                operator_S(axis, -n, basis))

    generators = [Sn]
    name = f'S{n}'
    order = n

    return Group(generators, name=name, order=order) 


def _Dn_group(n, axes = None, basis=None):
    """
    Dihedral group of order n.

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 

    axes          - ArrayType, (default=None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0, 0, 1] and [0, 1, 0]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
    
    ## Check that n > 1
    if n == 1:
        raise ValueError("D1 point group is not defined!")
    
    ## Assign primary axes
    if not_None(axes):
        if len(axes) != 2:
            raise Exception(f"Two axes are required for "
                            f"Dn point groups ({len(axes)} provided).")
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the C2 axis
        axis_1 = [0, 0, 1]
        axis_2 = [0, 1, 0]

    ## Define generators
    Cn = MatrixGroupElement.input_args(operator_C(axis_1, n, basis),
                                         cycle_order = n,
                                         operator_inverse = \
                                                operator_C(axis_1, -n, basis))

    C2 = MatrixGroupElement.input_args(operator_C(axis_2, 2, basis),
                                         cycle_order = 2,
                                         operator_inverse = \
                                                operator_C(axis_2, -2, basis))

    generators = [Cn, C2]
    name = f'D{n}'
    order = 2*n

    return Group(generators, name=name, order=order) 


def _Dnd_group(n, axes = None, basis=None):
    """
    Dihedral group of order n with dihedral mirror reflections (bipyramidal
    symmetry).

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 

    axes          - ArrayType, (default=None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0, 0, 1] and [0, 1, 0]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
    
    ## Check that n > 1
    if n == 1:
        raise ValueError("D1d point group is not defined!")
    
    ## Assign primary axes
    if not_None(axes):
        if len(axes) != 2:
            raise Exception(f"Two axes are required for "
                            f"Dnd point groups ({len(axes)} provided).")
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the Mv axis
        axis_1 = [0, 0, 1]
        axis_2 = [0, 1, 0]

    ## Define generators
    S2n = MatrixGroupElement.input_args(operator_S(axis_1, 2*n, basis),
                                         cycle_order = 2*n,
                                         operator_inverse = \
                                                operator_S(axis_1, -2*n, basis))

    Mv = MatrixGroupElement.input_args(operator_M(axis_2, basis),
                                       cycle_order = 2,
                                       operator_inverse = \
                                                operator_M(axis_2, basis))

    generators = [S2n, Mv]
    name = f'D{n}d'
    order = 4*n

    return Group(generators, name=name, order=order) 


def _Dnh_group(n, axes = None, basis=None):
    """
    Dihedral group of order n with inversion symmetry (n-gon prism symmetry).

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 

    axes          - ArrayType, (default=None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0, 0, 1] and [0, 1, 0]; 

    basis         - ArrayType (default=None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
    
    ## Check that n > 1
    if n == 1:
        raise ValueError("D1h point group is not defined!")
    
    ## Assign primary axes
    if not_None(axes):
        if len(axes) != 2:
            raise Exception(f"Two axes are required for "
                            f"Dnh point groups ({len(axes)} provided).")
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the Mv axis
        axis_1 = [0, 0, 1]
        axis_2 = [0, 1, 0]

    ## Define generators
    Cn = MatrixGroupElement.input_args(operator_C(axis_1, n, basis),
                                         cycle_order = n,
                                         operator_inverse = \
                                                operator_C(axis_1, -n, basis))

    Mv = MatrixGroupElement.input_args(operator_M(axis_2, basis),
                                         cycle_order = 2,
                                         operator_inverse = \
                                                operator_M(axis_2, basis))

    Mh = MatrixGroupElement.input_args(operator_M(axis_1, basis),
                                       cycle_order = 2,
                                       operator_inverse = \
                                                operator_M(axis_1, basis))

    generators = [Cn, Mv, Mh]
    name = f'D{n}h'
    order = 4*n

    return Group(generators, name=name, order=order) 
