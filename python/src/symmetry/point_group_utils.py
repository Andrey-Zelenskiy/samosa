#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
====================
point_group_utils.py
====================

This program defines methods for defining and manipulating 3D point groups.
"""

import numpy as np
from symmetry_utils import group, matrix_group_element, not_None

def point_group(pg_symbol, store_inverse = True, basis = None, *axes):
    """
    Returns the 3D point group given its Schoenflies symbol.   
    
    Arguments:
    pg_symbol     - str, Schoenflies symbol of the point group;
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis;
    axes          - array_type, in the case of axial groups, provides the 
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

    symbol_error = ValueError("Incorrect Schoenflies symbol: " + pg_symbol) 

    pg_type = pg_symbol[0]


    # Setup the coordinate system

    if not_None(basis):

        basis = [_normalize_vector(b) for b in basis]
        basis = np.array(basis)

        if len(basis.shape) != 2:
            raise TypeError("Basis must consist of 3 vectors of length 3, "\
                            "but has dimensions {}".format(basis.shape))

        if basis.shape[0] != 3 or basis.shape[1] != 3:
            raise TypeError("Basis must consist of 3 vectors of length 3, "\
                            "but has dimensions {}".format(basis.shape))
        

    # Test if the point group is polyhedral
    if pg_type == 'T':
        ## Tetrahedral point groups
        if len(pg_symbol) == 1:
            return _T_group(store_inverse,basis)
        
        elif len(pg_symbol) == 2 and pg_type[1] == 'd':
            return _Td_group(store_inverse,basis)
        
        elif len(pg_symbol) == 2 and pg_type[1] == 'h':
            return _Th_group(store_inverse,basis)
        
        else:
            raise symbol_error

    elif pg_type == 'O':
        ## Octahedral point groups
        if len(pg_symbol) == 1:
            return _O_group(store_inverse,basis)
        
        elif len(pg_symbol) == 2 and pg_type[1] == 'h':
            return _Oh_group(store_inverse,basis)
        
        else:
            raise symbol_error

    elif pg_type == 'I':
        ## Icosahedral point groups
        if len(pg_symbol) == 1:
            return _I_group(store_inverse,basis)
        
        elif len(pg_symbol) == 2 and pg_type[1] == 'h':
            return _Ih_group(store_inverse,basis)
        
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
                return _Cn_group(n,axes,store_inverse,basis)
            
            elif pg_symbol_tail == 'v':
                return _Cnv_group(n,axes,store_inverse,basis)

            elif pg_symbol_tail == 'h':
                return _Cnh_group(n,axes,store_inverse,basis)

            else:
                raise symbol_error
            
        elif pg_type == 'S':
            ## Cyclic improper rotation point groups
            if pg_symbol_tail == '':
                return _Sn_group(n,axes,store_inverse,basis)

            else:
                raise symbol_error

        elif pg_type == 'D':
            ## Dihedral point groups
            if pg_symbol_tail == '':
                return _Dn_group(n,axes,store_inverse,basis)
            
            elif pg_symbol_tail == 'd':
                return _Dnd_group(n,axes,store_inverse,basis)
            
            elif pg_symbol_tail == 'h':
                return _Dnh_group(n,axes,store_inverse,basis)
            
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

def _T_group(store_inverse = True, basis = None):
    """
    Tetrahedral rotational point group. 

    Selected generators are C3,[111], and C2,[001].
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [1,1,1]
    C3 = matrix_group_element(operator_C(axis_1,3,basis),
                              operator_inv = operator_C(axis_1,-3,basis),
                              cycle_order = 3,
                              store_inverse = store_inverse)

    axis_2 = [0,0,1]
    C2 = matrix_group_element(operator_C(axis_2,2,basis),
                              operator_inv = operator_C(axis_2,2,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)
    
    generators = [C3,C2]

    return group(generators) 


def _Td_group(store_inverse = True, basis = None):
    """
    Tetrahedral point group.

    Selected generators are S4,[001], and S4,[100].
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [0,0,1]
    S4_1 = matrix_group_element(operator_S(axis_1,4,basis),
                                operator_inv = operator_S(axis_1,-4,basis),
                                cycle_order = 4,
                                store_inverse = store_inverse)

    axis_2 = [1,0,0]
    S4_2 = matrix_group_element(operator_S(axis_2,4,basis),
                                operator_inv = operator_S(axis_2,-4,basis),
                                cycle_order = 4,
                                store_inverse = store_inverse)
    
    generators = [S4_1,S4_2]

    return group(generators) 


def _Th_group(store_inverse = True, basis = None):
    """
    Tetrahedral rotation-inversion point group.

    Selected generators are C3,[111], and Mh,[001]. 
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [1,1,1]
    C3 = matrix_group_element(operator_C(axis_1,3,basis),
                              operator_inv = operator_S(axis_1,-3,basis),
                              cycle_order = 3,
                              store_inverse = store_inverse)

    axis_2 = [0,0,1]
    Mh = matrix_group_element(operator_M(axis_2,basis),
                              operator_inv = operator_M(axis_2,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)
    
    generators = [C3,Mh]

    return group(generators) 


def _O_group(store_inverse = True, basis = None):
    """
    Octahedral rotational point group. 

    Selected generators are C4,[001], and C4,[100].
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [0,0,1]
    C4_1 = matrix_group_element(operator_C(axis_1,4,basis),
                                operator_inv = operator_C(axis_1,-4,basis),
                                cycle_order = 4,
                                store_inverse = store_inverse)

    axis_2 = [1,0,0]
    C4_2 = matrix_group_element(operator_C(axis_2,4,basis),
                                operator_inv = operator_C(axis_2,4,basis),
                                cycle_order = 4,
                                store_inverse = store_inverse)
    
    generators = [C4_1,C4_2]

    return group(generators) 


def _Oh_group(store_inverse = True, basis = None):
    """
    Octahedral point group. 

    Selected generators are S6,[111], and S4,[001].
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    axis_1 = [1,1,1]
    S6 = matrix_group_element(operator_S(axis_1,6,basis),
                              operator_inv = operator_S(axis_1,-6,basis),
                              cycle_order = 6,
                              store_inverse = store_inverse)

    axis_2 = [0,0,1]
    S4 = matrix_group_element(operator_S(axis_2,4,basis),
                              operator_inv = operator_S(axis_2,4,basis),
                              cycle_order = 4,
                              store_inverse = store_inverse)
    
    generators = [S6,S4]

    return group(generators) 


def _I_group(store_inverse = True, basis = None):
    """
    Icosahedral rotational point group. 

    Selected generators are C5,[01p], and C5,[p01], where

    p = (1 + sqrt(5))/2,
    
    is the golden ratio.
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    phi = (1 + np.sqrt(5))/2
    axis_1 = [0,1,phi]
    C5_1 = matrix_group_element(operator_C(axis_1,5,basis),
                                operator_inv = operator_C(axis_1,-5,basis),
                                cycle_order = 5,
                                store_inverse = store_inverse)

    axis_2 = [phi,0,1]
    C5_2 = matrix_group_element(operator_C(axis_2,5,basis),
                                operator_inv = operator_C(axis_2,5,basis),
                                cycle_order = 5,
                                store_inverse = store_inverse)
    
    generators = [C5_1,C5_2]

    return group(generators) 


def _Ih_group(store_inverse = True, basis = None):
    """
    Icosahedral point group. 

    Selected generators are S10,[01p], and S10,[p01], where

    p = (1 + sqrt(5))/2,
    
    is the golden ratio.
    
    Arguments:
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Define generators
    phi = (1 + np.sqrt(5))/2
    axis_1 = [0,1,phi]
    S10_1 = matrix_group_element(operator_S(axis_1,10,basis),
                                 operator_inv = operator_S(axis_1,-10,basis),
                                 cycle_order = 10,
                                 store_inverse = store_inverse)

    axis_2 = [phi,0,1]
    S10_2 = matrix_group_element(operator_S(axis_2,10,basis),
                                 operator_inv = operator_S(axis_2,10,basis),
                                 cycle_order = 10,
                                 store_inverse = store_inverse)
    
    generators = [S10_1,S10_2]

    return group(generators) 


# Axial point groups

def _Cn_group(n, axis = None, store_inverse = True, basis = None):
    """
    Cyclic group of n-fold rotations.

    Arguments:
    n             - int, cycle order of the rotations; 
    axis          - array_type, (optional, default = None), defines the primary
                    rotation axis. If None, assumes [0,0,1]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Assign primary axis
    if not_None(axis):
        
        if len(axis) != 1:
            raise Exception("Only one axis is required for "\
                            "Cn point groups ({} provided).".format(len(axis)))

        else:    
            axis = axis[0]

    else:
        ### By default, choose z-axis as the primary rotation axis
        axis = [0,0,1]

    ## Define generators
    Cn = matrix_group_element(operator_C(axis,n,basis),
                              operator_inv = operator_C(axis,-n,basis),
                              cycle_order = n,
                              store_inverse = store_inverse)

    generators = [Cn]

    return group(generators) 


def _Cnv_group(n, axes = None, store_inverse = True, basis = None):
    """
    Cyclic group of n-fold rotations and reflections with mirror planes 
    parallel to the n-fold axis.

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 
    axes          - array_type, (optional, default = None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0,0,1] and [0,1,0]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
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
            raise Exception("Two axes are required for "\
                            "Cnv point groups ({} provided).".format(len(axes)))
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the mirror plane normal
        axis_1 = [0,0,1]
        axis_2 = [0,1,0]

    ## Define generators
    Cn = matrix_group_element(operator_C(axis_1,n,basis),
                              operator_inv = operator_C(axis_1,-n,basis),
                              cycle_order = n,
                              store_inverse = store_inverse)

    Mv = matrix_group_element(operator_M(axis_2,basis),
                              operator_inv = operator_M(axis_2,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)

    generators = [Cn, Mv]

    return group(generators) 


def _Cnh_group(n, axis = None, store_inverse = True, basis = None):
    """
    Cyclic group of n-fold rotations and a reflection with mirror plane 
    perpendicular to the n-fold axis.

    Arguments:
    n             - int, cycle order of the rotations; 
    axis          - array_type, (optional, default = None), defines the primary
                    rotation axis. If None, assumes [0,0,1]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
    
    ## Assign primary axis
    if not_None(axis):
        
        if len(axis) != 1:
            raise Exception("Only one axis is required for "\
                            "Cnh point groups ({} provided).".format(len(axis)))

        else:    
            axis = axis[0]

    else:
        # By default, choose z-axis as the primary rotation axis
        axis = [0,0,1]
    
    ## Define generators
    Cn = matrix_group_element(operator_C(axis,n,basis),
                              operator_inv = operator_C(axis,-n,basis),
                              cycle_order = n,
                              store_inverse = store_inverse)

    Mh = matrix_group_element(operator_M(axis,basis),
                              operator_inv = operator_M(axis,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)

    generators = [Cn, Mh]

    return group(generators) 


def _Sn_group(n, axis = None, store_inverse = True, basis = None):
    """
    Group of n-fold improper rotations.

    Arguments:
    n             - int, cycle order of the improper rotations, must be an even
                    integer; 
    axis          - array_type, (optional, default = None), defines the primary
                    rotation axis. If None, assumes [0,0,1]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
                    the basis of the transformations, assuming basis[n] = nth 
                    basis vector. If None, assumes Cartesian basis.

    Returns: 
    group, point group object.
    """
   
    ## Check that n is even
    if n%2 != 0:
        raise ValueError("Sn point group requires n to be an even interger!\n"
                         "n = {}".format(n))
    
    ## Assign primary axis
    if not_None(axis):
        
        if len(axis) != 1:
            raise Exception("Only one axis is required for "\
                            "Sn point groups ({} provided).".format(len(axis)))

        else:    
            axis = axis[0]

    else:
        # By default, choose z-axis as the primary rotation axis
        axis = [0,0,1]

    ## Define generators
    Sn = matrix_group_element(operator_S(axis,n,basis),
                              operator_inv = operator_S(axis,-n,basis),
                              cycle_order = n,
                              store_inverse = store_inverse)

    generators = [Sn]

    return group(generators) 


def _Dn_group(n, axes = None, store_inverse = True, basis = None):
    """
    Dihedral group of order n.

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 
    axes          - array_type, (optional, default = None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0,0,1] and [0,1,0]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
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
            raise Exception("Two axes are required for "\
                            "Dn point groups ({} provided).".format(len(axes)))
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the C2 axis
        axis_1 = [0,0,1]
        axis_2 = [0,1,0]

    ## Define generators
    Cn = matrix_group_element(operator_C(axis_1,n,basis),
                              operator_inv = operator_C(axis_1,-n,basis),
                              cycle_order = n,
                              store_inverse = store_inverse)

    C2 = matrix_group_element(operator_C(axis_2,2,basis),
                              operator_inv = operator_C(axis_2,2,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)

    generators = [Cn, C2]

    return group(generators) 


def _Dnd_group(n, axes = None, store_inverse = True, basis = None):
    """
    Dihedral group of order n with dihedral mirror reflections (bipyramidal
    symmetry).

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 
    axes          - array_type, (optional, default = None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0,0,1] and [0,1,0]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
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
            raise Exception("Two axes are required for "\
                            "Dnd point groups ({} provided).".format(len(axes)))
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the Mv axis
        axis_1 = [0,0,1]
        axis_2 = [0,1,0]

    ## Define generators
    S2n = matrix_group_element(operator_S(axis_1,2*n,basis),
                               operator_inv = operator_S(axis_1,-2*n,basis),
                               cycle_order = 2*n,
                               store_inverse = store_inverse)

    Mv = matrix_group_element(operator_M(axis_2,basis),
                              operator_inv = operator_M(axis_2,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)

    generators = [S2n, Mv]

    return group(generators) 


def _Dnh_group(n, axes = None, store_inverse = True, basis = None):
    """
    Dihedral group of order n with inversion symmetry (n-gon prism symmetry).

    Arguments:
    n             - int, cycle order of the rotations, must be larger than 1; 
    axes          - array_type, (optional, default = None), defines the primary
                    rotation axis and the direction of the mirror plane normal.
                    If None, assumes [0,0,1] and [0,1,0]; 
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses;
    basis         - array_type (optional, default None), if not None, specifies
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
            raise Exception("Two axes are required for "\
                            "Dnh point groups ({} provided).".format(len(axes)))
    
        else:
            axis_1 = axes[0]
            axis_2 = axes[1]

    else:
        # By default, choose z-axis as the primary rotation axis, and y-axis as
        # the Mv axis
        axis_1 = [0,0,1]
        axis_2 = [0,1,0]

    ## Define generators
    Cn = matrix_group_element(operator_C(axis_1,n,basis),
                              operator_inv = operator_C(axis_1,-n,basis),
                              cycle_order = n,
                              store_inverse = store_inverse)

    Mv = matrix_group_element(operator_M(axis_2,basis),
                              operator_inv = operator_M(axis_2,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)

    Mh = matrix_group_element(operator_M(axis_1,basis),
                              operator_inv = operator_M(axis_1,basis),
                              cycle_order = 2,
                              store_inverse = store_inverse)

    generators = [Cn, Mv, Mh]

    return group(generators) 


"""
-------------------------------------------------------------------------------
Symmetry operators in 3D
-------------------------------------------------------------------------------
"""

def operator_C(axis, n, basis = None):
    """
    Defines a proper 3D rotation.

    The general expression for a proper rotation around a 3D axis is found in
    (https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)

    Arguments:
    axis  - np.1darray[3], rotation axis, not necesserally normalized;
    n     - int, defines the angle of rotation as 2*pi/n;
    basis - None or np.2darray[3][3], (optional, default None), if not None,
            defines the basis of the transformation.

    Returns:
    rotation - np.2darray[3][3], 3D proper rotation matrix.
    """
    
    axis = _normalize_vector(axis)
    angle = 2*np.pi/n
    
    # Define trig functions
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation = np.zeros((3,3))

    rotation[0,0] = cos_a + axis[0]**2*(1-cos_a)
    rotation[0,1] = axis[0]*axis[1]*( 1 - cos_a ) - axis[2]*sin_a 
    rotation[0,2] = axis[0]*axis[2]*( 1 - cos_a ) + axis[1]*sin_a

    rotation[1,0] = axis[1]*axis[0]*( 1 - cos_a ) + axis[2]*sin_a
    rotation[1,1] = cos_a + axis[1]**2*(1-cos_a) 
    rotation[1,2] = axis[1]*axis[2]*( 1 - cos_a ) - axis[0]*sin_a
    
    rotation[2,0] = axis[2]*axis[0]*( 1 - cos_a ) - axis[1]*sin_a
    rotation[2,1] = axis[2]*axis[1]*( 1 - cos_a ) + axis[0]*sin_a 
    rotation[2,2] = cos_a + axis[2]**2*(1-cos_a)
    
    if not_None(basis):
        basis_inv = np.linalg.inv(basis)
        rotation = basis_inv.dot(rotation.dot(basis))

    return rotation


def operator_M(axis, basis = None):
    """
    Defines a 3D mirror reflection.
    
    Mirror reflection is defined as a 180 degree rotation followed by an
    inversion:

    M(axis) = I * C(axis,2) = -C(axis,2).

    Arguments:
    axis - np.1darray[3], normal vector of the mirror plane, not necesserally 
           normalized;
    basis - None or np.2darray[3][3], (optional, default None), if not None,
            defines the basis of the transformation.

    Returns:
    reflection_matrix - np.2darray[3][3], 3D reflection matrix.
    """

    axis = _normalize_vector(axis)

    reflection = -operator_C(axis,2,basis)

    return reflection


def operator_S(axis, n, basis = None):
    """
    Defines an improper 3D rotation.

    Improper rotation is defined as

    S(axis,n) = C(axis,n) * M(axis)

    Arguments:
    axis - np.1darray[3], rotation axis, not necesserally normalized;
    n    - defines the angle of rotation as 2*pi/n;
    basis - None or np.2darray[3][3], (optional, default None), if not None,
            defines the basis of the transformation.

    Returns:
    rotoinversion - np.2darray[3][3], 3D improper rotation matrix.
    """

    axis = _normalize_vector(axis)
    angle = 2*np.pi/n

    rotoinversion = operator_C(axis,n,basis).dot(operator_M(axis,basis))

    return rotoinversion


def operator_to_symbol(operator):
    """
    Determines the correct symmetry symbol of the 3D matrix operation.

    ---------------------------------------------------------------------------
    Proper and improper operations
    ---------------------------------------------------------------------------
    3D operations are defined as orthogonal matrices, meaning that 

    |det(operator)| = 1.

    If matrix operation is proper (rotation), we have

    det(operator) = 1

    Otherwise, det(operator = -1, which means that the operation is improper
    (reflections, rotoinversions). 
    

    ---------------------------------------------------------------------------
    Angle of rotation
    ---------------------------------------------------------------------------
    The angle of the rotation can be determined for proper and improper 
    operators by using

    trace(operator)/det(operator) = 1 + 2cos(angle).

    ---------------------------------------------------------------------------
    Axis of rotation
    ---------------------------------------------------------------------------
    The rotation axis can be calculated using the property

    Skew(axis) = (operator - operator.T),

    where Skew(axis) is the skew-symmetric matrix. 
    (see https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis)


    Arguments:
    operator - np.2darray[3][3], 3D matrix symmetry operation.

    Returns:
    symbol - str, symbolic representation of the symmetry operation.
    """
   
    # Calculate the trace and determinant of the operator matrix
    operator_trace = np.trace(operator)
    operator_det = np.linalg.det(operator)

    # Define numerical precision
    log_eps = 10
    eps = 1e-10
    
    if abs(abs(operator_det) - 1.0) > eps:
        raise ValueError("Cannot determine symmetry symbol: operator "\
                         "{}".format(operator)\
                       + " is a non-orthogonal matrix!")

    # Calculate the angle of rotation
    cos_a = np.round(0.5*(operator_trace/operator_det - 1),log_eps)
    angle = np.arccos(cos_a)

    n = np.round(2*np.pi/angle,log_eps).astype(int)

    # Calculate the axis of rotation
    operator_skew = (operator-operator.T)/operator_det
    axis = 0.5*np.array([operator_skew[2,1] - operator_skew[1,2],
                         operator_skew[0,2] - operator_skew[2,0],
                         operator_skew[1,0] - operator_skew[0,1]])

    axis /= np.max(abs(axis))
    
    # Correct the axis for the reflections
    if n == 2:
        axis /= operator_det
    
    axis = np.round(axis,log_eps).astype(int)
    axis = "".join(str(a) for a in axis)

    # Determine if the operation is proper or improper
    if operator_det > 0:
        symbol = 'C' + str(n) + ',' + axis

    else:
        if n > 2:
            symbol = 'S' + str(n) + ',' + axis
        else:
            symbol = 'M' + ',' + axis

    return symbol

def symbol_to_operator(symbol, basis = None):
    """
    Determines the 3D matrix operation for a given symbolic representation.

    Arguments:
    symbol - str, symbolic representation of the symmetry operation;
    basis  - None or np.2darray[3][3], (optional, default None), if not None,
             defines the basis of the transformation.

    Returns:
    operator - np.2darray[3][3], 3D matrix symmetry operation.
    """

    operation, axis = symbol.split(',') 
    axis = np.array(list(axis), dtype=float)

    if operation == 'M':
        operator = operator_M(axis,basis)

    else:
        operation_type, n = list(operation)
        n = int(n)

        if operation_type == 'C':
            operator = operator_C(axis,n,basis)

        elif operation_type == 'S':
            operator = operator_S(axis,n,basis)

        else:
            raise ValueError("Incorrect symmetry symbol: " + symbol)

    return operator


"""
-------------------------------------------------------------------------------
Frequently used supplementary functions
-------------------------------------------------------------------------------
"""

def _normalize_vector(v):
    """
    Shortcut normalization function with checks for unit and zero vectors.

    Arguments:
    v - array_type, vector to normalize.

    Returns:
    v - if |v| > 0, np.1darray, v = v/|v| normalized vector;
        if |v| = 0, Value Error.
    """
    
    v_norm = np.linalg.norm(np.array(v))
    
    # Define numerical precision for the norm
    eps = 1e-10


    if v_norm < eps:
        raise ValueError("Cannot normalize a vector with zero norm!")

    elif v_norm - 1.0 > eps:
        v /= v_norm

    return v
