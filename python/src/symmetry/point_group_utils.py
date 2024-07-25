#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
====================
point_group_utils.py
====================

This program defines methods for defining and manipulating 3D point groups.
"""

import numpy as np
from symmetry_utils import matrix_group_element

# Frequently used functions

def normalize_vector(v):
    """
    Shortcut normalization function with checks for unit and zero vectors.

    Arguments:
    v - np.1darray, vector to normalize.

    Returns:
    v - if |v| > 0, np.1darray, v = v/|v| normalized vector;
        if |v| = 0, Value Error.
    """
    
    v_norm = np.linalg.norm(v)
    
    # Define numerical precision for the norm
    eps = 1e-10


    if v_norm < eps:
        raise ValueError("Cannot normalize a vector with zero norm!")

    elif v_norm - 1.0 > eps:
        v /= v_norm

    return v


# Definitions of symmetry elements

def operator_C(axis, n):
    """
    Defines a proper 3D rotation.

    The general expression for a proper rotation around a 3D axis is found in
    (https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)

    Arguments:
    axis - np.1darray[3], rotation axis, not necesserally normalized;
    n    - defines the angle of rotation as 2*pi/n.

    Returns:
    rotation - np.2darray[3][3], 3D proper rotation matrix.
    """
    
    axis = normalize_vector(axis)
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
    
    return rotation


def operator_M(axis):
    """
    Defines a 3D mirror reflection.
    
    Mirror reflection is defined as a 180 degree rotation followed by an
    inversion:

    M(axis) = I * C(axis,2) = -C(axis,2).

    Arguments:
    axis - np.1darray[3], normal vector of the mirror plane, not necesserally 
           normalized.

    Returns:
    reflection_matrix - np.2darray[3][3], 3D reflection matrix.
    """

    axis = normalize_vector(axis)

    reflection = -operator_C(axis,2)

    return reflection


def operator_S(axis, n):
    """
    Defines an improper 3D rotation.

    Improper rotation is defined as

    S(axis,n) = C(axis,n) * M(axis)

    Arguments:
    axis - np.1darray[3], rotation axis, not necesserally normalized;
    n    - defines the angle of rotation as 2*pi/n.

    Returns:
    rotoinversion - np.2darray[3][3], 3D improper rotation matrix.
    """

    axis = normalize_vector(axis)
    angle = 2*np.pi/n

    rotoinversion = operator_C(axis,n).dot(operator_M(axis))

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
                       + "{}".format(operator)\
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

def symbol_to_operator(symbol):
    """
    Determines the 3D matrix operation for a given symbolic representation.

    Arguments:
    symbol - str, symbolic representation of the symmetry operation.

    Returns:
    operator - np.2darray[3][3], 3D matrix symmetry operation.
    """

    operation, axis = symbol.split(',') 
    axis = np.array(list(axis), dtype=float)

    if operation == 'M':
        operator = operator_M(axis)

    else:
        operation_type, n = list(operation)
        n = int(n)

        if operation_type == 'C':
            operator = operator_C(axis,n)

        elif operation_type == 'S':
            operator = operator_S(axis,n)

        else:
            raise ValueError("Incorrect symmetry symbol: " + symbol)

    return operator

# Routines that construct a point group from its Schoenflies symbol 

def point_group(pg_symbol, store_inverse = True):
    """
    Creates a group object from the Schoenflies symbol of the point group.

    Arguments:
    pg_symbol     - str, Schoenflies symbol of the point group;
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses.

    Returns: group object with the correct point group generator set.
    """

    generators = point_group_generators(pg_symbol,store_inverse)

    return group(generators)

def point_group_generators(pg_symbol, store_inverse = True):
    """
    Calculates the generators of a 3D point group given its Schoenflies symbol.   
    
    Arguments:
    pg_symbol     - str, Schoenflies symbol of the point group;
    store_inverse - bool, (optional, default = True), if True, the generator
                    objects explicitly store their inverses.

    Returns: 
    generator_list - list of matrix_group_element, point group generator set.
    """

    symbol_error = ValueError("Incorrect Schoenflies symbol: " + pg_symbol) 

    pg_type = pg_symbol[0]
    
    try:
        n = int(pg_symbol[1])
    
    except:
        raise symbol_error
        

    if pg_type == 'C':
        if len(pg_symbol) == 2:
            return Cn_generators(n, store_inverse)
        
        elif pg_symbol[2] == 'v':
            return Cnv_generators(n, store_inverse)

        elif pg_symbol[2] == 'h':
            return Cnh_generators(n, store_inverse)

        else:
            raise symbol_error
        
    elif pg_type == 'S':
        if len(pg_symbol) != 2:
            raise symbol_error

        return Sn_generators(n, store_inverse)

    elif pg_type == 'D':
        if len(pg_symbol) == 2:
            return Dn_generators(n, store_inverse)
        
        elif pg_symbol[2] == 'd':
            return Dnd_generators(n, store_inverse)
        
        elif pg_symbol[2] == 'h':
            return Dnh_generators(n, store_inverse)
        
        else:
            raise symbol_error



    else:
        raise symbol_error
