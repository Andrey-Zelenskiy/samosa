#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
symmetry_utils.py
=================


"""

import numpy as np
import toml
import warnings

def custom_formatwarning(msg, *args, **kwargs):
    """
    When throwing a warning, only throw the message
    """
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

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


def array_in_list(a, a_list):
    """
    Determines if a np.ndarray is included in a list of np.ndarrays. 
    """
    return any((a == p).all() for p in a_list)



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



# Routines for orbit/stabilizers calculations

def stabilizer_generator_filter(stabilizer, stabilizer_list, blacklist):
    """
    Determines whether a stabilizing operation qualifies as a new generator
    of the stabilizer group. 

    Arguments:
    stabilizer      - np.2darray, stabilizer matrix;
    stabilizer_list - list of np.2darray, list of known generators of the
                      stabilizer group;
    blacklist       - list of np.2darray, list of operators related to the known
                      generators of the stabilizer group.
    
    Returns:
    Either the original, or the updated stabilizer_list, blacklist.
    """

    # Check if the stabilizer is already in the stabilizer list
    if not array_in_list(stabilizer, stabilizer_list):
        
        # Check if the stabilizer is related to the known generators
        if not array_in_list(stabilizer, blacklist):  

            # Both tests are passed, update stabilizer_list and blacklist  
            extended_stabilizer = [np.eye(3,dtype=int)] + stabilizer_list
            
            for s in extended_stabilizer:
                g_head = s.dot(stabilizer)
                g_chain = g_head
                while not array_in_list(g_chain, blacklist):
                    blacklist += [g_chain]
                    g_chain = g_chain.dot(g_head)
            
            stabilizer_list += [stabilizer]
    
    return stabilizer_list, blacklist         

def point_transform(point_0, generators):
    """
    Applies generators of the group to a single point. 
    Operations assume generators to be matrices of type np.2darray, and point
    of type np.1darray.

    Arguments:
    point_0          - np.1darray, 'seed' point for the transformation;
    generators       - array of np.2darray, generators of the group.

    Returns:
    orbit            - list, a set of partners generated from the seed point;
    transporter_dict - dict tuple:np.2darray, a set of group operators (values)
                       that transform the seed to the other points in the 
                       orbit (keys);
    stabilizer_list  - list of np.2darray, operators that leave the seed point
                       unchanged.
    """

    orbit = [point_0]
    transporter_dict = {tuple(point_0):np.eye(len(generators[0]),dtype=int)}
    stabilizer_list = []
    blacklist = [np.eye(len(generators[0]),dtype=int)]
    
    for p in orbit:
        for g in generators:
            q = g.dot(p)
            
            if not array_in_list(q, orbit):
                orbit += [q]
                transporter_dict[tuple(q)] = g.dot(transporter_dict[tuple(p)])

            else:
                """
                We assume that some group generator g satisfies

                g_21 p_1 = p_2

                where p and q are points in the orbit of a known seed point
                p_0. Assuming that

                p_1 = g_10 p_0
                p_2 = g_20 p_0

                where operators g_10 and g_20 are known, we can obtain two
                stabilizer operations as

                s_1 = g_20^-1 g_21 g_10
                s_2 = g_10^-1 g_21^-1 g_20 = s_1^-1

                Since we are interested in generators, we only need s_1.
                """
                
                g_10 = transporter_dict[tuple(p)]
                g_20 = transporter_dict[tuple(q)]
                g_20_inv = np.linalg.inv(g_20).astype(int)
                s_1 = g_20_inv.dot(g.dot(g_10))

                stabilizer_list,blacklist = stabilizer_generator_filter(s_1,
                                                            stabilizer_list,
                                                            blacklist)
    
    return orbit, transporter_dict, stabilizer_list

def as_permutation(a_list, matrix_operator):
    """
    Transforms a matrix operator into a cycle/permutation basis using a closed
    set of points.

    Arguments:
    a_list          - list of np.1darray, a set of points, which defines the
                      space of permutations;
    matrix_operator - np.2darray, a symmetry operator acting on the points in 
                      the a_list.

    Returns:
    permutation_operator - tuple of size len(a_list), permutation 
                           representation of the matrix_operator.
    """

    # Construct a hash map for the points in the a_list  
    dict_keys = [tuple(p) for p in a_list]
    dict_vals = [i for i in range(len(a_list))]
    index = dict(zip(dict_keys,dict_vals))

    try:
        permutation_operator = [index[tuple(matrix_operator.dot(p))] \
                                for p in a_list]

    except KeyError:
        raise ValueError("Cannot determine the permutation representation: "\
                       + "list of points is not a closed set")


    return permutation_operator
