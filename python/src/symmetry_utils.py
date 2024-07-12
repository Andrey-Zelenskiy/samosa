#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

import numpy as np
import toml
import warnings
import pandas as pd

def custom_formatwarning(msg, *args, **kwargs):
    """
    When throwing a warning, only throw the message
    """
    return str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

# Define lookup libraries
point_group_lib = toml.load('../../include/point_groups.toml')
symmetry_operators = toml.load('../../include/symmetry_operations.toml')

symbol_to_operator = {}
operator_to_symbol = {}
for axis in symmetry_operators.keys():
    for operation in symmetry_operators[axis].keys():
        symbol = operation+','+axis
        matrix_operator = np.array(symmetry_operators[axis][operation])
        
        symbol_to_operator[symbol] = matrix_operator
        operator_to_symbol[tuple(matrix_operator.flatten())] = symbol

def get_symmetry_matrix(symbol):
    """
    Translates the symmetry symbol, as appearing in symmetry_operators
    library, to a matrix operator in the lattice basis.

    Arguments:
    symbol - str, symmetry symbol written as '_operation_,_axis_'.

    Returns:
    matrix_operator - np.2darray, matrix operator in the lattice basis.
    """
    operation, axis = symbol.split(',')

    # Check syntax
    axis_list = list(symmetry_operators.keys())

    if axis not in axis_list:
        raise NameError('Invalid axis: ' + axis + \
                        '\nSupported axes: ' + ", ".join(axis_list))

    operation_list = list(symmetry_operators[axis].keys())
    
    if operation not in operation_list:
        raise NameError('Invalid operation symbol: ' + operation + \
                        '\nSupported operations for the given symmetry axis '\
                        'are ' + ", ".join(operation_list))
    
    matrix_operator = symbol_to_operator[symbol]

    return matrix_operator

def get_symmetry_symbol(matrix_operator):
    """
    Reverse of the get_symmetry_matrix: given a matrix operator, conver it to 
    a symmetry symbol, as appearing in symmetry_operators

    Arguments:
    matrix_operator - np.2darray, matrix operator in the lattice basis.

    Returns:
    symbol - str, symmetry symbol written as '_operation_,_axis_'.
    """

    lib_key = tuple(matrix_operator.flatten())

    if lib_key not in operator_to_symbol.keys():
        raise ValueError('Invalid matrix operator: ' + str(matrix_operator))
    
    symbol = operator_to_symbol[lib_key]
    
    return symbol

def array_in_list(a, a_list):
    """
    Determines if a np.ndarray is included in a list of np.ndarrays. 
    """
    return any((a == p).all() for p in a_list)

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

def permutation_representation(a_list, matrix_operator):
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
