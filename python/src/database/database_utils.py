#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
database_utils.py
=================

This module provides tools to translate symmetry information contained in
frusa_symmetry/database to symmetry objects defined in symmetry_utils.py.
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


# Define lookup libraries
point_group_lib = toml.load('../../database/point_groups.toml')
symmetry_operators = toml.load('../../database/symmetry_operations.toml')

symbol_to_operator = {}
operator_to_symbol = {}
for axis in symmetry_operators.keys():
    for operation in symmetry_operators[axis].keys():
        symbol = operation+','+axis
        matrix_operator = np.array(symmetry_operators[axis][operation])
        
        symbol_to_operator[symbol] = matrix_operator
        operator_to_symbol[tuple(matrix_operator.flatten())] = symbol

# Routines for going between symbolic and matrix representations
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
