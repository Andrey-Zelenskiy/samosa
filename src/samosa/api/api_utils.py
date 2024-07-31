#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
api_utils.py
=================

This script defines useful methods for type checking and error handling.
"""

import numpy as np
import warnings


"""
-------------------------------------------------------------------------------
Only show the message of the warning
-------------------------------------------------------------------------------
"""
def custom_format_warning(msg, *args, **kwargs):
    """
    When throwing a warning, only throw the message
    """
    return str(msg) + '\n'

warnings.formatwarning = custom_format_warning


"""
-------------------------------------------------------------------------------
Type checking
-------------------------------------------------------------------------------
"""
# Group np.ndarray, list, and tuple into a single class category
array_type = (np.ndarray, list, tuple)


def check_type(var_name, var, *var_type):
    """
    Checks that the variable is of the specified type.

    Arguments:
    var_name - str, name of the variable;
    var      - generic type, variable of interest;
    var_type - tuple, a list of acceptible types for var.

    Returns:
    None if the check is successful,
    TypeError if the check is failed.
    """

    if not isinstance(var, var_type):
        var_type_print = [str(t.__name__) for t in var_type]
        var_type_print = ", ".join(var_type_print)
        raise TypeError(var_name\
                      + " must be of type {}, not {}".format(
                                                      var_type_print,
                                                      type(var).__name__))


def check_len(var_name, var, var_len):
    """
    Checks the length of an array.

    Arguments:
    var_name - str, name of the variable;
    var      - array_type, array variable of interest;
    var_len  - int, required length of the array.

    Returns:
    None if the check is successful,
    Exception if the check is failed.
    """

    if len(var) != var_len:
        raise: Exception(var_name\
                       + " must be of length {}, not {}".format(var_len,
                                                                len(var)))


def check_shape(var_name, var, *var_shape):
    """
    Checks the shape of the np.ndarray.

    Arguments:
    var_name  - str, name of the variable;
    var       - np.ndarray, array variable of interest;
    var_shape - tuple of int, required shape of the array.

    Returns:
    None if the check is successful,
    Exception if the check is failed.
    """
    
    if var.shape != var_shape:
        raise: Exception(var_name\
                       + " must have shape {}, not {}".format(var_shape,
                                                              var.shape))
