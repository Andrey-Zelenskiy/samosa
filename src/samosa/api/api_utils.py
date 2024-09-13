#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
============
api_utils.py
============

This script defines useful methods for type checking and error handling.
"""

import numpy as np

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


"""
-------------------------------------------------------------------------------
Argument checking
-------------------------------------------------------------------------------
"""
# Group np.ndarray, list, and tuple into a type categories
ArrayType = (np.ndarray, list, tuple)
MutableArrayType = (np.ndarray, list)
NoneType = type(None)


def is_None(var):
    """
    Checks if var is of NoneType, returns True if it is.
    """
    return isinstance(var, NoneType)


def not_None(var):
    """
    Checks if var is of NoneType, returns True if it isn't.
    """
    return not isinstance(var, NoneType)


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
        raise TypeError(f"{var_name} must be of type {var_type_print}, not "
                        f"{type(var).__name__}")


def check_len(var_name, var, var_len):
    """
    Checks the length of an array.

    Arguments:
    var_name - str, name of the variable;
    var      - ArrayType, array variable of interest;
    var_len  - int, required length of the array.

    Returns:
    None if the check is successful,
    Exception if the check is failed.
    """

    if len(var) != var_len:
        raise Exception(f"{var_name} must be of length {var_len}, not "
                        f"{len(var)}")


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
        raise Exception(f"{var_name} must have shape {var_shape}, not "
                        f"{var.shape}")


def check_ifdef(var_name, var):
    """
    Checks if the specified argument has a value of None.

    Arguments:
    var_name  - str, name of the variable;
    var       - generic type, variable of interest.

    Returns:
    None if the check is successful,
    Exception if the check is failed.
    """

    if not_None(var):
        raise Exception(f"{var_name} is already set to {var}, "
                        f"use overwrite=True to force a new value.")


def check_in_list(var_name, var, var_list):
    """
    Checks that the variable is contained in a list.

    Arguments:
    var_name  - str, name of the variable;
    var       - generic type, variable of interest;
    var_list  - list, list that is required to contain var.

    Returns:
    None if the check is successful,
    Exception if the check is failed.
    """

    if var not in var_list:
        raise Exception(f"{var_name} must be one of {var_list}, not {var}")
