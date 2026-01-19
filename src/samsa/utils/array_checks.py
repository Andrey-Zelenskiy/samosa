#! /usr/bin/env python3
# Andrey Zelenskiy, 2024-2026

"""
===========================
samsa/utils/array_checks.py
===========================

This script defines useful methods for testing various array properties.
"""

import numpy as np


def _array_equal(a1, a2, eps=1e-10):
    """
    Returns True if elements of array 1 are the same as elements of array 2
    up to specified numerical precision.

    Arguments:
    a1, a2 - ArrayType, arrays to be compared;

    eps    - float, (default=1e-10) numerical precision for element comparison.

    Returns:
    bool, result of the comparison test.
    """
    return np.isclose(a1, a2, eps).all()


def _array_in_list(a, a_list, eps=1e-10):
    """
    Determines if a np.ndarray is included in a list of np.ndarrays.

    Arguments:
    a      - ArrayType, array of interest;

    a_list - list, list of arrays;

    eps    - float, (default=1e-10) numerical precision for element comparison.

    Returns:
    bool, result of the search test.
    """
    return any(_array_equal(a, p, eps) for p in a_list)


def _element_in_list(g, g_list):
    """
    Determines if an object is included in a list of other objects.

    Arguments:
    g      - object interest;

    g_list - list of objects.

    Returns:
    bool, result of the comparison test.
    """
    return any(g == p for p in g_list)


def _check_orthogonal(operator, eps=1e-10):
    """
    Checks if a matrix operator (square) is orthogonal.

    Arguments:
    operator - ArrayType, square matrix of interest;

    eps    - float, (default=1e-10) numerical precision for element comparison.

    Returns:
    bool, result of the orthogonality test.
    """
    operator = np.array(operator)

    # Check if O * O.T = identity
    diff = np.max(np.abs(operator.dot(operator.T) - np.eye(len(operator))))

    return diff < eps
