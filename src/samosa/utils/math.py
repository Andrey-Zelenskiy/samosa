#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
====================
samosa/utils/math.py
====================

This script defines custom algebraic operations used in samosa.
"""

import numpy as np

from samosa.utils.type_checks import not_None


def _mod(a, n):
    """
    If n is not None, return a % n, otherwise return a.
    """
    if not_None(n):
        return a % n
    else:
        return a


def _normalize_vector(v, eps=1e-10):
    """
    Shortcut normalization function with checks for unit and zero vectors.

    Arguments:
    v   - ArrayType, vector to normalize;

    eps    - float, (default=1e-10) numerical precision for element comparison.

    Returns:
    v - if |v| > 0, np.1darray, v = v/|v| normalized vector;
        if |v| = 0, Value Error.
    """
    v_norm = np.linalg.norm(np.array(v))

    if v_norm < eps:
        raise ValueError("Cannot normalize a vector with zero norm!")

    elif np.abs(v_norm - 1.0) > eps:
        v /= v_norm

    return v
