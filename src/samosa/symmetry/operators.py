#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
============================
samosa/symmetry/operators.py
============================

This program defines 3D symmetry operations, as well as methods for converting
between different representations of these operations.
"""

import numpy as np

from fractions import Fraction

from samosa.utils.type_checks import (
    ArrayType,
    NoneType,
    not_None,
    check_type,
    check_len,
    check_shape,
)

from samosa.utils.math import _normalize_vector

"""
-------------------------------------------------------------------------------
Symmetry operators in 3D
-------------------------------------------------------------------------------
"""


def operator_C(axis, angle, basis=None):
    """
    Defines a proper 3D rotation.

    The general expression for a proper rotation around a 3D axis is found in
    (https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)

    Arguments:
    axis  - np.1darray[3], rotation axis, not necesserally normalized;

    angle - if float, defines the angle of rotation;
            if int n, defines the angle of rotation as 2*pi/n;
            if tuple of ints (k, n), defines the angle of rotation as 2*pi*k/n;

    basis - None or np.2darray[3][3], (default=None), if not None,
            defines the basis of the transformation.

    Returns:
    rotation - np.2darray[3][3], 3D proper rotation matrix.
    """

    check_type("axis", axis, ArrayType)
    check_len("axis", axis, 3)
    check_type("angle", angle, float, int, tuple)

    axis = _normalize_vector(axis)

    if isinstance(angle, int):
        n = angle
        angle = 2 * np.pi / n

    elif isinstance(angle, tuple):
        check_len("angle = (k, n)", angle, 2)
        check_type("k", angle[0], int)
        check_type("n", angle[1], int)

        k, n = angle
        angle = 2 * np.pi * k / n

    # Define trig functions
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    rotation = np.zeros((3, 3))

    rotation[0, 0] = cos_a + axis[0] ** 2 * (1 - cos_a)
    rotation[0, 1] = axis[0] * axis[1] * (1 - cos_a) - axis[2] * sin_a
    rotation[0, 2] = axis[0] * axis[2] * (1 - cos_a) + axis[1] * sin_a

    rotation[1, 0] = axis[1] * axis[0] * (1 - cos_a) + axis[2] * sin_a
    rotation[1, 1] = cos_a + axis[1] ** 2 * (1 - cos_a)
    rotation[1, 2] = axis[1] * axis[2] * (1 - cos_a) - axis[0] * sin_a

    rotation[2, 0] = axis[2] * axis[0] * (1 - cos_a) - axis[1] * sin_a
    rotation[2, 1] = axis[2] * axis[1] * (1 - cos_a) + axis[0] * sin_a
    rotation[2, 2] = cos_a + axis[2] ** 2 * (1 - cos_a)

    if not_None(basis):
        basis = basis.T
        basis_inv = np.linalg.inv(basis)
        rotation = basis_inv.dot(rotation.dot(basis))

    return rotation


def operator_M(axis, basis=None):
    """
    Defines a 3D mirror reflection.

    Mirror reflection is defined as a 180 degree rotation followed by an
    inversion:

    M(axis) = I * C(axis, 2) = -C(axis, 2).

    Arguments:
    axis - np.1darray[3], normal vector of the mirror plane, not necesserally
           normalized;

    basis - None or np.2darray[3][3], (default=None), if not None,
            defines the basis of the transformation.

    Returns:
    reflection_matrix - np.2darray[3][3], 3D reflection matrix.
    """

    check_type("axis", axis, ArrayType)
    check_len("axis", axis, 3)

    axis = _normalize_vector(axis)

    reflection = -operator_C(axis, 2, basis)

    return reflection


def operator_S(axis, angle, basis=None):
    """
    Defines an improper 3D rotation.

    Improper rotation is defined as

    S(axis, angle) = C(axis, angle) * M(axis)

    Arguments:
    axis - np.1darray[3], rotation axis, not necesserally normalized;

    angle - if float, defines the angle of rotation;
            if int n, defines the angle of rotation as 2*pi/n;
            if tuple of ints (k, n), defines the angle of rotation as 2*pi*k/n;

    basis - None or np.2darray[3][3], (default=None), if not None,
            defines the basis of the transformation.

    Returns:
    rotoinversion - np.2darray[3][3], 3D improper rotation matrix.
    """

    check_type("axis", axis, ArrayType)
    check_len("axis", axis, 3)
    check_type("angle", angle, float, int, tuple)

    axis = _normalize_vector(axis)

    rotoinversion = operator_C(axis, angle, basis).dot(operator_M(axis, basis))

    return rotoinversion


"""
-------------------------------------------------------------------------------
Tools for converting between matrix and symbolic representations
-------------------------------------------------------------------------------
"""


def operator_to_symbol(operator, as_dict=False, eps=1e-10):
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

    From here, we assume that angle = 2 pi k / n, and extract the two integers.

    ---------------------------------------------------------------------------
    Axis of rotation
    ---------------------------------------------------------------------------
    The rotation axis can be calculated using the property

    Skew(axis) = (operator - operator.T),

    where Skew(axis) is the skew-symmetric matrix.
    (see https://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis)

    Arguments:
    operator - ArrayType[3][3], 3D matrix symmetry operation;

    as_dict  - bool, (default=False) if True, returns a dictionary with keys
               ['type', 'n, 'k', 'axis']. If False, returns the symbol as a
               formatted string;

    eps      - float, (default=1e-10) defines numerical precision of float
               operations;

    Returns:
    symbol - (as_dict=False) str, symbolic representation of the symmetry
             operation;
             (as_dict=True) dict, returns the operator information as a
             dictionary with keys ['type', 'n, 'k', 'axis'].
    """

    # Type checks
    check_type("operator", operator, ArrayType)
    check_type("as_dict", as_dict, bool)
    check_type("eps", eps, float)

    log_eps = (-np.log10(eps)).astype(int)

    operator = np.array(operator)
    check_shape("operator", operator, 3, 3)

    # Calculate the trace and determinant of the operator matrix
    operator_trace = np.trace(operator)
    operator_det = np.linalg.det(operator)

    if abs(abs(operator_det) - 1.0) > eps:
        raise ValueError(
            "Cannot determine symmetry symbol: operator "
            f"{operator} is a non-orthogonal matrix!"
        )

    # Calculate the angle of rotation
    cos_a = np.round(0.5 * (operator_trace / operator_det - 1), log_eps)
    angle = np.arccos(cos_a)

    frac = Fraction(np.round(angle / (2 * np.pi), log_eps)).limit_denominator()
    k = frac.numerator
    n = frac.denominator

    # Calculate the axis of rotation
    operator_skew = (operator - operator.T) / operator_det
    axis = 0.5 * np.array(
        [
            operator_skew[2, 1] - operator_skew[1, 2],
            operator_skew[0, 2] - operator_skew[2, 0],
            operator_skew[1, 0] - operator_skew[0, 1],
        ]
    )

    axis /= np.max(abs(axis)) + eps

    # Correct the axis for the reflections
    if n == 2:
        axis /= operator_det

    axis = np.round(axis, log_eps).astype(int)

    # Determine if the operation is proper or improper
    if operator_det > 0:
        operator_type = "C"

    else:
        if n > 2:
            operator_type = "S"
        else:
            operator_type = "M"

    if as_dict:
        symbol = {"type": operator_type, "n": n, "k": k, "axis": axis}

    else:
        axis = "".join(str(a) for a in axis)
        if operator_type != "M":
            symbol = f"{operator_type}{n}^{k}; {axis}"
        else:
            symbol = f"{operator_type}; {axis}"

    return symbol


def symbol_to_operator(symbol, basis=None):
    """
    Determines the 3D matrix operation for a given symbolic representation.

    Arguments:
    symbol - str, symbolic representation of the symmetry operation;

    basis  - None or np.2darray[3][3], (default=None), if not None,
             defines the basis of the transformation.

    Returns:
    operator - np.2darray[3][3], 3D matrix symmetry operation.
    """
    # Type checks
    check_type("symbol", symbol, str)
    check_type("basis", basis, NoneType, ArrayType)

    # Separate the operation and the axis
    operation, axis = symbol.split(";")
    axis = np.array(list(axis), dtype=float)

    if operation == "M":
        operator = operator_M(axis, basis)

    else:
        operation, k = operation.split("^")
        operation_type, n = list(operation)
        k = int(k)
        n = int(n)

        if operation_type == "C":
            operator = operator_C(axis, (k, n), basis)

        elif operation_type == "S":
            operator = operator_S(axis, (k, n), basis)

        else:
            raise ValueError(f"Incorrect symmetry symbol: {symbol}.")

    return operator
