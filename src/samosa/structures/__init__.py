#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
samosa/structures
=================

This submodule defines geometric structures that can be used as subjects of the
symmetry analysis.
"""

import numpy as np

from samosa.symmetry import space_group

from samosa.symmetry.group import Orbit

from samosa.database import SpaceGroupDatabase

from samosa.utils.array_checks import _array_in_list

from samosa.utils.type_checks import (
    ArrayType,
    NoneType,
    is_None,
    not_None,
    check_type,
)

from samosa.utils.errors import custom_format_warning

import warnings

warnings.formatwarning = custom_format_warning

"""
-------------------------------------------------------------------------------
Lattice class
-------------------------------------------------------------------------------
"""


class Lattice:
    """
    Defines a container for storing all relevant spatial information about a
    crystallographic lattice.
    """

    # Lattice initializers
    def __init__(
        self,
        dimension,
        space_group_index,
        positions=[[0, 0, 0]],
        crystal_parameters=None,
    ):
        """
        Initialize all relevant space group information from the lattice
        dimension and space group number.

        Arguments:
        dimension          - int, dimension of the lattice;

        space_group_index  - int, space group number (as per International
                             Tables for Crystallography, volume A);

        positions          - list of ArrayType[3], (default=[[0, 0, 0]]),
                             vertex (Wyckoff) positions written in fractional
                             coordinates. Only inequivalent Wyckoff sites need
                             to be specified;

        crystal_parameters - (default=None), physical parameters of the lattice
                             (unit cell dimensions and angles):

                             if None, initializes the unconstrained parameters
                             randomly;

                             if dict with allowed keys
                             ['a', 'b', 'c', 'alpha', 'beta', 'gamma'],
                             initializes parameters from user input.
        """
        # Type checks
        check_type("dimension", dimension, int)
        check_type("space_group_index", space_group_index, int)
        check_type("positions", positions, list)

        input_positions = []
        for p in positions:
            check_type("vertex position", p, ArrayType)

            p = np.array(p) % 1

            if not _array_in_list(p, input_positions):
                input_positions += [p]

        check_type("crystal_parameters", crystal_parameters, NoneType, dict)

        # Initialize space group database container
        database = SpaceGroupDatabase()

        # Initialize symmetry properties
        sg_info = space_group(
            dimension, space_group_index, database, return_info=True
        )

        self.__dimension = dimension
        self.__space_group_index = space_group_index
        self.__space_group = sg_info["space_group"]
        self.__point_group_symbol = sg_info["point_group_symbol"]
        self.__lattice_type = sg_info["lattice_type"]
        self.__symmorphic = sg_info["symmorphic"]

        # Initialize structural properties
        self.__initialize_crystal_parameters(crystal_parameters, database)

        # Group sites into Wyckoff classes
        self.__wyckoff_list = Orbit.sort_into_orbits(
            input_positions, self.space_group
        )

    def __initialize_crystal_parameters(self, parameters, database):
        """
        Determines lattice parameters (primitive cell dimensions and angles)
        constrained by the lattice type.

        Arguments:
        parameters - (default=None), physical parameters of the lattice
                     (unit cell dimensions and angles):

                     if None, initializes the unconstrained parameters
                     randomly;

                     if dict with allowed keys
                     ['a', 'b', 'c', 'alpha', 'beta', 'gamma'],
                     initializes parameters from user input;

        database   - SpaceGroupDatabase object, lookup data structure.
        """
        self.__random_parameters = False

        # If no input crystal parameters are provided, initialize them randomly
        if is_None(parameters):
            self.__random_parameters = True
            parameters = {}

            # Normalize dimensions with respect to x-direction
            parameters["a"] = 1.0

            for p in ["b", "c"]:
                # Initialize random dimension values between 0 and 5
                parameters[p] = np.random.rand() * 5.0

            for p in ["alpha", "beta", "gamma"]:
                # Initialize random angle values between 0 and pi
                parameters[p] = np.random.rand() * 180

        # Determine constraints on the physical crystal parameters from the
        # lattice type
        reference = database.lattice_reference[self.lattice_type]
        self.__unconstrained_parameters = reference["unconstrained_parameters"]
        self.__constrained_parameters = reference["constrained_parameters"]

        self.__crystal_parameters = None
        self.set_crystal_parameters(parameters)

    @classmethod
    def initialize_simple(cls, name, point_group):
        """
        Initializes Lattice object corresponding to one of the common Bravais
        lattices.

        Arguments:
        name        - str, valid names are: chain, square, hexagonal, cubic,
                      bcc, fcc;

        point_group - Schoenflies symbol of the site point group symmetry.
        """
        # Define space groups for the allowed lattice names/point group
        # combinations
        allowed_names = {
            "chain": {"C1": (1, 1), "Cs": (1, 2)},
            "square": {"C4": (2, 10), "C4v": (2, 11)},
            "hexagonal": {"C6": (2, 16), "C6v": (2, 17)},
            "cubic": {"O": (3, 207), "Oh": (3, 221)},
            "bcc": {"O": (3, 211), "Oh": (3, 229)},
            "fcc": {"O": (3, 209), "Oh": (3, 225)},
        }

        # Type checks
        check_type("name", name, str)
        check_type("point_group", point_group, str)

        if name not in allowed_names.keys():
            names_str = ", ".join(allowed_names.keys())
            raise ValueError(
                f"Lattice name must be one of {names_str}, not "
                f"{name}. Please choose a valid name or use the "
                f"default initializer."
            )

        if point_group not in allowed_names[name].keys():
            point_groups_str = ", ".join(allowed_names[name].keys())
            raise ValueError(
                f"Point group for lattice '{name}' must be one "
                f"of {point_groups_str}, not {point_group}. "
                f"Please choose a valid point group or use the "
                f"default initializer."
            )

        # Initialize the Lattice object
        dimension, space_group_index = allowed_names[name][point_group]

        return cls(dimension, space_group_index)

    # Lattice methods
    def set_crystal_parameters(self, parameters):
        """
        Method used to set unit cell parameters with respected symmetry
        constraints.
        """

        if not_None(self.crystal_parameters) and self.random_parameters:
            self.__random_parameters = False

        self.__crystal_parameters = {}

        try:
            for p in self.unconstrained_parameters:
                self.__crystal_parameters[p] = parameters[p]

        except KeyError:
            raise Exception(
                f"{self.lattice_type} lattice type requires "
                f"definition of {self.unconstrained_parameters} "
                f"crystal parameters."
            )

        for p in self.constrained_parameters.keys():
            p_val = self.constrained_parameters[p]
            if p in ["b", "c"]:
                self.__crystal_parameters[p] = self.crystal_parameters[p_val]

            else:
                self.__crystal_parameters[p] = p_val

        # Initialize basis vectors
        """
        In order to calculate the crystal basis vectors a_{1-3}, it is
        sufficient to know their dot products:

        a_1 * a_2 = a b cos(gamma),
        a_2 * a_3 = b c cos(alpha),
        a_3 * a_1 = c a cos(beta).

        By convention, we fix the global orientation of the coordinate system
        by taking a_1 parallel to the x-axis, and a_2 - to be lying in the
        xy-plane. In this case,

        a_1 = a [1, 0, 0],
        a_2 = b [cos(gamma), sin(gamma), 0].

        To calculate the third basis vector, we note that the unit vectors
        parallel to the y- and z- axes can be defined as

        z/|z| = (a_1 x a_2)/(a b sin(gamma)),
        y/|y| = ((a_1 x a_2) x a_1)/(a^2 b sin(gamma)).

        Then the third crystal vector can be written as
        a_3 = [c_x, c_y, c_z],

        where

        c_x = c cos(beta),
        c_y = c (cos(alpha) - cos(beta) cos(gamma))/sin(gamma),
        c_z = sqrt(c^2 - c_x^2 - c_y^2).
        """

        a = self.crystal_parameters["a"]

        self.__basis_vectors = [a * np.array([1, 0, 0])]

        if self.dimension > 1:
            b = self.crystal_parameters["b"]
            gamma = self.crystal_parameters["gamma"] / 180 * np.pi

            self.__basis_vectors += [
                b * np.array([np.cos(gamma), np.sin(gamma), 0])
            ]

        if self.dimension == 3:
            c = self.crystal_parameters["c"]
            alpha = self.crystal_parameters["alpha"] / 180 * np.pi
            beta = self.crystal_parameters["beta"] / 180 * np.pi

            cx = c * np.cos(beta)
            cy = (
                c
                * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
                / np.sin(gamma)
            )
            cz = np.sqrt(c ** 2 - cx ** 2 - cy ** 2)
            self.__basis_vectors += [np.array([cx, cy, cz])]

    # Lattice properties
    @property
    def dimension(self):
        """
        Lattice dimension (number of basis vectors).
        """
        return self.__dimension

    @property
    def space_group_index(self):
        """
        Index of the space group of the lattice.
        """
        return self.__space_group_index

    @property
    def point_group_symbol(self):
        """
        Schoenflies symbol of the crystallographic point group.
        """
        return self.__point_group_symbol

    @property
    def lattice_type(self):
        """
        Bravais lattice type.
        """
        return self.__lattice_type

    @property
    def space_group(self):
        """
        Space group symmetry of the lattice.
        """
        return self.__space_group

    @property
    def random_parameters(self):
        """
        True if unit cell parameters were initialized from random.
        """
        return self.__random_parameters

    @property
    def constrained_parameters(self):
        """
        Unit cell parameters for which the values are constrained by the
        symmetry of the lattice.
        """
        return self.__constrained_parameters

    @property
    def unconstrained_parameters(self):
        """
        Unit cell parameters which are specified through user input.
        """
        return self.__unconstrained_parameters

    @property
    def crystal_parameters(self):
        """
        Unit cell parameters (dimensions and angles).
        """
        return self.__crystal_parameters

    @property
    def basis_vectors(self):
        """
        Lattice basis vectors.
        """
        return self.__basis_vectors

    @property
    def wyckoff_list(self):
        """
        All vertex positions in a single unit cell, sorted into non-equivalent
        Wyckoff classes.
        """
        return self.__wyckoff_list

    # Output summary functions
    def symmetry_info(self, print_info=True):
        """
        Summary of the symmetry properties of the lattice.

        Arguments:
        print_info - bool, (default=True), if True prints the summary,
                     otherwise outputs it as a string.
        """
        summary_string = "----------------------------------\n"
        summary_string += "Symmetry properties of the lattice\n"
        summary_string += "----------------------------------\n\n"

        summary_string += f"Lattice type: {self.lattice_type};\n"

        summary_string += (
            f"Crystallographic point group: " f"{self.point_group_symbol};\n"
        )

        summary_string += (
            f"Space group index (ITC vol.A): " f"{self.space_group_index}."
        )

        if print_info:
            print(summary_string)

        else:
            return summary_string

    def structure_info(self, print_info=True):
        """
        Summary of the structural properties of the lattice.

        Arguments:
        print_info - bool, (default=True), if True prints the summary,
                     otherwise outputs it as a string.
        """
        summary_string = "------------------------------------\n"
        summary_string += "Structural properties of the lattice\n"
        summary_string += "------------------------------------\n\n"

        summary_string += "Unit cell parameters"

        if self.random_parameters:
            summary_string += " (initialized randomly)"

        summary_string += ":\n"

        for p in self.crystal_parameters.keys():
            summary_string += f"{p} = {self.crystal_parameters[p]:1.4f},\n"

        if len(self.wyckoff_list) == 1:
            summary_string += (
                f"\nSites in the unit cell occupy "
                f"a single Wyckoff site, which has "
                f"multiplicity "
                f"{self.wyckoff_list[0].size};"
            )

        else:
            m_list = [str(w.size) for w in self.wyckoff_list]
            m_list = ", ".join(m_list)
            summary_string += (
                f"\nSites in the unit cell occupy "
                f"{len(self.wyckoff_list)} inequivalent "
                f"Wyckoff sites, and have multiplicities "
                f"{m_list};"
            )

        if print_info:
            print(summary_string)

        else:
            return summary_string

    def __str__(self):
        """
        Provides user-friendly summary of the Lattice container.
        """

        summary_string = "Lattice object.\n\n"

        summary_string += self.symmetry_info(print_info=False)

        summary_string += "\n\n"

        summary_string += self.structure_info(print_info=False)

        return summary_string

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return (
            f"{cls}(dimension={self.dimension}, "
            f"space_group_index={self.space_group_index})"
        )
