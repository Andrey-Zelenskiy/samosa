#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
===============
smaosa/database
===============

This module provides tools to translate symmetry information contained in
samosa/database to symmetry objects defined in group_utils.py.
"""

import toml
import os
from pathlib import Path, PosixPath

from samosa.utils.type_checks import (
    NoneType,
    is_None,
    not_None,
    check_type,
    check_in_list,
)

from samosa.utils.errors import custom_format_warning

import warnings

warnings.formatwarning = custom_format_warning

"""
-------------------------------------------------------------------------------
Database classes
-------------------------------------------------------------------------------
"""


class Database:
    """
    Container for methods that access and examine the data files in
    samosa/database.
    """

    def __init__(self):
        """
        Creates an index of .toml files in the database and initializes
        containers for uploaded data.
        """
        # Create an index of available files
        self.get_file_index()

        # Initialize data file containers
        self.__file_list = []
        self.__data = {}

    def get_file_index(self):
        """
        Creates an index of .toml files in samosa/database.
        """
        self.__path = Path(__file__).parent.resolve()
        self.__file_index = []

        for file in os.listdir(self.path_to_database):
            if file.endswith(".toml"):
                self.__file_index += [self.path_to_database / file]

    def load_file(self, *files):
        """
        Creates a dictionary from the database file.
        """

        for f in files:
            # Check types
            check_type("file", f, int, str, PosixPath)

            if isinstance(f, int):
                file = self.__file_index[f]

            elif isinstance(f, str):
                file = self.path_to_database / PosixPath(f)

            else:
                file = f

            if file.name in self.file_list:
                continue

            self.__file_list += [file.name]
            self.__data[file.name] = toml.load(file)

    @property
    def available_files(self):
        """
        Index of .toml files in samosa/database
        """
        return [f.name for f in self.__file_index]

    @property
    def file_list(self):
        """
        List of uploaded files
        """
        return self.__file_list

    @file_list.deleter
    def file_list(self):
        self.__file_list = []
        self.__data = {}

    @property
    def data(self):
        """
        Uploaded data
        """
        return self.__data

    @data.deleter
    def data(self):
        self.__file_list = []
        self.__data = {}

    @property
    def path_to_database(self):
        """
        Absolute path to samosa/database
        """
        return self.__path

    def __str__(self):

        if len(self.file_list) == 0:
            return "Empty Database object"

        else:
            files_str = [str(f) for f in self.file_list]
            files_str = ", ".join(files_str)
            return f"Database object containing files {files_str}"

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(file_list = {self.file_list!r})"


class SpaceGroupDatabase(Database):
    """
    Database designated for space group data.
    """

    def __init__(self):
        """
        Uploads space group data files and creates lookup dictionaries.
        """

        Database.__init__(self)

        # Initialize data file containers
        self.load_file(
            self.path_to_database / "space_group_lattices.toml",
            self.path_to_database / "space_group_generators.toml",
        )

        self.get_lookup_dicts()

    def get_lookup_dicts(self):
        """
        Populates lookup dictionaries.
        """

        # Lookup tables for space group - point group - lattice relationships
        data = self.data["space_group_lattices.toml"]

        # List of lattice types
        self.__lattice_list = list(data.keys())

        # Lookup for allowed symmetry groups given lattice type
        self.__lattice_reference = {}
        # { "{lattice}" : {
        #   "dimension" : int,
        #   "unconstrained_parameters" : list,
        #   "constrained_parameters" : dict,
        #   "groups: { "{PG}" : list }
        #                 }
        # }

        # Lookup for allowed lattices given a point group
        self.__point_group_reference = {}
        # { "{PG}" : { "{lattice}" : list } }

        # Lookup for point group and lattice type for a given space group
        self.__space_group_reference = {}
        # { ( dim, SG ) : ( "{PG}", "{lattice}" ) }

        # Lookup for allowed lattice types and point groups given lattice
        # dimension
        self.__dimension_reference = {}
        # { int : { "lattices"     : list,
        #           "point groups" : list,
        #           "space groups" : list,
        #         }
        # }

        for lat in self.__lattice_list:
            dim = data[lat]["dimension"]
            param_u = data[lat]["unconstrained_parameters"]
            param_c = data[lat]["constrained_parameters"]
            pg_list = list(data[lat].keys())[3:]

            for pg in pg_list:
                sg_list = data[lat][pg]["space_groups"]

                for sg in sg_list:
                    # Storing the dimension and the space group index as a
                    # tuple removes ambiguity between 1d, 2d, and 3d space
                    # group indices
                    sg_t = (dim, sg)

                    self.__fill_lattice_reference(
                        lat, pg, sg, dim, param_u, param_c
                    )

                    self.__fill_point_group_reference(lat, pg, sg_t)

                    self.__fill_space_group_reference(lat, pg, sg_t)

                    self.__fill_dimension_reference(dim, lat, pg, sg)

        # List of allowed crystallographic groups
        self.__point_group_list = list(self.__point_group_reference.keys())

        # Number of space group in 1, 2, and 3D
        self.__n_space_groups = {1: 2, 2: 17, 3: 230}  # { dim : N_SG }

        # Separate the generators in a separate dict
        data = self.data["space_group_generators.toml"]
        self.__space_group_generators = {}  # { ( dim, SG ) : generators }

        for sg in data.keys():
            sg_t = tuple(int(n) for n in sg.split(","))
            self.__space_group_generators[sg_t] = data[sg]

    def __fill_lattice_reference(self, lat, pg, sg, dim, param_u, param_c):
        """
        Method for filling the lattice_reference table.
        """
        if lat not in self.__lattice_reference.keys():
            self.__lattice_reference[lat] = {
                "dimension": dim,
                "unconstrained_parameters": param_u,
                "constrained_parameters": param_c,
                "groups": {pg: [sg]},
            }
        else:
            if pg not in self.__lattice_reference[lat]["groups"].keys():
                self.__lattice_reference[lat]["groups"][pg] = [sg]
            else:
                self.__lattice_reference[lat]["groups"][pg] += [sg]

    def __fill_point_group_reference(self, lat, pg, sg_t):
        """
        Method for filling the point_group_reference table
        """
        if pg not in self.__point_group_reference.keys():
            self.__point_group_reference[pg] = {lat: [sg_t]}
        else:
            if lat not in self.__point_group_reference[pg].keys():
                self.__point_group_reference[pg][lat] = [sg_t]
            else:
                self.__point_group_reference[pg][lat] += [sg_t]

    def __fill_space_group_reference(self, lat, pg, sg_t):
        """
        Method for filling the space_group_reference table
        """
        if sg_t not in self.__space_group_reference.keys():
            self.__space_group_reference[sg_t] = (pg, lat)
        else:
            self.__space_group_reference[sg_t] += (pg, lat)

    def __fill_dimension_reference(self, dim, lat, pg, sg):
        """
        Method for filling the dimension_reference table
        """
        if dim not in self.__dimension_reference.keys():
            self.__dimension_reference[dim] = {
                "lattices": [lat],
                "point groups": [pg],
                "space groups": [sg],
            }
        else:
            dim_ref = self.__dimension_reference[dim]
            if lat not in dim_ref["lattices"]:
                self.__dimension_reference[dim]["lattices"] += [lat]
            if pg not in dim_ref["point groups"]:
                self.__dimension_reference[dim]["point groups"] += [pg]
            if sg not in dim_ref["space groups"]:
                self.__dimension_reference[dim]["space groups"] += [sg]

    def allowed_symmetry(
        self, lattice_type=None, point_group_symbol=None, as_str=False
    ):
        """
        Returns allowed point groups and, optionally, space groups for a given
        lattice type.

        Arguments:
        lattice_type       - str, (default = None), type of Bravais lattice, as
                             per self.lattice_list;
        point_group_symbol - str, (default = None), Schoefiles point group
                             symbol;
        as_str             - bool, (default = False), return output as a
                             string message.

        Returns:
        allowed_groups - dict PG : SG (point_group_symbol = None), allowed
                         point and space groups for a given lattice type;

                         dict lat : SG (lattice_type = None), allowed
                         lattices and space groups for a given point group;

                         list (lattice_type != None,
                               point_group_symbol != None), allowed space
                         groups for a given lattice type and point group;

                         str (as_str = True), string message with allowed
                         point/space groups;
        """
        check_type("lattice_type", lattice_type, str, NoneType)
        check_type("point_group_symbol", point_group_symbol, str, NoneType)
        check_type("as_str", as_str, bool)

        # Neither lattice type nor point group symbol are specified
        if is_None(lattice_type) and is_None(point_group_symbol):
            warnings.warn(
                "No constraints for lattice type or point group "
                "are specified"
            )
            allowed_groups = []

            for i in range(1, 4):
                allowed_groups += [
                    [(i, j) for j in range(1, self.n_space_groups[i])]
                ]

            output_type = 0

        # Both lattice type and point group symbol are given
        elif not_None(lattice_type) and not_None(point_group_symbol):
            self.check_lattice_type(lattice_type)

            data = self.lattice_reference[lattice_type]["groups"]
            allowed_pg = {k: data[k] for k in list(data.keys())}

            check_in_list(
                "point_group_symbol", point_group_symbol, allowed_pg.keys()
            )

            allowed_groups = allowed_pg[point_group_symbol]

            output_type = 1

        # Only lattice type is specified
        elif not_None(lattice_type) and is_None(point_group_symbol):
            self.check_lattice_type(lattice_type)

            data = self.lattice_reference[lattice_type]["groups"]
            allowed_groups = {k: data[k] for k in list(data.keys())}

            output_type = 2

        # Only point_group is specified
        elif is_None(lattice_type) and not_None(point_group_symbol):
            self.check_point_group_symbol(point_group_symbol)

            allowed_lattices = self.point_group_reference[point_group_symbol]

            allowed_groups = [{}, {}, {}]
            for lat in allowed_lattices.keys():
                dim = self.lattice_reference[lat]["dimension"] - 1

                sg_list = []
                for sg_t in allowed_lattices[lat]:
                    sg_list += [sg_t[1]]

                if lat not in allowed_groups[dim].keys():

                    allowed_groups[dim][lat] = sg_list
                else:
                    allowed_groups[dim][lat] += sg_list

            output_type = 3

        if as_str:
            if output_type == 0:
                m = "All {n}D space groups with indices 1 - {NS} are allowed."
                message = [
                    m.format(n=dim, NS=self.n_space_groups[dim])
                    for dim in range(1, 4)
                ]

            elif output_type == 1:
                message = (
                    "The allowed space groups for lattice type "
                    f"{lattice_type} and point group "
                    f"{point_group_symbol} are:\n"
                )
                for sg in allowed_groups:
                    message += f"{sg}  "

            elif output_type == 2:
                message = (
                    "The allowed point groups for lattice type "
                    f"{lattice_type} are:\n"
                )
                for pg in allowed_groups.keys():
                    message += f"{pg} with space groups "
                    for sg in allowed_groups[pg]:
                        message += f"{sg}  "
                    message += "\n"

            elif output_type == 3:
                message = []
                for dim in range(3):
                    if len(allowed_groups[dim].keys()) == 0:
                        m = ""
                    else:
                        m = (
                            "the allowed lattice types for point group "
                            f"{point_group_symbol} are:\n"
                        )
                        for lat in allowed_groups[dim].keys():
                            m += f"{lat} with space groups "
                            for sg in allowed_groups[dim][lat]:
                                m += f"{sg}  "
                            m += "\n"
                    message += [m]

            return message

        else:
            return allowed_groups

    def find_space_group(
        self,
        dimension=None,
        point_group_symbol=None,
        lattice_type=None,
        as_str=False,
    ):
        """
        Return all possible space groups for a given set of constraints.

        Optional arguments:
        dimension           - int (= 1, 2, or 3), dimension of the lattice;
        point_group_symbol  - str, Schoenflies symbol of the crystallographic
                              point group;
        lattice_type        - str, name of the lattice, as per
                              self.lattice_list;
        as_str              - bool, (default = False), return output as a
                              string message.

        Returns:
        space_group_list - list (as_str = False), list of allowed space group
                           indices satisfying the constraints;
                           str (as_str = True), string message with allowed
                           point/space groups;
                           Exception, if constraints can't be satisfied.
        """

        # Type checks
        check_type("dimension", dimension, int, NoneType)
        check_type("point_group_symbol", point_group_symbol, str, NoneType)
        check_type("lattice_type", lattice_type, str, NoneType)

        if not_None(dimension):
            self.check_dimension(dimension)

            allowed_pg = self.dimension_reference[dimension]["point groups"]
            allowed_lat = self.dimension_reference[dimension]["lattices"]

            if not_None(lattice_type):
                if lattice_type not in allowed_lat:
                    raise Exception(
                        f"In {dimension}D, the allowed lattice "
                        f"types are {allowed_lat}, "
                        f"not {lattice_type}"
                    )

            if not_None(point_group_symbol):
                if point_group_symbol not in allowed_pg:
                    raise Exception(
                        f"In {dimension}D, the allowed point "
                        f"groups are {allowed_pg}, "
                        f"not {point_group_symbol}"
                    )

            allowed_groups = self.allowed_symmetry(
                lattice_type, point_group_symbol, as_str
            )

            if isinstance(allowed_groups, list):
                allowed_groups = allowed_groups[dimension - 1]

                cond = not_None(lattice_type) or not_None(point_group_symbol)
                if as_str and cond:
                    allowed_groups = f"In {dimension}D, " + allowed_groups

        else:
            allowed_groups = self.allowed_symmetry(
                lattice_type, point_group_symbol, as_str
            )
            if isinstance(allowed_groups, list):
                if len(allowed_groups) == 3:
                    if as_str:
                        message = ""
                        for dim in range(1, 4):
                            if len(allowed_groups[dim - 1]) != 0:
                                message += f"In {dim}D, "
                                message += allowed_groups[dim - 1] + "\n"
                        allowed_groups = message

        return allowed_groups

    # Methods for checking the values of the space group input
    def check_dimension(self, dimension):
        if dimension < 1 or dimension > 3:
            raise ValueError(
                "Lattice dimension must be 1, 2, or 3, not " f"{dimension}"
            )

    def check_lattice_type(self, lattice_type):
        if lattice_type not in self.lattice_list:
            raise ValueError(
                "Lattice type must be one of "
                f"{self.lattice_list}, not "
                f"{lattice_type}"
            )

    def check_point_group_symbol(self, point_group_symbol):
        if point_group_symbol not in self.point_group_list:
            raise ValueError(
                "Point group symbol must be one of "
                f"{self.point_group_list}, not "
                f"{point_group_symbol}"
            )

    def check_space_group_index(self, space_group_index, dimension):

        max_index = self.n_space_groups[dimension]

        if space_group_index < 1 or space_group_index > max_index:
            raise ValueError(
                "Space group index must be between 1 and 2 "
                f"for a 1D lattice, not {space_group_index}"
            )

    @property
    def lattice_list(self):
        return self.__lattice_list

    @property
    def lattice_reference(self):
        return self.__lattice_reference

    @property
    def point_group_list(self):
        return self.__point_group_list

    @property
    def point_group_reference(self):
        return self.__point_group_reference

    @property
    def space_group_reference(self):
        return self.__space_group_reference

    @property
    def space_group_generators(self):
        return self.__space_group_generators

    @property
    def dimension_reference(self):
        return self.__dimension_reference

    @property
    def n_space_groups(self):
        return self.__n_space_groups
