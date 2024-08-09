#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
lattice_utils.py
=================

This submodule defines Lattice class, which contains methods for working with
space group symmetries.
"""

import numpy as np

from samosa.symmetry.group_utils import Group, MatrixGroupElement
from samosa.symmetry.point_group_utils import point_group

from samosa.database.database_utils import SpaceGroupDatabase


from samosa.api.api_utils import array_type, NoneType 
from samosa.api.api_utils import is_None, not_None
from samosa.api.api_utils import check_type, check_in_list
from samosa.api.api_utils import custom_format_warning 

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
    lattice. 
    """

    def __init__(self, dimension = None, space_group = None, 
                       point_group = None, lattice_type = None):
        """
        """

        self.__database = SpaceGroupDatabase()

        self.dimension = dimension
        
        if self.dimension != 1: 
            self.space_group = space_group

            if is_None(self.space_group):
                self.point_group = point_group
                self.lattice_type = lattice_type


    def update_symmetry(self, dimension = None, space_group_index = None, 
                              point_group_symbol = None, lattice_type = None):
        """
        Checks if the input symmetry properties yield a correct space group
        symmetry.

        Optional arguments:
        dimension           - int (= 1, 2, or 3), dimension of the lattice;
        space_group_index   - int, index of the space group as per International
                              Tables for Crystallography;
        point_group_symbol  - str, Schoenflies symbol of the crystallographic
                              point group;
        lattice_type        - str, name of the lattice, as per 
                              self.__database.lattice_list.

        Returns:
        None if no contridictions arise, otherwise raises an Exception.
        """

        # Type checks
        check_type('dimension',dimension,int,NoneType)
        check_type('space_group_index',space_group_index,int,NoneType)
        check_type('point_group_symbol',point_group_symbol,str,NoneType)
        check_type('lattice_type',lattice_type,str,NoneType)
    
        # Replace missing arguments with class members
        if is_None(dimension):
            dimension = self.dimension

        if is_None(space_group_index):
            space_group_index = self.space_group_index
        
        if is_None(point_group_symbol):
            point_group_symbol = self.point_group_symbol
        
        if is_None(lattice_type):
            lattice_type = self.lattice_type

        # Attempt to create a (dimension, space_group_index) tuple
        if is_None(dimension) and not_None(space_group_index):
            warnings.warn("Lattice dimension not set: assuming 3D.")
            dimension = 3

        if not_None(dimension) and not_None(space_group_index):
            self.__check_dimension(dimension)
            self.__check_space_group_index(space_group_index, dimension)

            sg_t = ( dimension, space_group_index )
            pg, lat = self.__database.space_group_reference[sg_t]

            if is_None(point_group_symbol):
                point_group_symbol = pg

            if is_None(lattice_type):
                lattice_type = lat

            if point_group_symbol != pg or lattice_type != lat:
                raise ValueError(f"{dimension}D space group with index "
                                 f"{space_group_index} must have point group "
                                 f"{pg} and lattice type {lat}, not "
                                 f"{point_group_symbol} and {lattice_type}.")

        # Space group not set
        else:
            if not_None(dimension):

            

            # Attempt to set lattice dimension based on lattice type




            if not_None(lattice_type):
                self.__check_lattice_type(lattice_type)

                lat_data = self.__database.lattice_reference[lattice_type]
                dim = lat_data["dimension"]
                pg_list = lat_data.keys()[1:]

                if is_None(dimension):
                    dimension = dim

                if dimension != dim:
                    raise ValueError(f"{lattice_type} lattice must have "
                                     f"dimension {dim}, not{dimension}.")

                if not_None(point_group_symbol):
                    if point_group_symbol not in pg_list:
                        raise ValueError(f"{lattice_type} lattice must have "
                                         f"a point group in {pg_list}, "
                                         f"not {point_group_symbol}.")
                    
                    #TODO This is incorrect, we need to output only the allowed
                    #TODO space groups, not just the point groups
                    message = self.__database.allowed_point_groups(lattice_type,
                                                               as_str=True,
                                                               include_sg=False)
                    warnings.warn("Space group is not set.\n"
                                 f"{message}")

                else:
                    message = self.__database.allowed_point_groups(lattice_type,
                                                               as_str=True,
                                                               include_sg=True)
                    warnings.warn("Space group and point group are not set.\n"
                                 f"{message}")

            else:
                # Space group and lattice type are not provided, attempt to 
                # extract information from the point group

                if not_None(point_group_symbol):
                    pg_data = self.__database.point_group_reference[pg]
                    
                    dim_list = []

                    for lat in pg_data.keys():
                        for sg_t in pg_data[lat]:
                            dim = sg_t[0]

                            if dim not in dim_list:
                                dim_list += [dim]

                    if is_None(dimension):
                        if len(dim_list) == 1:
                            dimension = dim_list[0]
                    
                    else:
                        if dimension not in dim_list:
                            raise ValueError(f"{point_group_symbol} point "
                                             f"group is incompatible with a "
                                             f"{dimension}D lattice.")

                        if len(dim_list) > 1:
                        #TODO determine the allowed lattice types and space
                        #TODO groups based on the point group and dimension 

                    
                
                else:
                    warnings.warn("No symmetry data provided!")


    # Methods for checking the values of the space group input
    def __check_dimension(self, dimension):
        if dimension < 1 or dimension > 3:
            raise ValueError("Lattice dimension must be 1, 2, or 3, not "
                             "{}".format(dimension))
    

    def __check_space_group_index(self, space_group_index, dimension): 
        if dimension == 1:
            if space_group_index < 1 or space_group_index > 2:
                raise ValueError("Space group index must be between 1 and 2 "
                                 "for a 1D lattice, not "
                                 "{}".format(space_group_index)) 
        
        elif dimension == 2:
            if space_group_index < 1 or space_group_index > 17:
                raise ValueError("Space group index must be between 1 and 17 "
                                 "for a 2D lattice, not "
                                 "{}".format(space_group_index)) 
        
         elif dimension == 3: 
            if space_group_index < 1 or space_group_index > 230:
                raise ValueError("Space group index must be between 1 and 230 "
                                 "for a 3D lattice, not "
                                 "{}".format(space_group_index)) 
    
         else: 
            if space_group_index < 1 or space_group_index > 230:
                raise ValueError("Space group index must be between 1 and 230, "
                                 "not {}".format(space_group_index)) 
    

    def __check_lattice_type(self, lattice_type):
        if lattice_type not in self.__database.lattice_list:
            raise ValueError("Lattice type must be one of "
                             "{}, not {}".format(self.__database.lattice_list,
                                                 lattice_type))


    @property
    def dimension(self):
        return self.__dimension

    
    @dimension.setter
    def dimension(self, val):
        # Type check
        check_type('dimension',val,int,NoneType)
        
        if not_None(val):
            if val == 1:
                self.__dimension = 1
                self.__space_group = 1
                self.__lattice_type = 'chain'

            elif val == 2 or val == 3:
                self.__dimension = val

            else:
                raise ValueError(f"Invalid lattice dimension: {val}")


    @dimension.deleter
    def dimension(self):
        self.__dimension = None


    @property
    def space_group_index(self):
        pass

    
    @property
    def point_group_symbol(self):
        pass

    
    @property
    def lattice_type(self):
        pass


