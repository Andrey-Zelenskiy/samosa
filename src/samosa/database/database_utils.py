#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
database_utils.py
=================

This module provides tools to translate symmetry information contained in
samosa/database to symmetry objects defined in group_utils.py.
"""

import numpy as np
import toml
import os
from pathlib import Path, PosixPath

from samosa.api.api_utils import array_type, NoneType 
from samosa.api.api_utils import not_None, check_type, check_len, check_shape
from samosa.api.api_utils import custom_format_warning 

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
        self.__path = Path(__file__).parent 
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
            check_type('file',f,int,str,PosixPath)
            
            if isinstance(f,int):
                file = self.available_files[f]
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
        return self.__file_index


    @property
    def file_list(self):
        """
        List of uploaded files
        """
        return self.__file_list


    @property
    def data(self):
        """
        Uploaded data
        """
        return self.__data


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
        self.load_file(self.path_to_database / "space_group_lattices.toml",
                       self.path_to_database / "space_group_generators.toml")

        self.get_lookup_dicts()
        

    def get_lookup_dicts(self):
        """
        Populates lookup dictionaries.
        """

        # Lookup tables for space group - point group - lattice relationships
        data = self.data['space_group_lattices.toml']

        self.__lattice_list = list(data.keys())
        
        self.__lattice_reference = {} # { lattice : 
                                      #   [ { PG : [( dim, SG ), ] }, ] }
        
        self.__point_group_reference = {}  # { PG : 
                                           # [ { lattice : 
                                           #   [ ( dim, SG ), ] }, ] }
        
        self.__space_group_reference = {}  # { ( dim, SG ) : ( PG, lattice ) }

        for lat in self.__lattice_list:
            dim = data[lat]['dimension']
            pg_list = list(data[lat].keys())[1:]
            
            for pg in pg_list:
                sg_list = data[lat][pg]['space_groups']
                
                for sg in sg_list:
                    # 'space_group_lattices.toml' stores 2d space group numbers
                    # as negative numbers. Instead, by creating a tuple 
                    # (dimension, space group number) we can remove ambiguity
                    # between 3d and 2d space groups and avoid negative numbers
                    sg_t = ( dim, abs(sg) )

                    if lat not in self.__lattice_reference.keys():
                        self.__lattice_reference[lat] = { pg : [ sg_t ] }
                    else:
                        if pg not in self.__lattice_reference[lat].keys():
                            self.__lattice_reference[lat][pg] = [ sg_t ]
                        else:
                            self.__lattice_reference[lat][pg] += [ sg_t ]


                    if pg not in self.__point_group_reference.keys():
                        self.__point_group_reference[pg] = { lat : [ sg_t ] }
                    else:
                        if lat not in self.__point_group_reference[pg].keys():
                            self.__point_group_reference[pg][lat] = [ sg_t ]
                        else:
                            self.__point_group_reference[pg][lat] += [ sg_t ]


                    if sg_t not in self.__space_group_reference.keys():
                        self.__space_group_reference[sg_t] = ( pg, lat )
                    else:
                        self.__space_group_reference[sg_t] += ( pg, lat )

        self.__point_group_list = list(self.__point_group_reference.keys())

        # Separate the generators in a separate dict
        data = self.data['space_group_generators.toml']
        self.__space_group_generators = {}

        for sg in data.keys():
            sg_t = tuple(int(n) for n in sg.split(','))
            self.__space_group_generators[sg_t] = data[sg]
    

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
