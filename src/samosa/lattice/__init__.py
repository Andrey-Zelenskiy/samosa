#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
==============
samosa/lattice
==============

This submodule defines Lattice class, which contains methods for working with
space group symmetries.
"""

import numpy as np

from samosa.symmetry.group_utils import Group

from samosa.symmetry.representations import SpaceGroupElement

from samosa.symmetry.space_group_utils import space_group

from samosa.database import SpaceGroupDatabase


from samosa.utils.type_checks import ArrayType, NoneType, is_None, not_None, \
                                     check_type, check_len, check_in_list

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
    lattice. 
    """

    def __init__(self, 
                 dimension, 
                 space_group_index, 
                 positions=[[0, 0, 0]],
                 crystal_parameters=None):
        """
        Initialize all relevant space group information from the lattice 
        dimension and space group number. 

        Arguments:
        dimension         - int, dimension of the lattice;

        space_group_index - int, space group number (as per International 
                            Tables for Crystallography, volume A);
        
        positions         - list of ArrayType[3], (default=[[0, 0, 0]]), vertex
                            (Wyckoff) positions written in fractional 
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
        check_type('dimension', dimension, int)
        check_type('space_group_index', space_group_index, int)
        check_type('positions', positions, list) 

        for p in positions:
            check_type('vertex position', p, ArrayType)

        check_type('crystal_parameters', crystal_parameters, NoneType, dict)

        # Initialize symmetry properties
        sg_info = space_group(dimension, space_group_index, return_info=True)
        
        self.__dimension = dimension
        self.__space_group_index = space_group_index
        self.__space_group = sg_info['space_group']
        self.__point_group_symbol = sg_info['point_group_symbol']
        self.__lattice_type = sg_info['lattice_type']
        self.__symmorphic = sg_info['symmorphic']

        # Initialize structural properties
        self.__initialize_crystal_parameters(crystal_parameters)

        self.__wyckoff_list = []
        for p in positions:
            wp = WyckoffPosition(p, self.space_group)
            positions = [p for p in positions if p not in wp.positions]
            self.__wyckoff_list += [wp]

    def __initialize_crystal_parameters(self, parameters=None):
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
                     initializes parameters from user input.
        """
        # If no input crystal parameters are provided, initialize them randomly
        if is_None(parameters):
            parameters = {}

            # Normalize dimensions with respect to x-direction
            parameters['a'] = 1.0

            for p in ['b', 'c']:
                # Initialize random dimension values between 0 and 5
                parameters[p] = np.random.rand()*5.0 

            for p in ['alpha', 'beta', 'gamma']:
                # Initialize random angle values between 0 and pi
                parameters[p] = np.random.rand()*np.pi

        # Determine constraints on the physical crystal parameters from the
        # lattice type
        crystal_family = self.lattice_type.split('_')[0]
        
        self.__crystal_parameters = {}
        try:
            # 1D crystal family
            if crystal_family == 'chain':
                
                unconstrained_parameters = ['a']
                self.__crystal_constraints = 'a'

                self.__crystal_parameters['a'] = parameters['a']

            # 2D crystal families
            elif crystal_family == 'oblique':

                unconstrained_parameters = ['a', 'b', 'gamma']
                self.__crystal_constraints = 'a, b, gamma'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['b']
                
                self.__crystal_parameters['gamma'] = parameters['gamma']
        
            elif crystal_family == 'rectangular':

                unconstrained_parameters = ['a', 'b']
                self.__crystal_constraints = 'a, b, gamma = 90'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['b']
                
                self.__crystal_parameters['gamma'] = np.pi/2
        
            elif crystal_family == 'square':

                unconstrained_parameters = ['a']
                self.__crystal_constraints = 'a = b, gamma = 90'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['a']
                
                self.__crystal_parameters['gamma'] = np.pi/2
        
            elif crystal_family == 'hexagonal':

                unconstrained_parameters = ['a']
                self.__crystal_constraints = 'a = b, gamma = 120'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['a']
                
                self.__crystal_parameters['gamma'] = 2*np.pi/3
        
            # 3D crystal families
            elif crystal_family == 'Triclinic':

                unconstrained_parameters = ['a', 'b', 'c', 
                                            'alpha', 'beta', 'gamma']
                self.__crystal_constraints = 'a, b, c, alpha, beta, gamma'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['b']
                self.__crystal_parameters['c'] = parameters['c']
                
                self.__crystal_parameters['alpha'] = parameters['alpha']
                self.__crystal_parameters['beta']  = parameters['beta']
                self.__crystal_parameters['gamma'] = parameters['gamma']
        
            elif crystal_family == 'Monoclinic':

                unconstrained_parameters = ['a', 'b', 'c', 'beta']
                self.__crystal_constraints = 'a, b, c, alpha = gamma = 90, '\
                                             'beta'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['b']
                self.__crystal_parameters['c'] = parameters['c']
                
                self.__crystal_parameters['alpha'] = np.pi/2
                self.__crystal_parameters['beta']  = parameters['beta']
                self.__crystal_parameters['gamma'] = np.pi/2
        
            elif crystal_family == 'Orthorombic':

                unconstrained_parameters = ['a', 'b', 'c']
                self.__crystal_constraints = 'a, b, c, alpha = beta = gamma '\
                                             '= 90, '

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['b']
                self.__crystal_parameters['c'] = parameters['c']
                
                self.__crystal_parameters['alpha'] = np.pi/2
                self.__crystal_parameters['beta']  = np.pi/2
                self.__crystal_parameters['gamma'] = np.pi/2
        
            elif crystal_family == 'Cubic':

                unconstrained_parameters = ['a']
                self.__crystal_constraints = 'a = b = c, alpha = beta = gamma'\
                                             ' = 90'

                self.__crystal_parameters['a'] = parameters['a']
                self.__crystal_parameters['b'] = parameters['a']
                self.__crystal_parameters['c'] = parameters['a']
                
                self.__crystal_parameters['alpha'] = np.pi/2
                self.__crystal_parameters['beta']  = np.pi/2
                self.__crystal_parameters['gamma'] = np.pi/2
        
        except KeyError:
            raise Exception(f'{self.lattice_type} lattice type requires '
                            f'definition of {unconstrained_parameters} '
                            f'crystal parameters.')

            for p in unconstrained_parameters:
                self.__crystal_parameters[p] = crystal_parameters[p]

        # Initialize basis vectors
        self.basis_vectors = self.crystal_parameters['a']

        if self.dimension > 1: 
            gamma = self.crystal_parameters['gamma']
            self.__basis_vectors += [np.array([np.cos(gamma), np.sin(gamma)])]


    # Lattice methods
    def set_units(self, a=None, b=None, c=None, 
                        alpha=None, beta=None, gamma=None):
        """
        Primitive lattice cell parameters (dimensions and angles).
        
        Arguments:
        a, b, c            - floats, lengths of the primitive lattice vectors;

        alpha, beta, gamma - floats, angles between the primitive lattice 
                             vectors.
        """
        # Type checks

        if self.


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
    def wyckoff_list(self):
        """
        All vertex positions in a single unit cell, sorted into non-equivalent
        Wyckoff classes.
        """
        return self.__wyckoff_list

    # Output summary functions
    def __str__(self):
        """
        Provides user-friendly summary of the Lattice container.
        """

        summary_string = "Lattice object.\n\n"
        
        summary_string += f"Lattice type: {self.lattice_type};\n"
        
        summary_string += f"Crystallographic point group: "\
                          f"{self.point_group_symbol};\n"
        
        summary_string += f"Space group index (ITC vol.A): "\
                          f"{self.space_group_index};"
        
        return summary_string

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(dimension={self.dimension}, "\
               f"space_group_index={self.space_group_index})"

"""
-------------------------------------------------------------------------------
WyckoffPosition class
-------------------------------------------------------------------------------
"""

class WyckoffPosition:
    """
    Defines a container for storing all properties of a speciefied Wyckoff
    site. 
    """

    def __init__(self, position, space_group):
        """
        Initializes the properties related to the Wyckoff position.

        Arguments:
        position    - ArrayType, fractional positions of a single vertex;

        space_group - Group, space group of the lattice.
        """
        # Type checks
        check_type('position', position, ArrayType)
        check_len('position', position, 3)
        check_type('space_group', space_group, Group)

        # Store original space group generators
        self.__generators_group = space_group.generators

        # Calculate orbit
        output = space_group.calculate_orbit(position, as_pointer=True)
        self.__positions = output[0]
        self.__generators_wyckoff = output[1]
        self.__transporter_pointers = output[2]
        self.__generators_pg = output[3]

    # WyckoffPosition properties
    @property
    def positions(self):
        """
        Orbit of the input position (vertices related by crystallographic point
        group operations).
        """
        return self.__positions

    @property
    def multiplicity(self):
        """
        Returns the total number of Wyckoff positions.
        """
        return len(self.positions)

    @property
    def generators(self):
        """
        Generators of the space group written as permutations of the Wyckoff 
        positions.
        """
        return self.__generators_wyckoff

    def transporters(self, as_permutations=False):
        """
        Space group transformations that move the selected vertex to other
        positions in the orbit. 

        Arguments:
        as_permutations - bool, (default=False), if True, returns the 
                          transporters as permutations of the Wyckoff
                          positions; 
                          otherwise, returns transporters in a
                          SpaceGroupElement representation.

        Returns:
        transporter_dict - dict {(position) : transporter}, transporter
                           operators that change the initial vertex to another
                           vertex in the orbit.
        """
        if as_permutations:
            generators = self.generators
        else:
            generators = self.__generators_group

        transporter_dict = {}
        for p in self.__transporter_pointers.keys():
            pointer = self.__transporter_pointers[p]
            element = pointer.pointer_to_element(generators, skip_checks=True)
            transporter_dict[p] = element

        return transporter_dict

    def point_group(self, as_permutations=False):
        """
        Stabilizer point group of a single Wyckoff position. 

        Arguments:
        as_permutations - bool, (default=False), if True, returns the 
                          Group object generated by permutations of the Wyckoff
                          positions; 
                          otherwise, returns Group wit henerators in a
                          MatrixGroupElement representation.

        Returns:
        PointGroup - dict {(position) : transporter}, transporter operators
                     that change the initial vertex to another vertex in the
                     orbit.
        """
        if as_permutations:
            generators = self.generators
        else:
            generators = [g.matrix for g in self.__generators_group]

        return Group(generators, filter_generators=True)

    # Output summary functions
    def __str__(self):
        """
        Provides user-friendly summary of the WyckoffPosition container.
        """

        summary_string = f"Wyckoff positions with multiplicity "\
                         f"{self.multiplicity}\n"
        
        summary_string += f"{self.positions}"
        
        return summary_string

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(position={self.positions[0]})"

