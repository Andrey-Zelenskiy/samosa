#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
=================
representations.py
=================

This submodule defines objects that correspond to some common representations
of group elements, including matrices, permutations, and space group operators.
Two additional classes define the universal identity element and pointers to
the group elements in a group with known generators.
"""

import numpy as np

from samosa.api.api_utils import ArrayType, NoneType
from samosa.api.api_utils import is_None, not_None
from samosa.api.api_utils import check_type, check_len, check_shape
from samosa.api.api_utils import custom_format_warning

import warnings
warnings.formatwarning = custom_format_warning
"""
-------------------------------------------------------------------------------
Interface for group elements representation
-------------------------------------------------------------------------------
"""


class GroupElement:
    """
    Interface class for a group element representation.
    """

    def __init__(self):
        self.__dim = None
        self.__is_identity = None

    @property
    def dim(self):
        """
        Returns the dimensionality of the basis of group element
        representation.
        """
        return self.__dim

    @dim.setter
    def dim(self, val):
        if not_None(self.dim):
            raise Exception("dim cannot be modified!")
        else:
            check_type('dim', val, int, np.int64, NoneType)
            self.__dim = val

    @dim.deleter
    def dim(self):
        self.__dim = None

    @property
    def is_identity(self):
        """
        Returns True if the group element represents identity, 
        otherewise returns False.
        """
        return self.__is_identity

    @is_identity.setter
    def is_identity(self, val):
        if not_None(self.__is_identity):
            raise Exception("is_identity cannot be modified!")
        else:
            check_type('is_identity', val, bool, np.bool_, NoneType)
            self.__is_identity = val

    @is_identity.deleter
    def is_identity(self):
        self.__is_identity = None

    def __mul__(self, element):
        """
        Left group element multiplication and group action.
        """
        pass

    def __rmul__(self, element):
        """
        Right group element multiplication and group action.
        """
        pass

    def __eq__(self, element):
        """
        Group element comparison.
        """
        pass


"""
-------------------------------------------------------------------------------
Universal identity element
-------------------------------------------------------------------------------
"""


class IdentityGroupElement(GroupElement):
    """
    An object used to represent a universal identity element.
    """

    # Object initialization
    def __init__(self, dim=None):
        """
        If given, initializes the dimension/trace of the identity.
        """
        GroupElement.__init__(self)

        self.dim = dim
        self.is_identity = True

    # Common operations
    @property
    def inv(self):
        """
        Returns itself.
        """
        return self

    @property
    def trace(self):
        """
        Returns the dimension of the identity.
        """
        if is_None(self.dim):
            raise ValueError("Cannot calculate trace for an "
                             "IdentityGroupElement with unknown "
                             "dimensionality")
        return self.dim

    def __mul__(self, element):
        """
        Shortcut for calculating (left) action.
        """
        return element

    def __rmul__(self, element):
        """
        Shortcut for calculating (right) action.
        """
        return element

    def __eq__(self, element):
        """
        Determines if two matrix group elements are the same up to numerical
        precision.
        """
        if isinstance(element, IdentityGroupElement):
            return element.dim == self.dim
        else:
            return False

    # Output summary functions
    def __str__(self):
        """
        User-friendly output of the group element.
        """
        cls = self.__class__.__name__
        if is_None(self.dim):
            return f"{cls} with undefined dimension"

        else:
            return f"{cls} with dimension {self.dim}" 

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(dim = {self.dim})"


"""
-------------------------------------------------------------------------------
Matrix representation of group elements
-------------------------------------------------------------------------------
"""


class MatrixGroupElement(GroupElement):
    """
    Representation of a group element as a matrix operator with strict
    multiplication and inversion rules.
    """

    # Object initialization
    def __init__(self, operator):
        """
        Initializes the matrix operator of the group representation.

        Arguments:
        operator - 2dArrayType, matrix operator defining the group element or 
                   IdentityGroupElement, defines an identity element.
        """

        GroupElement.__init__(self)

        # Numerical precision for rounding error
        self.__eps_r = 1e-10
        self.__log_eps_r = 10

        # Numerical precision for printing
        self.__log_eps_p = 4

        # By default, all object's properties are calculated from the matrix
        # operator
        self.args_calculated = True
        
        # Initialize the properties
        self.__operator = None
        self.__cycle_order = None
        self.__orthogonal_basis = None
        self.__operator_inverse = None
        self.__trace = None

        # Type checks
        check_type('operator', operator, ArrayType, IdentityGroupElement)

        if isinstance(operator, IdentityGroupElement):
            if operator.dim == None:
                raise ValueError("Cannot initialize MatrixGroupElement "
                                 "from IdentityGroupElement with unknown "
                                 "dimension.")

            self.dim = operator.dim
            self.is_identity = True
            self.__operator = np.eye(self.dim)
            self.__cycle_order = 1
            self.__orthogonal_basis = True
            self.__operator_inverse = None
            self.__trace = self.dim

        else:
            operator = np.round(np.array(operator), self.__log_eps_r)
            self.dim = len(operator)

            check_shape('operator', operator, self.dim, self.dim)
            
            self.__operator = operator
            self.is_identity = _array_equal(self.operator,
                                            np.eye(self.dim),
                                            self.__eps_r)

    @classmethod
    def input_args(cls, 
                   operator,
                   cycle_order = None,
                   orthogonal_basis = None,
                   operator_inverse = None,
                   trace = None):
        """
        Initializes the properties of the MatrixGroupElement from user input.
        
        WARNING: this initialization method does not check the input values,
        which could lead to incompatible properties. Only use this method 
        when you are certain about the input!

        Arguments:
        operator         - 2dArrayType, matrix operator defining the group
                           element or
                           IdentityGroupElement, defines an identity element;

        cycle_order      - int, (default=None), gives the cycle order of the
                           operator, i.e. n for which operator^n = identity;

        orthogonal_basis - bool or np.bool_, (default=None) True if the basis
                           of the matrix operator is orthonormal, otherwise
                           False;

        operator_inverse - 2dArrayType, (default=None), proposes an inverse
                           of operator;

        trace            - float, (default=None), trace of the matrix 
                           operator.

        Returns:
        MatrixGroupElement object.
        """
        element = cls(operator)

        if isinstance(operator, IdentityGroupElement):
            raise Exception("Please use "
                            "MatrixGroupElement(IdentityGroupElement(dim)) "
                            "as the initializer.")

        element.args_calculated = False

        elements.cycle_order = cycle_order
        elements.orthogonal_basis = orthogonal_basis
        elements.inv = operator_inverse
        elements.trace = trace

        return element

    # Object's properties
    @property
    def operator(self):
        """
        Returns matrix operator.
        """
        return self.__operator

    @property
    def cycle_order(self):
        """
        Returns the cycle order of the matrix operator 
        (n for which operator^n = identity).
        """
        if is_None():
            self.__cycle_order = self.get_cycle_order(self.operator, 
                                                            self.__eps_r)

        return self.__cycle_order

    @cycle_order.setter
    def cycle_order(self, val):
        if self.args_calculated:
            raise Exception("cycle_order cannot be modified!")
        else:
            check_type('cycle_order', val, int, np.int64, NoneType)
            self.__cycle_order = val

    @cycle_order.deleter
    def cycle_order(self):
        self.__cycle_order = None

    @property
    def orthogonal_basis(self):
        """
        Returns True if the matrix operator is defined in the orthogonal
        basis, otherwise returns False.
        """
        if is_None(self.__orthogonal_basis):
            self.__orthogonal_basis = _check_orthogonal(self.operator)

        return self.__orthogonal_basis

    @orthogonal_basis.setter
    def orthogonal_basis(self, val):
        if self.args_calculated:
            raise Exception("orthogonal_basis cannot be modified!")
        else:
            check_type('orthogonal_basis', val, bool, np.bool_, NoneType)
            self.__orthogonal_basis = val

    @orthogonal_basis.deleter
    def orthogonal_basis(self):
        self.__orthogonal_basis = None

    # Common operations
    @property
    def inv(self):
        """
        Returns the inverse of the matrix group element.
        """
        if is_None(self.__operator_inverse):
            # If the operator is orthogonal, return the transpose
            if self.orthogonal_basis:
                self.__operator_inverse = self.operator.T
            else:
                self.__operator_inverse = np.linalg.inv(self.operator)

        return MatrixGroupElement(self.__operator_inverse)

    @inv.setter
    def inv(self, val):
        if self.args_calculated:
            raise Exception("operator_inverse cannot be modified!")
        else:
            check_type('operator_inverse', val, ArrayType, NoneType)
            operator_inverse = np.round(np.array(val), self.__log_eps_r)
            
            check_shape('operator_inv', operator_inverse, self.dim, self.dim)

            id_test = operator_inverse.dot(self.operator) - np.eye(self.dim)

            if np.max(np.abs(id_test)) > self.__eps_r:
                raise ValueError("Proposed inverse does not produce "\
                                 "identity under multiplication with the "\
                                 "operator.")

            self.__operator_inverse = operator_inverse

    @inv.deleter
    def inv(self):
        self.__operator_inverse = None

    @property
    def trace(self):
        """
        Returns the trace of the matrix operator.
        """
        if is_None(self.__trace):
            self.__trace = np.trace(self.operator)

        return self.__trace

    @trace.setter
    def trace(self, val):
        if self.args_calculated:
            raise Exception("trace cannot be modified!")
        else:
            check_type('trace', val, int, np.int64, NoneType)
            self.__trace = val

    @trace.deleter
    def trace(self):
        self.__trace = None

    def __mul__(self, element):
        """
        Shortcut for calculating (left) group element action.
        """

        # Multiplication of two group elements
        if isinstance(element, MatrixGroupElement):

            if element.dim != self.dim:
                raise TypeError(f"Cannot perform multiplication between "
                                f"matrices of dimension {self.dim} and "
                                f"{element.dim}.")

            product = self.operator.dot(element.operator)

            return MatrixGroupElement(product)

        # Left action on an array
        elif isinstance(element, ArrayType):

            element = np.array(element)

            if element.shape[0] != self.dim:
                raise TypeError(f"Cannot perform multiplication between "
                                f"matrices of dimension {self.dim} and "
                                f"{element.shape[0]}.")

            return self.operator.dot(element)
        
        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{self.__class__.__name__} and "
                            f"{type(element).__name__}.")

    def __rmul__(self, element):
        """
        Shortcut for calculating (right) group element action.
        """

        # Right action on an array
        if isinstance(element, ArrayType):

            element = np.array(element)

            if element.shape[-1] != self.dim:
                raise TypeError(f"Cannot perform multiplication between "
                                f"matrices of dimension {element.shape[-1]}"
                                f"and {self.dim}.")

            return element.dot(self.operator)

        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{type(element).__name__} and "
                            f"{self.__class__.__name__}.")

    def __eq__(self, element):
        """
        Determines if two matrix group elements are the same up to numerical
        precision.
        """

        # Comparison of two MatrixGroupElements
        if isinstance(element, MatrixGroupElement):
            return _array_equal(self.operator, element.operator, self.__eps_r)

        # Comparison with IdentityGroupElement
        elif isinstance(element, IdentityGroupElement):
            return self.is_identity()

        # Any other comparison yields False
        else:
            warnings.warn(f"Comparison of "
                          f"{self.__class__.__name__} with "
                          f"{type(element).__name__} yields False by default.")
            return False

    # Supplementary functions
    @staticmethod
    def get_cycle_order(operator, eps):
        """
        Calculates the cycle of the matrix operator via iterative matrix 
        multiplication.
        """
        matrix = np.copy(operator)
        cycle_order = 1
        
        while not _array_equal(matrix, np.eye(len(operator)), eps):
            matrix = matrix.dot(operator)
            cycle_order += 1

        return cycle_order

    # Output summary functions
    def __str__(self):
        """
        User-friendly output of the group element.
        """
        return str(np.round(self.operator, self.__log_eps_p))

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        operator_p = np.round(self.operator, self.__log_eps_p)
        return f"{cls}(operator={operator_p!r}, "\
               f"args_calculated={self.args_calculated})"


"""
-------------------------------------------------------------------------------
Permutation representation of group elements
-------------------------------------------------------------------------------
"""


class PermutationGroupElement(GroupElement):
    """
    Representation of a group element object as a permutation with strict
    multiplication and inversion rules.
    """

    # Object initialization
    def __init__(self, permutation):
        """
        Initializes the permutation tuple of the group representation.

        Arguments:
        permutation - ArrayType of int, assume map n -> permutation[n], or
                      dict int : int, assume a map n -> m, where n and m are
                      in the same (closed) set, or
                      IdentityGroupElement, return cycle (1, 2, ..., dim).
        """

        GroupElement.__init__(self)
        
        # Initialize the properties
        self.__permutation = None
        self.__permutation_cycles = None
        self.__permutation_dict = None
        self.__cycle_order = None
        self.__permutation_inverse = None
        self.__trace = None

        # Type checks
        check_type('permutation', permutation, dict, ArrayType,
                   IdentityGroupElement)

        if isinstance(permutation, IdentityGroupElement):
            if permutation.dim == None:
                raise ValueError("Cannot initialize PermutationGroupElement "
                                 "from IdentityGroupElement with unknown "
                                 "dimension.")

            self.dim = permutation.dim
            self.is_identity = True
            self.__permutation = tuple(i+1 for i in range(self.dim))
            self.__cycle_order = 1
            self.__permutation_inverse = self.__permutation
            self.__trace = self.dim

        elif isinstance(permutation, dict):
            if min(permutation.keys()) < 1:
                raise ValueError("Permutation values must not be smaller "
                                 "than 1.")

            self.dim = max(permutation.keys())

            try:
                self.__permutation = \
                        tuple(permutation[n+1] for n in range(self.dim))

            except KeyError:
                raise TypeError("Permutation dictionary must contain "
                                "transformations of all points in a closed "
                                "set.")

            test_id = self.permutation == tuple(i+1 for i in range(self.dim))
            self.is_identity = test_id

        else:
            # Type check
            for n in permutation:
                check_type('permutation values', n, int, np.int64)

            if min(permutation) < 1:
                raise ValueError("Permutation values must not be smaller "
                                 "than 1.")

            self.__permutation = tuple(permutation)
            self.dim = len(permutation)

            test_id = self.permutation == tuple(i+1 for i in range(self.dim))
            self.is_identity = test_id

    # Object's properties
    @property
    def permutation(self):
        """
        Returns the permutation tuple.
        """
        return self.__permutation

    @property
    def permutation_cycles(self):
        """
        Returns the permutation in cycle representation.
        """
        if is_None(self.__permutation_cycles):
            self.__permutation_cycles = \
                    self.get_permutation_cycles(self.permutation) 

        return self.__permutation_cycles

    @property
    def permutation_dict(self):
        """
        Returns the permutation in dictionary map (int : int) representation.
        """
        if is_None(self.__permutation_dict):
            self.__permutation_dict = \
                    self.get_permutation_dict(self.permutation)

        return self.__permutation_dict

    @property
    def cycle_order(self):
        """
        Returns the cycle order of the permutation tuple 
        (n for which operator^n = identity).
        """
        if is_None(self.__cycle_order):
            self.__cycle_order = self.get_cycle_order(self.permutation_cycles)

        return self.__cycle_order

    # Common operations
    @property
    def inv(self):
        """
        Returns inverse of self.permutation.
        """
        if is_None(self.__permutation_inverse):
            self.__permutation_inverse = np.zeros(self.dim, int)

            for i in range(self.dim):
                self.__permutation_inverse[self.permutation[i]-1] = i+1

        return PermutationGroupElement(self.__permutation_inverse)

    @property
    def trace(self):
        """
        Returns the trace of the corresponding permutation matrix.
        """
        if is_None(self.__trace):
            self.__trace = 0
            for n in range(self.dim):
                if self.permutation[n] == n+1:
                    self.__trace += 1

        return self.__trace

    def __mul__(self, element):
        """
        Shortcut for calculating (left) group element action.
        """

        # Multiplication of two group elements
        if isinstance(element, PermutationGroupElement):

            if element.dim != self.dim:
                raise TypeError(f"Cannot perform multiplication between "\
                                f"permutations of dimension "
                                f"{self.dim} and {element.dim}.")

            product = tuple(self.permutation[p-1] for p in element.permutation)

            return PermutationGroupElement(product)

        # Left action on an array
        elif isinstance(element, ArrayType):

            if len(element) < self.dim:
                raise Exception(f"Array must be of length at least {self.dim} "
                                f"in order to permute its elements.")

            new_element = np.array(element)

            for i in range(self.dim):
                new_element[i] = element[self.permutation[i]-1]

            if isinstance(element, np.ndarray):
                return new_element

            else:
                return type(element)(new_element)

        # Left action on int
        elif isinstance(element, int):
            if element > 0 and element <= self.dim:
                return self.permutation[element-1]
            else:
                return element

        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{self.__class__.__name__} and "
                            f"{type(element).__name__}.")

    def __rmul__(self, element):
        """
        Right permutation action is only defined for Identity.
        """

        raise TypeError(f"Cannot multiply objects of type "
                        f"{type(element).__name__} and "
                        f"{self.__class__.__name__}.")

    def __eq__(self, element):
        """
        Determines if two permutations are the same.
        """

        # Comparison of two PermutationGroupElements
        if isinstance(element, PermutationGroupElement):
            return self.permutation == element.permutation

        # Comparison with IdentityGroupElement
        elif isinstance(element, IdentityGroupElement):
            return self.is_identity()

        # Any other comparison yields False
        else:
            warnings.warn(f"Comparison of "
                          f"{self.__class__.__name__} with "
                          f"{type(element).__name__} yields False by default.")
            return False

    # Supplementary functions
    @staticmethod
    def get_permutation_cycles(permutation):
        """
        Returns the permutation in cycle representation.
        """
        # Type check
        check_type('permutation', permutation, tuple)
        for n in permutation:
            check_type('permutation values', n, int, np.int64)

        permutation_list = list(permutation)
        permutation_cycles = []
        for n in range(len(permutation_list)):
            if permutation_list[n] not in (0, n+1):
                x = n + 1
                cycle = []
                while True:
                    cycle.append(x)
                    permutation_list[x-1], x = 0, permutation_list[x-1]
                    if x == n + 1:
                        break

                permutation_cycles.append(tuple(cycle))
        if len(permutation_cycles) == 0:
            permutation_cycles.append(tuple())

        return permutation_cycles

    @staticmethod
    def get_permutation_dict(permutation):
        """
        Returns the permutation in dictionary map (int : int) representation.
        """
        # Type check
        check_type('permutation', permutation, tuple)
        for n in permutation:
            check_type('permutation values', n, int, np.int64)

        permutation_dict = {}
        for n in range(len(permutation)):
            if permutation[n] != n+1:
                permutation_dict[n+1] = permutation[n]
        
        return permutation_dict

    @staticmethod
    def get_cycle_order(permutation_cycles):
        """
        Calculates the cycle of the permutation by finding the lowest common
        multiple of the permutation cycle lengths.
        """
        # Type checks
        check_type('permutation_cycles', permutation_cycles, list)
        for cycle in permutation_cycles:
            check_type('cycle', cycle, tuple)
            for n in cycle:
                check_type('permutation value', n, int, np.int64)

        cycle_len = [len(n) for n in permutation_cycles]

        cycle_order = np.lcm.reduce(cycle_len)

        return cycle_order
    
    # Output summary functions
    def __str__(self):
        """
        User-friendly output of the group element.
        """
        str_cycles = [str(cycle) for cycle in self.permutation_cycles]
        return "".join(str_cycles)

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(permutation_cycles = {self.permutation_cycles})"


"""
-------------------------------------------------------------------------------
Generator pointer representation of group elements
-------------------------------------------------------------------------------
"""


class PointerGroupElement(GroupElement):
    """
    Translates group element properties (element multiplication, inverse) to
    pointer representation of the group elements.


    In the pointer representation a specified group element is represented
    as a list of integers that correspond to the sequence of generators that
    produce the specified group element. For example, if the generator set is

    generator_list = {g1, g2},

    and some other group element g3 can be written as

    g3 = g1 * g2 * g2 * g1 * g1 *...
       = generator_list[0] * generator_list[1] * generator_list[1] * ...,

    then we may represent g3 as a pointer

    g3_pointer = [(0, 1), (1, 2), (0, 2), ...],

    where the first number in the tuple is the index of the generator in
    generator_list, and the second is the multiplicative power of this
    generator, i.e. the number of elements that form an unbroken product chain.

    Note that the power of an element is defined modulo its cycle order.
    Therefore, if the cycle orders of the generators are known, the program
    will use them to simplify pointer products and inverses.
    """

    # Object initialization
    def __init__(self, pointer, generator_list=None, generator_order=None):
        """
        Initializes a group element in the pointer representation.

        Arguments:
        pointer         - list of tuple, pointer representation of some group
                          element;
        generator_list  - list of GroupElement_type (default None),
                          if not None, provides a list of group generators;
        generator_order - list (default None), if not None, provides
                          the list of generator cycle orders.
        """
        # Type checks
        check_type('pointer', pointer, list)

        for p in pointer:
            check_type('pointer components', p, tuple)

        check_type('generators', generator_list, list, NoneType)

        if not_None(generator_list):
            for g in generator_list:
                if not issubclass(g.__class__, GroupElement):
                    raise Exception("All generators must inherit from "
                                    "GroupElement class.")

                if isinstance(g, PointerGroupElement):
                    raise Exception("PointerGroupElement is not a valid "
                                    "generator type.")

        check_type('generator_order', generator_order, list, NoneType)

        if not_None(generator_order):
            for o in generator_order:
                check_type('generator order', o, int, NoneType)

        # Define the number of generators and the list of generator 
        # cycle orders
        if not_None(generator_list):
            self.n_generators = len(generator_list)
            self.order = [g.order for g in generator_list]

        else:
            if not_None(generator_order):
                self.order = generator_order
                self.n_generators = len(self.order)

            else:
                raise Exception("Generator data not provided!")

        if isinstance(pointer, IdentityGroupElement):
            self.pointer = []

        else:
            # Test the validity of the pointer
            self.pointer = pointer

            for i, p in enumerate(pointer):
                if p[0] >= self.n_generators:
                    raise ValueError(f"Pointer value {p[0]} exceeds "
                                     f"the number of generators "
                                     f"{self.n_generators}.")

            self.simplify()

    # Common operations
    def simplify(self):
        """
        Attempts to simplify a pointer chain by contracting pointers to the
        same generators.
        """

        if len(self.pointer) > 0:

            new_pointer = [self.pointer[0]]

            counter = 0

            for p in self.pointer[1:]:
                if counter == -1:
                    new_pointer = [p]

                else:
                    if p[0] == new_pointer[counter][0]:
                        new_power = _mod(p[1] + new_pointer[counter][1],
                                         self.order[p[0]])

                        if new_power == 0:
                            new_pointer = new_pointer[:counter]
                            counter -= 1

                        else:
                            new_pointer[counter] = (p[0], new_power)

                    else:
                        new_pointer += [p]
                        counter += 1

            self.pointer = new_pointer

    def __mul__(self, element):
        """
        Group element multiplication in pointer representation.
        """

        # Product of two pointers
        if isinstance(element, PointerGroupElement):
            # Special case where self or element correspond to the identity,
            # represented by an empty pointer []
            if len(self.pointer) == 0:
                return element

            elif len(element.pointer) == 0:
                return self

            else:
                # Determine the rightmost generator of the left operator (self)
                # and leftmost generator of the right operator (element)
                g_left = self.pointer[-1]
                g_right = element.pointer[0]

                # If g_left and g_right are the same, we need to increase the
                # power of the generator (extend the generator chain)
                if g_left[0] == g_right[0]:
                    g_order = self.order[g_right[0]]
                    g_power_new = _mod(g_left[1] + g_right[1], g_order)

                    if g_power_new == 0:
                        g_extend = []

                    else:
                        g_extend = [(g_right[0], g_power_new)]

                    new_pointer = self.pointer[:-1]\
                                  + g_extend\
                                  + element.pointer[1:]

                # If g_lef and g_right are different generators, simply append
                # the pointers
                else:
                    new_pointer = self.pointer + element.pointer

                return PointerGroupElement(new_pointer,
                                           generator_order=self.order)

        # (Left) product between a pointer and a tuple
        elif isinstance(element, tuple) and len(element) == 2:
            # Special case where self corresponds to the identity,
            # represented by an empty pointer []
            if len(self.pointer) == 0:
                return PointerGroupElement([element],
                                           generator_order=self.order)

            else:
                # Determine the rightmost generator of the left operator (self)
                g_left = self.pointer[-1]
                g_right = element

                # If g_left and g_right are the same, we need to increase the
                # power of the generator (extend the generator chain)
                if g_left[0] == g_right[0]:
                    g_order = self.order[g_right[0]]
                    g_power_new = _mod(g_left[1] + g_right[1], g_order)

                    if g_power_new == 0:
                        g_extend = []

                    else:
                        g_extend = [(g_right[0], g_power_new)]

                    new_pointer = self.pointer[:-1] + g_extend

                # If g_lef and g_right are different generators, simply append
                # the pointers
                else:
                    new_pointer = self.pointer + [element]

                return PointerGroupElement(new_pointer,
                                           generator_order=self.order)

        # Left action on Identity object
        elif isinstance(element, IdentityGroupElement):
            return self

        # Left multiplication by other types is not defined
        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{self.__class__.__name__} and "
                            f"{type(element).__name__}.")

    def __rmul__(self, element):
        """
        Right group action is only defined for tuples.
        """
        # (Left) product between a pointer and a tuple
        if isinstance(element, tuple) and len(element) == 2:
            # Special case where self corresponds to the identity,
            # represented by an empty pointer []
            if len(self.pointer) == 0:
                return PointerGroupElement([element],
                                           generator_order=self.order)

            else:
                # Determine the leftmost generator of the right operator (self)
                g_right = self.pointer[0]
                g_left = element

                # If g_left and g_right are the same, we need to increase the
                # power of the generator (extend the generator chain)
                if g_left[0] == g_right[0]:
                    g_order = self.order[g_right[0]]
                    g_power_new = _mod(g_left[1] + g_right[1], g_order)

                    if g_power_new == 0:
                        g_extend = []

                    else:
                        g_extend = [(g_right[0], g_power_new)]

                    new_pointer = g_extend + self.pointer[1:]

                # If g_lef and g_right are different generators, simply append
                # the pointers
                else:
                    new_pointer = [element] + self.pointer

                return PointerGroupElement(new_pointer,
                                           generator_order=self.order)

        # Right action on Identity object
        elif isinstance(element, IdentityGroupElement):
            return self

        # Right multiplication by other types is not defined
        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{type(element).__name__} and "
                            f"{self.__class__.__name__}.")

    def inv(self):
        """
        Group element inverse for pointer representation.
        """

        pointer_inv = []
        for p in self.pointer:
            pointer_inv += [(p[0], _mod(-p[1], self.order[p[0]]))]

        return PointerGroupElement(pointer_inv,
                                   generator_order=self.order)

    def is_identity(self):
        """
        Returns True if the pointer is empty.
        """
        if len(self.pointer) == 0:
            return True
        else:
            return False

    def __eq__(self, element):
        """
        Pointer comparison.

        Note that a pointer representation of a group element is not unique,
        so this function is equivalent to group element comparison.
        """

        # Comparison of two pointers
        if isinstance(element, PointerGroupElement):
            return self.pointer == element.pointer

        # Comparison with IdentityGroupElement
        elif isinstance(element, IdentityGroupElement):
            return self.is_identity()

        # Any other comparison yields False
        else:
            warnings.warn(f"Comparison of "
                          f"{self.__class__.__name__} with "
                          f"{type(element).__name__} yields False by default.")
            return False

    # Output summary functions
    def __str__(self):
        """
        User-friendly output.
        """

        return str(self.pointer)

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(pointer = {self.pointer!r})"


"""
-------------------------------------------------------------------------------
Frequently used supplementary functions
-------------------------------------------------------------------------------
"""

def _array_equal(a1, a2, eps):
    """
    Returns True if elements of array 1 are the same as elements of array 2
    up to specified numerical precision.
    """
    return np.isclose(a1, a2, eps).all()

def _array_in_list(a, a_list):
    """
    Determines if a np.ndarray is included in a list of np.ndarrays.
    """
    # Define numerical precision
    eps = 1e-10
    return any(_array_equal(a,p,eps) for p in a_list)


def _element_in_list(g, g_list):
    """
    Determines if a GroupElement_type is included in a list of
    GroupElement_types.
    """
    return any(g == p for p in g_list)


def _mod(a, n):
    """
    If n is not None, return a % n, otherwise return a.
    """
    if not_None(n):
        return a % n
    else:
        return a


def _check_orthogonal(operator):
    """
    Checks if the operator is orthogonal.
    """

    operator = np.array(operator)

    # Define numerical precision required
    eps = 1e-10

    # Check if O * O.T = identity
    diff = np.max(np.abs(operator.dot(operator.T) - np.eye(len(operator))))

    if diff < eps:
        return True
    else:
        return False
