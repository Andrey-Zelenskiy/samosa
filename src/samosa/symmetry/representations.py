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

    @property
    def is_identity(self):
        """
        Returns True if the group element represents identity, 
        otherewise returns False.
        """
        return self.__is_identity

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

        check_type('dim', dim, int, NoneType)
        
        self.__dim = dim
        self.__is_identity = True

    # Common operations
    @property
    def inv(self):
        """
        Returns itself
        """
        return self

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
            return element.dim == self.dim:
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
    def __init__(self, operator, cycle_order=None, operator_inv=None):
        """
        Defines the matrix operator and, optionally, its inverse.

        Arguments:
        operator     - 2dArrayType, matrix operator defining the group
                       element or
                       IdentityGroupElement, defines an identity element;

        cycle_order  - int, (default = None), if not None, gives
                       the cycle order of the operator, i.e. n for which
                       operator^n = identity;

        operator_inv - 2dArrayType, (default = None), if not
                       None, proposes an inverse of operator.
        """

        GroupElement.__init__(self)

        check_type('operator', operator, ArrayType, IdentityGroupElement)

        # Define numerical precision for the matrix values
        eps = 1e-10
        log_eps = 10

        # Define identity if IdentityGroupElement is given
        if isinstance(operator, IdentityGroupElement):

        else:
            operator = np.round(np.array(operator), log_eps)
            self.__dim = len(operator)
            self.__is_identity = _array_equal(self.operator,
                                              np.eye(self.dim),
                                              eps)
            check_shape('operator', operator, self.dim, self.dim)

            self.operator = operator
            self.order = cycle_order
            self.orthogonal_basis = _check_orthogonal(self.operator)

        if not_None(operator_inv) and not self.orthogonal_basis:

            check_type('operator_inv', operator_inv, ArrayType)

            # Test that the proposed inverse yields identity when multiplied
            # by self.operator
            operator_inv = np.array(operator_inv)

            check_shape('operator_inv', operator_inv, self.dim, self.dim)

            identity_test = operator_inv.dot(self.operator) - np.eye(self.dim)

            if np.max(np.abs(identity_test)) > eps:
                raise ValueError("Proposed inverse does not produce "\
                                 "identity under multiplication with the "\
                                 "operator.")

            self.operator_inv = MatrixGroupElement(operator_inv)

        else:
            self.operator_inv = None

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
        return self.__cycle_order

    @property
    def orthogonal_basis(self):
        """
        Returns True if the matrix operator is defined in the orthogonal
        basis, otherwise returns False.
        """
        return self.__orthogonal_basis

    # Common operations
    @property
    def inv(self):
        """
        Returns the inverse of the matrix group element.
        """

        check_type('store_inverse', store_inverse, bool)

        # Check if operator inverse is stored
        if not_None(self.operator_inv):
            return self.operator_inv

        else:
            # If the operator is orthogonal, return the transpose
            if self.orthogonal_basis:
                operator_inverse = self.operator.T
            # If we have to calculate matrix inverse, it's good to store it
            # for future calculations
            else:
                operator_inverse = np.linalg.inv(self.operator)
                self.operator_inv = MatrixGroupElement(operator_inverse)

            return MatrixGroupElement(operator_inverse)

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

        # Left action on Identity object
        elif isinstance(element, IdentityGroupElement):
            return self

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

        # Right action on Identity object
        elif isinstance(element, IdentityGroupElement):
            return self

        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{type(element).__name__} and "
                            f"{self.__class__.__name__}.")

    def __eq__(self, element):
        """
        Determines if two matrix group elements are the same up to numerical
        precision.
        """

        # Define numerical tolerance
        eps = 1e-10

        # Comparison of two MatrixGroupElements
        if isinstance(element, MatrixGroupElement):
            return _array_equal(self.operator, element.operator, eps)

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
    def calculate_cycle_order(self):
        """
        Returns the integer cycle order of the matrix operator.
        """
        if not_None(self.cycle_order):
            return self.cycle_order
        
        else:
            # Define numerical precision
            eps = 1e-10

            mat = self.operator
            cycle = 1
            while not _array_equal(mat,np.eye(self.dim),eps):
                mat = mat.dot(self.operator)
                cycle += 1

            self.__cycle_order = cycle_order
        
        return self.cycle_order

    def __initialize_from_identity(self, operator):
        """
        Initializes object's attributes assuming the operator is an identity
        matrix.
        """
        if operator.dim == None:
            raise ValueError("Cannot initialize MatrixGroupElement from "
                             "IdentityGroupElement with a None type value"
                             "of dim.")

        self.__dim = operator.dim
        self.__operator = np.eye(self.dim)
        self.__cycle_order = 1
        self.__orthogonal_basis = True
        self.__operator_inv = None

    #TODO add initialization from operator

    # Output summary functions
    def __str__(self):
        """
        User-friendly output of the group element.
        """

        # Define float precision for output
        log_eps = 4

        return str(np.round(self.operator, log_eps))

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(operator = {np.round(self.operator, 4)!r})"


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
    def __init__(self, permutation, cycle_order=None):
        """
        Defines the permutation tuple.

        Arguments:
        permutation - ArrayType or dict int : int or IdentityGroupElement,
                      Permutations in the 'ordered' representation:
                      if ArrayType, assume map n : permutation[n];
                      if dict, assume a map n : m, where n and m are in the
                      same (closed) set;
                      if IdentityGroupElement, return cycle (1, 2, ..., dim).

        cycle_order  - int, (default = None), if not None, gives
                       the cycle order of the permutation.
        """

        check_type('permutation', permutation, dict, ArrayType,
                   IdentityGroupElement)

        # Define identity if IdentityGroupElement is given
        if isinstance(permutation, IdentityGroupElement):
            if permutation.dim == None:
                raise ValueError("Cannot initialize PermutationGroupElement "
                                 "from IdentityGroupElement with a None type "
                                 "value of dim.")

            self.__dim = permutation.dim
            self.__is_identity = True
            self.permutation = tuple(i for i in range(self.dim))
            self.order = 1

        elif isinstance(permutation, dict):
            self.__dim = max(permutation.keys()) + 1
            try:
                self.permutation = tuple(permutation[i]\
                                         for i in range(self.dim))
            except KeyError:
                raise ValueError(f"Permutation dictionary is incomplete: "
                                 f"{permutation}.")

            self.__is_identity = self.permutation == tuple(np.arange(self.dim))
            self.order = cycle_order


        else:
            self.permutation = tuple(permutation)
            self.__dim = len(permutation)

    # Common operations
    @property
    def inv():

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

            product = tuple(self.permutation[p] for p in element.permutation)

            return PermutationGroupElement(product)

        # Left action on an array
        elif isinstance(element, ArrayType):

            if len(element) < self.dim:
                raise Exception(f"Array must be of length at least {self.dim} "
                                f"in order to permute its elements.")

            new_element = np.array(element)

            for i in range(self.dim):
                new_element[i] = element[self.permutation[i]]

            if isinstance(element, np.ndarray):
                return new_element

            else:
                return type(element)(new_element)

        # Left action on int
        elif isinstance(element, int):
            if element <= self.dim:
                return self.permutation[element]
            else:
                return element

        # Left action on Identity object
        elif isinstance(element, IdentityGroupElement):
            return self

        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{self.__class__.__name__} and "
                            f"{type(element).__name__}.")

    def __rmul__(self, element):
        """
        Right permutation action is only defined for Identity.
        """

        # Right action on Identity object
        if isinstance(element, IdentityGroupElement):
            return self

        else:
            raise TypeError(f"Cannot multiply objects of type "
                            f"{type(element).__name__} and "
                            f"{self.__class__.__name__}.")

    def inv(self):
        """
        Returns inverse of self.permutation.
        """

        permutation_inverse = np.zeros(self.dim, int)

        for i in range(self.dim):
            permutation_inverse[self.permutation[i]] = i

        return PermutationGroupElement(permutation_inverse)

    def is_identity(self):
        """
        Returns True if none of the elements are permuted.
        """
        return self.permutation == tuple(np.arange(self.dim))

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

    # Output summary functions
    def __str__(self):
        """
        User-friendly output of the group element.
        """
        return str(self.permutation)

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(permutation = {self.permutation})"


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

        # Define the number of generators and the list of generator cycle orders
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
