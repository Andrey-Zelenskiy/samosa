#! /usr/bin/env python3
# Andrey Zelenskiy, 2024

"""
==============================
samosa/symmetry/group_utils.py
==============================

This submodule defines the `Group` object, which contains all
of the necessary information and methods for symmetry analysis.
"""

import numpy as np

from samosa.utils.errors import custom_format_warning

from samosa.utils.type_checks import ArrayType, NoneType, is_None, not_None, \
                                     check_type, check_len, check_shape

from samosa.utils.array_checks import _array_in_list, _element_in_list

from samosa.symmetry.representations import GroupElement, \
                                            IdentityGroupElement, \
                                            PermutationGroupElement, \
                                            PointerGroupElement

import warnings
warnings.formatwarning = custom_format_warning
"""
-------------------------------------------------------------------------------
Group class
-------------------------------------------------------------------------------
"""


class Group:
    """
    A container for storing the properties and methods of a symmetry group.
    """

    def __init__(self, 
                 generators, 
                 name=None,  
                 order=None,
                 elements=None,
                 character_table=None,
                 irreps=None,
                 filter_generators=False):
        """
        Defines the basic group properties.

        Arguments:
        generators        - list of objects derrived from GroupElement class,
                            list of group generators;

        name              - str, (default=None) user-specified name of the
                            group;

        order             - int, (default=None) order of the group (number of 
                            elements);

        elements          - list of objects derrived from GroupElement class,
                            (default=None), list of group elements;

        character_table   - 2d_ArrayType, (default=None) character table of the
                            group;

        irreps            - list of ArrayType, (default=None) list of 
                            irreducible group representations;

        filter_generators - bool, (default=False) and option to perform
                            filtering of generators to remove redundant
                            operators.
        """
        # Initialize group generators
        self.__elements = None

        check_type('filter_generators', filter_generators, bool)
        if filter_generators == True:
            generators = self.generator_filter(generators)

        self.generators = generators

        # Initialize optional properties
        self.name = name
        self.order = order
        self.elements = elements
        self.character_table = character_table
        self.irreps = irreps
        
        self.__conjugate_classes = None
        self.__n_classes = None

    def calculate_elements(self, store_data=False):
        """
        Calculates all group elements from the list of generators.

        Arguments:
        store_data - bool, (default=False), if True, stores the calculated 
                     elements as self.elements.
        """
        # Type checks
        check_type('store_data', store_data, bool)

        # Generate all elements of the group
        dim = self.generators[0].dim
        element_list = list(self.generators) 

        for e in element_list:
            for g in self.generators:
                new_element = g*e
                if not _element_in_list(new_element, element_list):
                    element_list += [new_element]

        # Optionally store the elements in the group container
        if store_data == True:
            self.__elements = element_list
            self.order = len(self.elements)
            return self.elements

        else:
            return element_list
    
    def calculate_classes(self, store_data=False):
        """
        Sorts group elements into conjugate classes.

        Arguments:
        store_data - bool, (default=False), if True, stores the calculated 
                     elements as self.elements.
        """
        # Type checks
        check_type('store_data', store_data, bool)

        # Initialize array of group elements
        if is_None(self.elements):
            element_list = self.calculate_elements(store_data)
        else:
            element_list = list(self.elements)

        # Calculate conjugate classes 
        order = len(element_list)

        members = list(element_list)
        element_index = dict(zip(element_list, np.arange(order)))
        
        conjugate_classes = {}

        while len(members) > 0:
            e = members[0]
            e_sign = e.sign
            e_trace = e.trace

            class_ = [element_index[e]]
            members.remove(e)

            for h in element_list:
                e_h = h.inv*e*h

                if not element_index[e_h] in class_:
                    class_ += [element_index[e_h]]
                    members.remove(e_h)

            conjugate_classes[tuple(class_)] = {"sign" : e_sign, 
                                                "trace" : e_trace}
        
        # Optionally store conjugate classes in the class container
        if store_data == True:
            self.__conjugate_classes = conjugate_classes
            self.__n_classes = len(self.conjugate_classes.keys())
            return self.elements, self.conjugate_classes

        else:
            return element_list, conjugate_classes

    def calculate_orbit(self, p0, as_pointer=False):
        """
        Calculates the orbit of a single point under the action of the group.

        When a group element is applied to a seed point (p0), it either
        generates a new point or leaves the seed unchanged. The set of all
        points generated by applying all group elements to the seed is called
        the  orbit of the seed point.

        The following algorithm allows one to calculate the orbit, generators
        of the group, represented by permutations of the members of the orbit,
        transporters (group elements that transform the seed into other points
        in the orbit), and stabilizers (group elements that transform the seed
        into itself).

        Generators in the permutation representation are stored as a dict map
        from one point in the orbit to another (using integer hashing).

        To save memory, transporters and stabilizers may be stored as pointers
        to the sequence of generators that produce the correct group element.
        The option to output group elements as pointers is controlled by
        as_pointer argument.

        Note that the set of all stabilizers of the seed comprises a group
        (called the stabilizer group). The list of stabilizers in the output
        of this program will contain the generators of the stabilizer group,
        but will have redundant elements (i.e. not a miniomal generating set).
        Afterwards, one can use the generator_filter routine to obtaine the
        minimal generating set.


        Arguments:
        p0         - object for which multiplication GroupElement action is
                     supported, the 'seed' point of the orbit;

        as_pointer - bool, (default = False), when True, outputes the
                     group elements as pointers to the generating set.


        Returns:
        orbit            - list of type(p0), a closed set of points generated
                           from the seed  through application of group elements;

        permutations     - list of dict int:int, generators of the group
                           represented as permutations of members of the orbit;

        transporter_dict - dict tuple : PointerGroupElement (as_pointer = True)
                           or
                           dict tuple : GroupElement_type (as_pointer = False),
                           a set of group operators (values) that transform the
                           seed to other points in the orbit (keys);

        stabilizer_list  - list of PointerGroupElement (as_pointer = True) or
                           list of GroupElement_type (as_pointer = False),
                           operators that leave the seed point unchanged.
        """
        check_type("as_pointer", as_pointer, bool)

        # Initialize the orbit and orbit member index
        orbit = [p0]
        p_ind = {self.__hashed(p0) : 0}

        eye_element = IdentityGroupElement(self.generators[0].dim)

        if as_pointer:
            eye_element = PointerGroupElement.from_generators(eye_element, 
                                                              self.generators)

        permutations = [{} for i in range(len(self.generators))]
        transporter_dict = {self.__hashed(p0) : eye_element}
        stabilizer_list = [eye_element]

        for p in orbit:
            for ind_g, g in enumerate(self.generators):
                # Calculate a new point
                q = g*p

                p_h = self.__hashed(p)
                q_h = self.__hashed(q)

                # If the new point is not in the orbit, add it and update
                # generators/permutations and transporters
                if not _array_in_list(q, orbit):
                    orbit += [q]
                    p_ind[q_h] = len(orbit) - 1

                    if as_pointer:
                        transporter_dict[q_h] = \
                                (ind_g, 1)*transporter_dict[p_h]

                    else:
                        transporter_dict[q_h] = \
                                g*transporter_dict[p_h]

                # If the new point is already in the orbit, record the operation
                # as a stabilizer
                else:
                    """
                    We assume that some group generator g satisfies

                    g_21 p_1 = p_2

                    where p and q are points in the orbit of a known seed point
                    p_0. Assuming that

                    p_1 = g_10 p_0
                    p_2 = g_20 p_0

                    where operators g_10 and g_20 are known, we can obtain two
                    stabilizer operations as

                    s_1 = g_20^-1 g_21 g_10
                    s_2 = g_10^-1 g_21^-1 g_20 = s_1^-1

                    Since we are interested in generators, we only need s_1.
                    """

                    g_10 = transporter_dict[p_h]
                    g_20 = transporter_dict[q_h]
                    g_20_inv = g_20.inv

                    if as_pointer:
                        stabilizer = (g_20_inv*(ind_g, 1))*g_10
                    else:
                        stabilizer = g_20_inv*g*g_10

                    if not _element_in_list(stabilizer, stabilizer_list):
                        stabilizer_list += [stabilizer]

                permutations[ind_g][p_ind[p_h] + 1] = p_ind[q_h] + 1

        permutations = [PermutationGroupElement(p) for p in permutations]

        return orbit, permutations, transporter_dict, stabilizer_list

    def as_permutations(self, point):
        """
        Changes the basis of the group elements to permutations of the 
        members of an orbit formed from a single point.

        point - object for which multiplication GroupElement action is
                supported, seed of the orbit used for the permutation basis.

        Returns:
        None, updates group attributes to the new permutation basis.
        """
        out = self.calculate_orbit(point)

        if not_None(self.elements):
            del self.elements
            self.generators = out[1]
            element_list = self.calculate_elements(store_data=True)

        else:
            self.generators = out[1]

    # Method for filtering out redundant generators
    @staticmethod
    def generator_filter(generators):
        """
        Removes redundant operators from the candidate generator list.
        """

        # Conflict check
        self.__check_generators(generators)

        candidate_list = []
        filtered_generators = []
        
        dim = generators[0].dim

        # Remove potential duplicates and identity elements
        for g in generators:
            if not _element_in_list(g, candidate_list) and not g.is_identity:
                candidate_list += [g]

        # If the list is empty after sorting, the only generator is identity
        if len(candidate_list) == 0:
            filtered_generators = [IdentityGroupElement(dim)]

        # If only one generator is provided, no need to perform the check
        elif len(candidate_list) == 1:
            filtered_generators = candidate_list

        else:
            blacklist = [IdentityGroupElement(dim)]

            for g in candidate_list:
                # Check if the generator is related to the known generators
                if not _element_in_list(g, blacklist):

                    # Update self.__generators and blacklist
                    extended_element = [IdentityGroupElement(dim)]\
                                         + filtered_generators

                    for h in extended_element:
                        g_head = h*g
                        g_chain = g_head
                        while not _element_in_list(g_chain, blacklist):
                            blacklist += [g_chain]
                            g_chain = g_chain*g_head

                    filtered_generators += [g]

        return filtered_generators

    # Hashing for orbit calculation
    @staticmethod
    def __hashed(val):
        """
        If the input value is of ArrayType, changes the type to tuple, 
        otherwise returns the value itself.
        """
        if type(val) in ArrayType:
            return tuple(val)

        else:
            return val

    # Methods for property checks
    @staticmethod
    def __check_conflicts(*checks,**kwargs):
        """
        Performs conflict checks between group generators, all elements, and
        group order.

        Arguments:
        *checks    - str, instructs which checks to perform. Allowed values:
                     "number_of_elements", 
                     "generators_in_elements",
                     "group_order";
        
        **kwargs   - depending on the checks, the key-word arguments:

        generators - list of objects derrived from GroupElement class,
                     a candidate list of generators;

        elements   - list of objects derrived from GroupElement class,
                     a candidate list of group elements;

        order      - int, proposed group order.

        Returns:
        None if all checks are passed,
        Exception if at least one of the checks is failed.
        """
        # Number of generators does not exceed the number of group elements
        if "number_of_elements" in checks:
            generators = kwargs["generators"]
            elements = kwargs["elements"]
            
            if not_None(generators) and not_None(elements):
                if len(elements) < len(generators):
                    raise Exception(f"List of generators cannot be smaller "
                                    f"than the list of group generators:\n"
                                    f"|generators| = {len(generators)}, "
                                    f"|elements| = {len(elements)}.")

        # Generators are included in the list of all group elements
        if "generators_in_elements" in checks:
            generators = kwargs["generators"]
            elements = kwargs["elements"] 

            if not_None(generators) and not_None(elements):
                for g in generators:
                    if g not in elements:
                        elements_str = "\n".join([str(e) for e in elements])
                        raise Exception(f"Input conflict: generator {g} is "
                                        f"not a member of group elements list "
                                        f"{elements_str}.")

        # Group order is the same as the size of the elements list
        if "group_order" in checks:
            elements = kwargs["elements"]
            order = kwargs["order"]

            if not_None(elements) and not_None(order):
                if len(elements) != order:
                    raise Exception(f"Input conflict: group order {order} "
                                    f"does not equal to the size of group "
                                    f"elements list {len(elements)}.")

    def __check_generators(self, generators):
        """
        Performs type and conflict checks for an input list of group 
        generators.

        Arguments:
        generators - list of objects derrived from GroupElement class,
                     a candidate list of generators.

        Returns:
        None if all checks are passed,
        Exception if at least one of the checks is failed.
        """
        
        # Type checks
        check_type('generators', generators, list)

        for g in generators:
            if not issubclass(g.__class__, GroupElement):
                raise Exception("All generators must inherit from "
                                "GroupElement class.")

            if isinstance(g, PointerGroupElement):
                raise Exception("PointerGroupElement is not a valid generator "
                                "type.")

        # Non-zero length check
        if len(generators) == 0:
            raise Exception("List of generators cannot be empty.")

        # Conflict checks
        self.__check_conflicts("generators_in_elements",
                               generators=generators,
                               elements=self.elements)

    def __check_order(self, order):
        """
        Performs type and conflict checks for an input value of group order.

        Arguments:
        order      - int, proposed group order.

        Returns:
        None if all checks are passed,
        Exception if at least one of the checks is failed.
        """
        
        # Type checks
        check_type('order', order, int, NoneType)

        if not_None(order):
            # Conflict checks
            self.__check_conflicts("group_order",
                                   elements = self.elements,
                                   order = order)

    def __check_elements(self, elements):
        """
        Performs type and conflict checks for an input list of group elements.

        Arguments:
        elements   - list of objects derrived from GroupElement class,
                     a candidate list of group elements.

        Returns:
        None if all checks are passed,
        Exception if at least one of the checks is failed.
        """
        
        # Type checks
        check_type('elements', elements, list, NoneType)

        if not_None(elements):
            for e in elements:
                if not issubclass(e.__class__, GroupElement):
                    raise Exception("All group elements must inherit from "
                                    "GroupElement class.")

                if isinstance(e, PointerGroupElement):
                    raise Exception("PointerGroupElement is not a valid group "
                                    "element type.")

            # Non-zero length check
            if len(elements) == 0:
                raise Exception("List of group elements cannot be empty.")

            # Conflict checks
            self.__check_conflicts("number_of_elements",
                                   "generators_in_elements",
                                   "group_order",
                                   generators = self.generators,
                                   elements = elements,
                                   order = self.order)

    # Group properties
    @property
    def generators(self):
        """
        List of group generators.
        """
        return self.__generators

    @generators.setter
    def generators(self, val):
        self.__check_generators(val)
        self.__generators = val

    @property
    def name(self):
        """
        User-specified name of the group.
        """
        return self.__name

    @name.setter
    def name(self, val):
        check_type('name', val, str, NoneType)
        self.__name = val

    @name.deleter
    def name(self):
        self.__name = None

    @property
    def order(self):
        """
        Returns group order (number of group elements).
        """
        return self.__order

    @order.setter
    def order(self, val):
         self.__check_order(val)
         self.__order = val

    @order.deleter
    def order(self):
        self.__order = None

    @property
    def elements(self):
        """
        Returns a list of group elements.
        """    
        return self.__elements

    @elements.setter
    def elements(self, val):
        self.__check_elements(val)
        self.__elements = val

        if not_None(val):
            self.order = len(self.elements)

    @elements.deleter
    def elements(self):
        self.__elements = None

    @property
    def conjugate_classes(self):
        """
        Returns partition of group elements into conjugate classes. 
        """
        return self.__conjugate_classes

    @property
    def n_classes(self):
        """
        Returns the number of conjugate classes.
        """
        return self.__n_classes

    @property
    def character_table(self):
        """
        Returns group character table.
        """
        return self.__character_table

    @character_table.setter
    def character_table(self, val):
        self.__character_table = val

    @character_table.deleter
    def character_table(self):
        self.__character_table = None

    @property
    def irreps(self):
        """
        Returns irreducible representations of the group.
        """
        return self.__irreps

    @irreps.setter
    def irreps(self, val):
        self.__irreps = val

    @irreps.deleter
    def irreps(self):
        self.__irreps = None

    def __str__(self):
        """
        Provides user-friendly summary of the Group container.
        """

        generators_str = "\n\n".join([str(g) for g in self.generators])

        summary_string = ""
        if not_None(self.name):
            summary_string += f"Symmetry group {self.name}\n\n"
        
        summary_string += f"Generators:\n{generators_str};"
        
        if not_None(self.order):
            summary_string += f"\n\nGroup order: {self.order};"

        if not_None(self.elements):
            elements_str = "\n\n".join([str(e) for e in self.elements])
            summary_string += f"\n\nGroup elements:\n{elements_str};"

        if not_None(self.conjugate_classes):
            summary_string += f"\n\nConjugate classes:" +\
                              f"\n{self.conjugate_classes};"

        if not_None(self.character_table):
            summary_string += f"\n\nCharacter table:\n{self.character_table};"
        
        if not_None(self.irreps):
            summary_string += f"\n\nIrreducible representations:\n{self.irreps};"

        return summary_string

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(generators = {self.generators!r})"

"""
-------------------------------------------------------------------------------
Group class
-------------------------------------------------------------------------------
"""


class Orbit:
    """
    Defines a container for storing all properties of an orbit generated
    by a group. 
    """

    def __init__(self, 
                 generators, 
                 points, 
                 permutations, 
                 transporters, 
                 stabilizers):
        """
        Initializes properties of the orbit.

        Arguments:
        generators   - list, group generators in the original basis;
        
        points       - list, points in the orbit;
        
        permutations - list of PermutationGroupElement objects, group
                       generators represented as permutations of the orbit
                       elements;

        transporters - dict point : PointerGroupElement, group elements that
                       transport the first point in 
                       the orbit into all others;

        stabilizers  - list of PointerGroupElement objects, generators of the
                       stabilizer group (first point in the orbit).
        """
        # Type checks
        generators = Group.filter_generators(generators)

        check_type('points', points, list)
        
        check_type('permutations', permutations, Array)

        for perm in permutations:
            check_type(perm, PermutationGroupElement)

        check_type('transporters', transporters, dict)

        for p in points:
            check_type('transporter elements', 
                       transporters[p], PointerGroupElement)

        check_type('stabilizers', stabilizers, list)

        for s in stabilizers:
            check_type('stabilizer element', s, PointerGroupElement)

        # Store original space group generators
        self.__generators_group = generators

        # Calculate orbit
        self.__points = points
        self.__generators_permutations = permutations
        self.__transporter_pointers = transporters
        self.__stabilizers = stabilizers

    @classmethod
    def from_group(cls, group, position):
        """
        Calculate orbit given a group and the first member of the orbit.

        Arguments:
        group    - Group, symmetry group;

        position - first member of the orbit.
        """
        output = group.calculate_orbit(position, as_pointer=True)
        
        return cls(group.generators, 
                   output[0], 
                   output[1], 
                   output[2], 
                   output[3])

    @classmethod
    def sort_into_orbits(cls, item_list, group):
        """
        Sorts elements of a list into unique orbits.

        Arguments:
        item_list - list, initial list of objects;

        group     - Group, symmetry group for orbit calculations.

        Returns:
        orbit_list - list of Orbit, unique orbits of the input objects.
        """
        # Type checks
        check_type("item_list", item_list, list)
        check_type("group", group, Group)
        
        # Sort list elements into orbits
        orbit_list = []

        while len(item_list) != 0:
            p = item_list[0]
            item_orbit = cls.from_group(group, p)

            for q in item_orbit.points:
                if _array_in_list(q, item_list):
                    item_list.remove(q)

            orbit_list += [item_orbit]

        return orbit_list


    # WyckoffPosition properties
    @property
    def points(self):
        """
        Members of the orbit.
        """
        return self.__points

    @property
    def size(self):
        """
        Returns the size of the orbit.
        """
        return len(self.points)

    @property
    def generators(self):
        """
        Generators of the space group written as permutations of the Wyckoff 
        positions.
        """
        return self.__generators_permutations

    def transporters(self, as_permutations=False):
        """
        Operators that transform the first member of the orbit into other
        members. 

        Arguments:
        as_permutations - bool, (default=False), if True, returns the 
                          transporters as permutations of the orbit
                          members;
                          otherwise, returns transporters in the same
                          representation as the original group generators.

        Returns:
        transporter_dict - dict {(position) : transporter}, transporter
                           operators.
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

    def stabilizer_group(self, as_permutations=False):
        """
        Stabilizer group of the first orbit member. 

        Arguments:
        as_permutations - bool, (default=False), if True, returns the 
                          Group object generated by permutations of the orbit
                          members; 
                          otherwise, returns Group with generators in the
                          same representation as the original group generators.

        Returns:
        PointGroup - dict {(position) : transporter}, transporter operators
                     that change the initial vertex to another vertex in the
                     orbit.
        """
        if as_permutations:
            generators = self.generators
        else:
            generators = self.__generators_group

        return Group(generators, filter_generators=True)

    # Output summary functions
    def __str__(self):
        """
        Provides user-friendly summary of the Orbit container.
        """

        summary_string = f"Group orbit with size "\
                         f"{self.size}\n"
        
        summary_string += f"{self.points}"
        
        return summary_string

    def __repr__(self):
        """
        Provedes useful print output.
        """
        cls = self.__class__.__name__
        return f"{cls}(points={self.points})"

