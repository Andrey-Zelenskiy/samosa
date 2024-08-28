# Symmetry utilities 

This sub-module provides core tools for working with symmetries in the context
of group theory.

- `group_utils.py` is the main script, which defines a `Group` and 
`GroupElement` objects, used as a container to store all relevant symmetry 
properties and methods for a given problem.
- `point_group_utils.py` is a collection of methods used to define and 
manipulate 3D point groups.
- `symmetric_group_utils.py` is similar to `point_group_utils.py`, but is 
focused on symmetric groups of permutations.


## Main classes

## Group (in `group_utils.py`)

- [x] group generators;
- [x] method to calculate an orbit of of a point;
- [x] method to remove redundant operators from the generator list;
- [x] method to define a permutation representation for group generators from an
orbit/stabilizer chain;
- [x] Change `is_identity` to a class property;
- [ ] method to convert from pointers to group elements;
- [ ] method to change generator representation to permutations;
- [ ] group character table;
- [x] method for generating all group elements;
- [x] method for finding conjugate classes;
- [ ] method for direct product of groups;

## MatrixGroupElement (in `group_utils.py`)

- [x] Clean up inverse-storage options;
- [x] Type tests; 
- [x] Implement cycle order calculation method;

## PointerGroupElement (in `group_utils.py`)

- [x] Type tests;

## PermutationGroupElement (in `group_utils.py`)

- [x] Group action;
- [x] Inverse element;
- [x] Permutation comparison;
- [x] `__str__` magic;
- [x] `__repr__` magic;
- [x] Type tests;
- [x] Implement cycle order calculation method;

