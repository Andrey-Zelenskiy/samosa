# Symmetry utilities 

This sub-module provides core tools for working with symmetries in the context
of group theory.

- `symmetry_utils.py` is the main script, which defines a `Group` object, used
as a container to store all relevant symmetry properties and methods for a 
given problem.
- `point_group_utils.py` is a collection of methods used to define and 
manipulate 3D point groups.
- `symmetric_group_utils.py` is similar to `point_group_utils.py`, but is 
focused on symmetric groups of permutations.


## Main classes

## Group (in `symmetry_utils.py`)

- [x] group generators;
- [x] method to calculate an orbit of of a point;
- [x] method to remove redundant operators from the generator list;
- [ ] method to define a permutation representation for group generators from an
orbit/stabilizer chain;
- [ ] group character table;
- [ ] method for generating all group elements;
- [ ] method for finding conjugate classes;
- [ ] method for direct product of groups;

## MatrixGroupElement (in `symmetry_utils.py`)

### TODO

- [ ] Clean up inverse-storage options;
- [ ] Implement cycle order calculation method;

## PointerGroupElement (in `symmetry_utils.py`)



