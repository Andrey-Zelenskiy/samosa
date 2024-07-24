## Symmetry utilities 

This sub-module provides core tools for working with symmetries in the context
of group theory.

- `symmetry_utils.py` is the main script, which defines a `group` object, used
as a container to store all relevant symmetry properties and methods for a 
given problem.
- `point_group_utils.py` is a collection of methods used to define and 
manipulate 3D point groups.
- `symmetric_group_utils.py` is similar to `point_group_utils.py`, but is 
focused on symmetric groups of permutations.


### TO-DO

#### Organization
* Move methods that relate to point groups currently from `symmetry_utils.py` to
`point_group_utils.py`.

#### Features

* Implement `group_element` objects for matrix and permutation operators

The main properties of a `group_element` are

- method for element product;
- method for element inverse;
- method for group action; 
- `__str__` magic function for user-friendly output;
- method for returning the element as a matrix;
- method for returning the element as a permutation.

* Implement `group` class object in `symmetry_utils.py`.

The `group` object should contain 

- group generators;
- method to calculate an orbit of of a point;
- method to define a permutation representation for group generators from an
orbit/stabilizer chain;
- group character table;
- method for generating all group elements;
- method for finding conjugate classes;
- method for direct product of groups;

* Implement methods for defining 7 axial point groups in `point_group_utils.py`


