# `samosa.lattice`

This submodule defines `Lattice` object, which serves as a container for storing
information about crystal structure.

## Comments

- Perhaps it is better to create a more generic `Lattice` class with strict 
requirements (i.e. must input the exact space group number...). Then, just
like with the point groups, we can define a quick reference to some familiar
lattices (chain, square, hexagonal, cubic, bcc, fcc).


## General structure of the class

- The default initializer for the `Lattice` object takes in the space group 
number and a single coordinate of the lattice site;
- Additionally, we should have `classmethod` initializers that allow us to
initialize the class from Wyckoff position symbol, ...;
- The initializer should determine the site stabilizer point group, positions 
of nearest-neighbours, and bond stabilizer point group;
- For simulations, we need methods that generate sites inside of a single unit
cell, as well as a lattice of a prescribed size;
