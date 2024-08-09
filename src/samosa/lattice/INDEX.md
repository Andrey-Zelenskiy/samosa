# `samosa.lattice`

This submodule defines `Lattice` object, which serves as a container for storing
information about crystal structure.

## Comments

- Perhaps it is better to create a more generic `Lattice` class with strict 
requirements (i.e. must input the exact space group number...). Then, just
like with the point groups, we can define a quick reference to some familiar
lattices (chain, square, hexagonal, cubic, bcc, fcc).


## TODO

[ ] Space group symmetry is provided either as an integer (as per ITC vol. A) or
as a combination of lattice type and point group.
[ ] Wyckoff position is either calculated from a single coordinate or selected 
from the list.
[ ] 


We would like to implement adjacency function in the `lattice` class in order 
to calculate the coordinates of the nearest-neighbour bonds. 
To do this properly, one has to know the generators of the space group.
Rather than tabulating them, the easier thing is to just require them as input.
From there the procedure is

- Calculate the stabilizer of a single point: this is the point group of the 
site;
- Determine nearest-neighbours along the principle lattice directions;
- Calculate all nearest neighbours using orbit/stabilizer search;
- Separate independent neighbour groups and determine bond point groups.

