### Lattice utils

We would like to implement adjacency function in the `lattice` class in order t
o calculate the coordinates of the nearest-neighbour bonds. 
To do this properly, one has to know the generators of the space group.
Rather than tabulating them, the easier thing is to just require them as input.
From there the procedure is

- Calculate the stabilizer of a single point: this is the point group of the 
site;
- Determine nearest-neighbours along the principle lattice directions;
- Calculate all nearest neighbours using orbit/stabilizer search;
- Separate independent neighbour groups and determine bond point groups.

