# `frusa_symmetry` module

The goal of this module is to provide a toolset for symmetry analysis relevant
to the studies of anisotropic/frustrated lattice particles. 
The desired features of the module would include

- Decomposition of the particle characteristics (vertex/edge/face patches)
into irreducible representations (irreps);
- Automatic construction of interaction matrices for particles with specific
symmetry/characteristics;
- Calculation of the number of non-invariant degrees of freedom from the given
interaction matrix;
- Interface between Blender 3D and python for the extraction of the geometric
data;
- ...

## Tasks

### Implement adjacency function in the `lattice` class in order to calculate
the coordinates of the nearest-neighbour bonds. 

To do this properly, one has to know the generators of the space group.
Rather than tabulating them, the easier thing is to just require them as input.
From there the procedure is

- Calculate the stabilizer of a single point: this is the point group of the 
site;
- Determine nearest-neighbours along the principle lattice directions;
- Calculate all nearest neighbours using orbit/stabilizer search;
- Separate independent neighbour groups and determine bond point groups.
