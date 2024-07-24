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

### Symmetry database

Databases to add:

- Space group generators;
- Space group Wyckoff positions;
- Generic 2D and 3D point groups;

There should be a separate module for extracting information from the databases
in `frusa_symmetry/database/`.

- Given the name of the group in the database, define a group object;
- Convert between operator symbols and matrix representations (currently in 
`symmetry_utils`);

### Particle utils

The structure of an anisotropic particle is fully determined by the type of
lattice (specifically, nearest-neighbour (nn) connectivity), as well as the
site and bond symmetry.
The particle can be viewed as a Voronoi cell around a lattice point, with the
its faces directed normal to the nn bonds.
The subgroups of the site point group determine all of the possible types of
particle anisotropy, and the size of these point groups determines the number 
of unique particle orientations.
Note that if the point group of the nn bond is not trivial, the number of
particle orientations may be larger than the number of its faces.
This is due to the fact that for non-trivial bond symmetry, we may subdivide 
the faces of the particle into inequivalent sub-faces. 
The point group symmetry operations then permute these subfaces, leading to 
distinct particle images.


In the most symmetric case, an anisotropic particle has the same symmetry as 
the site point group, which implies that all faces are identical.
Correspondingly, the most anisotropic case refers to a particle for which all 
faces (and subfaces) are distinct.

- Given a vertex and bond stabilizers, determine possible particle 
orientations;
- Patchy particles: given a patch location, determine the number of unique
faces and orientations;

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
