# `frusa_symmetry` module

The goal of this module is to provide a toolset for symmetry analysis relevant
to the studies of anisotropic/frustrated lattice particles. 
The desired features of the module would include

- A parser function to obtain relevant symmetry information from online
databases (such as bilbao crystallographic server);
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

- Write a parser to extract character tables and irreducible representations of
specified point groups,
- Write a parser to extract Wyckoff positions for a given space group;
- Write a `Geometry` class which specifies how an object on a lattice connects
to its neighbours, using the information about object's symmetry  and the
crystal space group;
- Write a Blender 3D plugin which allows to color (groups of) faces of an obect;

