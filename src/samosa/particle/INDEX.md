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


