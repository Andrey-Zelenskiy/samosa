# `samosa`: Symmetry Analysis MOdule for Self-Assembly

The aim of this library is to provide essential tools for symmetry analysis of
discrete geomentric objects.

## Submodules

## `samosa/api`

- [x] Variable type checking;
- [x] Array shape checking;
- [x] List membership checking;
- [ ] Class for variable checking;
- [ ] General `utils` script;
- [ ] Module-specific errors script;

## `samosa/database`

- [x] General `Database` class for uploading symmetry data;
- [x] `SpaceGroupDatabase` class for analysing space group properties;
- [x] Method for identifying allowed lattice types, point groups, and space
groups from a set of constraints;
- [ ] Wyckoff positions data;
- [ ] All point group symmetries for each space group;

## `samosa/lattice` and `samosa/particle`

- [ ] To be combined into `samosa/structures`;
- [ ] Definition of `Lattice` objects as crystal data containers;
- [ ] Definition of polyhedral particles from symmetry;
- [ ] Calculation of anisotropic interactions from particle structure;

## `samosa/symmetry`

- [x] General `Group` class with methods for orbit/stabilizer calculations;
- [x] Classes for group element representations: `MatrixGroupElement`,
`PermutationGroupElement`, `IdentityGroupElement`, and `PointerGroupElement`;
- [x] Methods for defining point groups from Schoefields symbols;
- [x] Methods for defining common 3D matrix operations (proper/improper
rotations and reflections);
- [ ] Tools for representation theory analysis: irreps, character tables,
group element traces;

## General TODOs

- [ ] PEP8 checks;
- [ ] Documentation;
- [ ] Tutorials (see `samosa/tutorials/INDEX.md` for current tutorial status);
