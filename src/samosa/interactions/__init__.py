#! /usr/bin/env python3
# Andrey Zelenskiy, 2025

import numpy as np

from samosa.symmetry import point_group
from samosa.symmetry.representations import PermutationGroupElement
from samosa.utils.type_checks import check_shape
from samosa.utils.array_checks import _locate_point_in_array


class InteractionMap:
    """
    A class used to manage interaction maps. In particular, this class includes
    methods to interconvert between face-pair and orientation representations,
    as well as to remove the energetic contribution to the surface tension.
    """

    def __init__(self, group, face_location, j_mean=0.0, j_std=1.0):
        """
        Creates an InteractionMap object from a point group and location of a
        single particle face (in the appropriate basis). Here, it is assumed
        that the interacting patches are located at the centres of the faces.

        The initialized face-pair matrix initially consists of random normal
        elements.

        Arguments:
        group         - samosa.symmetry.Group object, point group of the
                        particle;

        face_location - array suitable for group.calculate_orbit() method,
                        location of a single particle face;

        j_mean        - float (default=0.0), mean of the randomly drawn
                        interactions;

        j_std         - float (default=1.0), standard deviation of the randomly
                        drawn interactions;

        """

        # Calculate the orbit of the particle face
        face_orbit = group.calculate_orbit(np.array(face_location))

        # Store the number of faces
        self.__n_faces = face_orbit.size

        # Sort the faces in the orbit, such that if the index of a bond is i, then
        # the index of the negative bond is i + n_faces / 2
        sorted_orbit = np.zeros(self.n_faces, int)

        i = 0
        for p in face_orbit.points:
            p1 = _locate_point_in_array(p, face_orbit.points)
            if p1 + 1 in sorted_orbit:
                continue

            p2 = _locate_point_in_array(-p, face_orbit.points)

            sorted_orbit[i] = p1 + 1
            sorted_orbit[i + face_orbit.size // 2] = p2 + 1

            i += 1

        # Permute the basis of symmetry elements
        basis = PermutationGroupElement(sorted_orbit)

        group.generators = [
            g.permute_basis(basis) for g in face_orbit.generators
        ]

        # Calculate group elements and order them using the stabilizers and
        # transporters of the faces

        stabilizers = face_orbit.stabilizer_group(True).calculate_elements()

        if len(stabilizers) > 1:
            stabilizers = np.array(
                [s.permute_basis(basis) for s in stabilizers]
            )

            inds = np.argsort([s.cycle_order for s in stabilizers])
            inds[1:] = np.argsort([-s.cycle_order for s in stabilizers])[:-1]

            stabilizers = np.array(stabilizers)[inds]

        elements_ordered = []
        transporters = face_orbit.transporters(True)

        for i in sorted_orbit:

            t = transporters[list(transporters.keys())[i - 1]]

            if i != 1:
                t = t.permute_basis(basis)

            for s in stabilizers:
                elements_ordered += [t * s]

        group.elements = elements_ordered

        # Set the lattice point group
        self.__group = group

        # Sample face-pair interaction matrix
        self.sample_face_coupling_matrix(j_mean, j_std)

    @classmethod
    def from_lattice(cls, lattice):
        if lattice == "chain":
            return cls(point_group("C2"), [1, 0, 0])

        elif lattice == "square":
            return cls(point_group("C4"), [1, 0, 0])

        elif lattice == "triangular":
            return cls(point_group("C6"), [1, 0, 0])

        elif lattice == "cubic":
            return cls(point_group("O"), [1, 0, 0])

        elif lattice == "bcc":
            return cls(point_group("O"), [0.5, 0.5, 0.5])

        elif lattice == "fcc":
            return cls(point_group("O"), [0.5, 0.5, 0.0])

        else:
            raise NotImplementedError(
                f"{lattice} is not a valid lattice or is not implemented. \
                The implemented lattices are: chain, square, triangular, \
                cubic, bcc, and fcc."
            )

    def new_face_coupling_matrix(self, matrix):
        """
        Sets the face-pair coupling matrix to the input array.

        Arguments:
        j_matrix - float array(n_face, n_faces), input face-pair coupling
                   matrix.
        """
        # Check that the input matrix has correct size
        check_shape("matrix", matrix, self.n_faces, self.n_faces)

        # Check that the input matrix is symmetric
        if np.sum(np.abs(matrix - matrix.T)) >= 1e-10:
            raise ValueError("matrix must be symmetric!")

        self.__j_matrix = matrix

    def sample_face_coupling_matrix(self, j_mean, j_std):
        """
        Samples a symmetric matrix of size (n_faces, n_faces) from a normal
        distribution with mean j_mean and standard deviation j_std.

        The matrix is sampled such that all matrix elements have the same mean
        j_mean, but different standard deviation for diagonal (j_std) and
        off-diagonal (j_std/sqrt(2)) elements. In this way, the average over
        all matrix elements is a normally distributed random value with mean
        j_mean and standard deviation j_std / n_faces.

        Arguments:
        j_mean  - float, average of the sampled couplings;

        j_std   - float, positive, standard deviation of the sampled couplings;

        Returns:
        j_matrix - float array(n_faces, n_faces), symmetric matrix with normally
                   distributed elements;
        """
        # Sample the off-diagonal elements
        self.__j_matrix = 0.5 * np.random.normal(
            j_mean, j_std, (self.n_faces, self.n_faces)
        )
        self.__j_matrix += self.__j_matrix.T

    # Private methods
    def __get_face_coupling_matrix(self, k_matrix):
        """
        Converts orientation coupling matrix to face-pair basis.

        Arguments:
        k_matrix - float array, orientation coupling matrix;

        Returns:
        j_matrix - float array(n_faces, n_faces), face_pair coupling matrix;
        """
        # Reduce the orientation coupling matrix
        step = self.n_orientations // self.n_faces
        k_matrix = k_matrix[::step, ::step]

        # Define the transformation matrix

        m_matrix = np.zeros((self.n_faces, self.n_faces))
        m_matrix[: self.n_faces // 2, self.n_faces // 2 :] = m_matrix[
            self.n_faces // 2 :, : self.n_faces // 2
        ] = np.eye(self.n_faces // 2)

        return k_matrix.dot(m_matrix)

    # Class properties
    @property
    def n_faces(self):
        """
        Returns the number of particle faces
        """
        return self.__n_faces

    @property
    def n_orientations(self):
        """
        Returns number of particle orientations
        """
        return self.group.order

    @property
    def group(self):
        """
        Returns the point group of the lattice with elements stored as
        permutations of particle faces
        """
        return self.__group

    @property
    def face_coupling_matrix(self):
        """
        Returns coupling matrix in the face-pair representation
        """
        return self.__j_matrix

    @property
    def reduced_face_coupling_matrix(self):
        """
        Returns coupling matrix in the face-pair representation with rows
        that sum up to a constant value
        """
        # Compute Fourier transform and filter out the interactions between
        # q=0 and q!=0 modes
        j_matrix_fft = np.fft.fft2(self.__j_matrix)

        j_matrix_fft[0, 1:] = 0.0
        j_matrix_fft[1:, 0] = 0.0

        # Invert the Fourier transform
        return np.real(np.fft.ifft2(j_matrix_fft))

    @property
    def orientation_coupling_matrix(self):
        """
        Returns coupling matrix in the orientation representation
        """
        # Define face-inversion operation
        m_matrix = np.zeros((self.n_faces, self.n_faces))
        m_matrix[: self.n_faces // 2, self.n_faces // 2 :] = m_matrix[
            self.n_faces // 2 :, : self.n_faces // 2
        ] = np.eye(self.n_faces // 2)

        # Invert the face-pair coupling matrix
        j_matrix_inv = self.face_coupling_matrix.dot(m_matrix)

        # Calculate the indices of the orientation coupling matrix
        inds = np.array([g * 1 - 1 for g in self.group.elements])
        j, i = np.meshgrid(inds, inds)

        return j_matrix_inv[i, j]

    @property
    def reduced_orientation_coupling_matrix(self):
        """
        Returns coupling matrix in the orientation representation with rows
        that sum up to a constant value
        """
        # Calculate coupling matrix in the orientational representation
        k_matrix = self.orientation_coupling_matrix

        # Compute Fourier transform and filter out the interactions between
        # q=0 and q!=0 modes
        k_matrix_fft = np.fft.fft2(k_matrix)

        k_matrix_fft[0, 1:] = 0.0
        k_matrix_fft[1:, 0] = 0.0

        # Invert the Fourier transform
        return np.real(np.fft.ifft2(k_matrix_fft))
