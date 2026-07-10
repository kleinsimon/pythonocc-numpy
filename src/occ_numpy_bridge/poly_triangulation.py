import numpy as np
from OCC.Core.Poly import Poly_Triangulation
from . import occ_bridge


class Poly_Triangulation_Helper:
    """
    Helper class for creating rendering meshes (Graphic3d_ArrayOfTriangles)
    and CAD geometry meshes (Poly_Triangulation) directly from NumPy arrays.
    """

    @staticmethod
    def _get_cpp_pointer(occ_obj) -> int:
        """Extracts the underlying C++ pointer from a SWIG wrapper object."""
        if not isinstance(occ_obj, Poly_Triangulation):
            raise TypeError("Provided object is not a valid Poly_Triangulation instance!")

        try:
            return int(occ_obj.this.this)
        except AttributeError:
            raise TypeError("Provided object is not a valid SWIG-wrapped OpenCASCADE instance!")

    @classmethod
    def fill_poly_triangulation(cls, poly: Poly_Triangulation, coords: np.ndarray, indices: np.ndarray):
        """
        Fills the polygon triangulation with vertex coordinates and indices. This
        method leverages a C++ function to populate the data, converting the
        0-based indices in Python to 1-based indices required by the C++ layer.

        :param poly: The polygon triangulation object to populate.
        :type poly: Poly_Triangulation
        :param coords: A numpy array representing the vertex coordinates.
        :type coords: numpy.ndarray
        :param indices: A numpy array representing the vertex indices for the
            triangulations.
        :type indices: numpy.ndarray
        :return: None
        """
        cpp_ptr = cls._get_cpp_pointer(poly)

        # Populate via C++ (automatically shifts indices from 0-based to 1-based)
        occ_bridge.mesh.fill_poly_triangulation(cpp_ptr, coords, indices)

    @classmethod
    def fill_poly_triangulation_coords(cls, poly: Poly_Triangulation, coords: np.ndarray):
        """
        Fills the coordinate data for a Poly_Triangulation instance using a given
        coordinate array. The function utilizes a C++ backend to manage the data
        population, ensuring a shift in indices from 0-based to 1-based indexing.

        :param poly: Instance of Poly_Triangulation whose coordinates are to be
            populated.
        :param coords: A NumPy ndarray containing the coordinate data to populate.
        :return: None
        """
        cpp_ptr = cls._get_cpp_pointer(poly)
        occ_bridge.mesh.fill_poly_triangulation_coords(cpp_ptr, coords)

    @classmethod
    def get_poly_triangulation_coords(cls, poly: Poly_Triangulation) -> np.ndarray:
        """
        Get the triangulation coordinates of a given polygon.

        This method takes a polygon represented by a Poly_Triangulation object and
        returns its triangulation coordinates in the form of a NumPy array. The
        triangulation data is fetched from the underlying C++ pointer associated
        with the polygon through an external mesh reading utility.

        :param poly: The polygon object for which the triangulation coordinates
            are to be retrieved.
        :type poly: Poly_Triangulation
        :return: A NumPy array containing the triangulation coordinates of the
            given polygon.
        :rtype: np.ndarray
        """
        cpp_ptr = cls._get_cpp_pointer(poly)
        return occ_bridge.mesh.read_poly_triangulation_coords(cpp_ptr)

    @classmethod
    def fill_poly_triangulation_indices(cls, poly: Poly_Triangulation, indices: np.ndarray):
        """
        Fills the polygon triangulation indices using a given polygon triangulation object
        and a numpy array of indices. This method modifies the provided indices to account
        for shifts from 0-based to 1-based indexing, and it relies on an underlying C++
        implementation for its functionality.

        :param poly: A `Poly_Triangulation` object representing the polygon triangulation.
        :param indices: A numpy array containing the indices to be filled. Indices are
            expected to be shifted to account for 1-based indexing during processing.
        :return: None
        """
        cpp_ptr = cls._get_cpp_pointer(poly)

        # Populate via C++ (automatically shifts indices from 0-based to 1-based)
        occ_bridge.mesh.fill_poly_triangulation_indices(cpp_ptr, indices)

    @classmethod
    def get_poly_triangulation_indices(cls, poly: Poly_Triangulation) -> np.ndarray:
        """
        Retrieves the triangulation indices of a given polygon. The method retrieves the
        Python pointer of the polygon, processes it using an underlying C++ bridge, and
        returns the triangulation indices as a NumPy array. The indices are adjusted to
        be zero-based.

        :param poly: The polygon triangulation object from which indices are obtained.
        :type poly: Poly_Triangulation
        :return: A NumPy array containing the triangulation indices, adjusted to zero-based indices.
        :rtype: numpy.ndarray
        """
        cpp_ptr = cls._get_cpp_pointer(poly)

        # Populate via C++ (automatically shifts indices from 1-based to 0-based)
        return occ_bridge.mesh.read_poly_triangulation_indices(cpp_ptr)

    @classmethod
    def create_poly_triangulation(
            cls, np_coords: np.ndarray, np_indices: np.ndarray
    ) -> Poly_Triangulation:
        """
        Creates a CAD geometry mesh (Poly_Triangulation) suitable for STL export,
        boolean operations, or attaching to TopoDS_Face shapes.

        :param np_coords: Shape (N, 3), float64 - Vertex XYZ coordinates.
        :param np_indices: Shape (M, 3), int32 - 0-based triangle indices.
        :return: Fully populated Poly_Triangulation instance.
        """
        coords = np.ascontiguousarray(np_coords, dtype=np.float64)
        indices = np.ascontiguousarray(np_indices, dtype=np.int32)

        num_nodes = len(coords)
        num_triangles = len(indices)

        # Allocate Poly_Triangulation (last argument: UV nodes = False)
        poly = Poly_Triangulation(num_nodes, num_triangles, False)
        cls.fill_poly_triangulation(poly, coords, indices)

        return poly