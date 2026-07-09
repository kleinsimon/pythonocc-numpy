"""
src/occ_numpy_bridge/mesh_helper.py

High-performance Python helper methods for zero-copy transfer of 3D meshes
from NumPy arrays into OpenCASCADE rendering and geometry objects.
"""

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
        cpp_ptr = cls._get_cpp_pointer(poly)

        # Populate via C++ (automatically shifts indices from 0-based to 1-based)
        occ_bridge.mesh.fill_poly_triangulation_coords(cpp_ptr, coords)

    @classmethod
    def fill_poly_triangulation_indices(cls, poly: Poly_Triangulation, indices: np.ndarray):
        cpp_ptr = cls._get_cpp_pointer(poly)

        # Populate via C++ (automatically shifts indices from 0-based to 1-based)
        occ_bridge.mesh.fill_poly_triangulation_indices(cpp_ptr, indices)


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