import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfTriangles

from . import occ_bridge
from .base import OCC_Wrapper_Base


class Graphic3d_ArrayOfTriangles_Helper(OCC_Wrapper_Base):
    """
    Helper class for data transfer between NumPy arrays and
    OpenCASCADE Graphic3d_ArrayOfTriangles objects.
    """

    _OCC_CLS = Graphic3d_ArrayOfTriangles

    # --- OCCT 7.6+ / 7.9+ Bitmask Constants ---
    FLAG_NONE = 0
    FLAG_VERTEX_NORMAL = 1
    FLAG_VERTEX_COLOR = 2
    FLAG_VERTEX_TEXEL = 4

    @classmethod
    def set_coordinates(cls, occ_array: Graphic3d_ArrayOfTriangles, np_coords: np.ndarray) -> None:
        """
        Sets the coordinates of triangles in the provided Graphic3d_ArrayOfTriangles object.

        This method uses a C++ bridge to populate the coordinates of triangles
        within the given OpenCASCADE object by converting the provided NumPy array
        to a contiguous array of double-precision floating point numbers.

        :param occ_array: The Graphic3d_ArrayOfTriangles object to which the
            coordinates will be applied.
        :type occ_array: Graphic3d_ArrayOfTriangles
        :param np_coords: A NumPy array containing the coordinates to set in the
            format compatible with the provided Graphic3d_ArrayOfTriangles object.
        :return: None
        """
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        coords = np.ascontiguousarray(np_coords, dtype=np.float64)
        occ_bridge.graphic3d.fill_array_of_triangles_coords(cpp_ptr, coords)

    @classmethod
    def read_coordinates(cls, occ_array) -> np.ndarray:
        """
        Reads and returns the coordinates of triangles from a given array.

        This method processes the input array to retrieve the triangle coordinate
        data using a C++ pointer. It utilizes a helper bridge to interact with an
        underlying graphics library for efficient computation and data extraction.

        :param occ_array: Input array containing data to extract triangle coordinates.
                          The structure and format of this array should comply with
                          the requirements of the C++ backend library used.
        :type occ_array: Any
        :return: A NumPy array containing the triangle coordinates extracted from
                 the input `occ_array`.
        :rtype: np.ndarray
        """
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        return occ_bridge.graphic3d.read_array_of_triangles_coords(cpp_ptr)

    @classmethod
    def set_normals(cls, occ_array: Graphic3d_ArrayOfTriangles, np_normals: np.ndarray) -> None:
        """
        Set normals for a Graphic3d_ArrayOfTriangles object from a NumPy array.

        This method converts the given NumPy array of normals into a contiguous array
        of type `np.float64` and assigns these normals to the specified
        Graphic3d_ArrayOfTriangles object by using the `fill_array_of_triangles_normals`
        function from the occ_bridge.graphic3d module.

        :param occ_array: The Graphic3d_ArrayOfTriangles object which will have its normals set.
        :param np_normals: A NumPy ndarray containing the normals to be set. Must be compatible
                           with the shape and data type expected by `fill_array_of_triangles_normals`.
        :return: None
        """
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        normals = np.ascontiguousarray(np_normals, dtype=np.float64)
        occ_bridge.graphic3d.fill_array_of_triangles_normals(cpp_ptr, normals)

    @classmethod
    def read_normals(cls, occ_array) -> np.ndarray:
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        return occ_bridge.graphic3d.read_array_of_triangles_normals(cpp_ptr)

    @classmethod
    def set_colors(cls, occ_array: Graphic3d_ArrayOfTriangles, np_colors: np.ndarray) -> None:
        """
        Sets the colors for a `Graphic3d_ArrayOfTriangles` object.

        This method converts the provided numpy array of colors into a contiguous
        array and assigns it to the given OCC array of triangles through the
        Graphic3d bridge.

        :param occ_array: The OCC object representing an array of triangles.
        :type occ_array: Graphic3d_ArrayOfTriangles
        :param np_colors: A numpy array representing the colors to be set.
        :return: This method does not return a value.
        :rtype: None
        """
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        colors = np.ascontiguousarray(np_colors, dtype=np.float64)
        occ_bridge.graphic3d.fill_array_of_triangles_colors(cpp_ptr, colors)

    @classmethod
    def read_colors(cls, occ_array) -> np.ndarray:
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        return occ_bridge.graphic3d.read_array_of_triangles_colors(cpp_ptr)

    @classmethod
    def set_indices(cls, occ_array: Graphic3d_ArrayOfTriangles, np_indices: np.ndarray) -> None:
        """
        Updates the indices of an array of triangles in the given OCC graphic object.

        This method assigns a new set of indices to an OCC (Open CASCADE Technology)
        Graphic3d_ArrayOfTriangles object by leveraging numpy arrays. The process involves
        retrieving a C++ pointer to the given OCC array and filling its indices using a
        NOX bridge which interfaces between the OCC object and python runtime.

        :param occ_array: The OCC Graphic3d_ArrayOfTriangles object whose index array
            is being updated.
        :type occ_array: Graphic3d_ArrayOfTriangles
        :param np_indices: A numpy array containing the new indices to apply to the
            triangles in the OCC graphic object.
        :type np_indices: np.ndarray
        :return: None
        """
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        indices = np.ascontiguousarray(np_indices, dtype=np.int32)
        occ_bridge.graphic3d.fill_array_of_triangles_indices(cpp_ptr, indices)

    @classmethod
    def read_indices(cls, occ_array) -> np.ndarray:
        cpp_ptr = cls._get_cpp_pointer(occ_array)
        return occ_bridge.graphic3d.read_array_of_triangles_indices(cpp_ptr)

    @classmethod
    def create_from_numpy(
            cls,
            np_coords: np.ndarray,
            np_indices: np.ndarray,
            np_normals: np.ndarray = None,
            np_colors: np.ndarray = None,
    ) -> Graphic3d_ArrayOfTriangles:
        """
        Creates a high-performance OpenGL rendering mesh optimized for the 3D viewer.

        :param np_coords: Shape (N, 3), float64 - Vertex XYZ coordinates.
        :param np_indices: Shape (M, 3), int32 - 0-based triangle indices.
        :param np_normals: Shape (N, 3), float64 - Optional vertex normals for lighting.
        :param np_colors: Shape (N, 3), float64 - Optional RGB vertex colors [0.0 - 1.0].
        :return: Fully populated Graphic3d_ArrayOfTriangles instance.
        """

        num_nodes = len(np_coords)
        num_triangles = len(np_indices)

        # In Graphic3d, total edge allocations must equal 3 * number of triangles
        num_edges = num_triangles * 3

        # Resolve bitmask flags (0: None, 1: Normals, 2: Colors)
        flags = cls.FLAG_NONE
        if np_normals is not None:
            flags |= cls.FLAG_VERTEX_NORMAL
        if np_colors is not None:
            flags |= cls.FLAG_VERTEX_COLOR

        # 1. Allocate mesh buffer in OpenCASCADE
        occ_mesh = Graphic3d_ArrayOfTriangles(num_nodes, num_edges, flags)

        cls.set_coordinates(occ_mesh, np_coords)
        cls.set_indices(occ_mesh, np_indices)

        # 3. Transfer optional attributes
        if np_normals is not None:
            cls.set_normals(occ_mesh, np_normals)

        if np_colors is not None:
            cls.set_colors(occ_mesh, np_colors)

        return occ_mesh
