"""
src/occ_numpy_bridge/helper.py

High-performance Python wrapper and factory methods for manipulating
OpenCASCADE Graphic3d_ArrayOfPoints objects using NumPy and zero-copy C++ adapters.
"""

import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfPoints, Graphic3d_ArrayOfTriangles
from . import occ_bridge


class Graphic3d_ArrayOfPoints_Helper:
    """
    Helper class for zero-copy data transfer between NumPy arrays and
    OpenCASCADE Graphic3d_ArrayOfPoints objects.
    """

    # --- OCCT 7.6+ / 7.9+ Bitmask Constants ---
    FLAG_NONE = 0
    FLAG_VERTEX_NORMAL = 1
    FLAG_VERTEX_COLOR = 2
    FLAG_VERTEX_TEXEL = 4

    @staticmethod
    def _get_cpp_pointer(occ_array: Graphic3d_ArrayOfPoints) -> int:
        """
        Reliably extracts the native C++ memory address from a pythonocc object
        by dereferencing the SWIG proxy object (double .this lookup).
        """
        try:
            return int(occ_array.this.this)
        except AttributeError:
            raise TypeError("Provided object is not a valid SWIG-wrapped OpenCASCADE instance!")

    @classmethod
    def set_coordinates(cls, occ_array: Graphic3d_ArrayOfPoints, np_coords: np.ndarray) -> None:
        """
        Populates the XYZ coordinates of an existing Graphic3d_ArrayOfPoints from a NumPy array.

        :param occ_array: Target OpenCASCADE Graphic3d_ArrayOfPoints instance.
        :param np_coords: NumPy array of shape (N, 3) with dtype=float64.
        """
        if not isinstance(np_coords, np.ndarray) or np_coords.dtype != np.float64:
            np_coords = np.ascontiguousarray(np_coords, dtype=np.float64)

        cpp_ptr = cls._get_cpp_pointer(occ_array)
        occ_bridge.graphic3d.fill_array_of_points_coords(cpp_ptr, np_coords)

    @classmethod
    def set_colors(cls, occ_array: Graphic3d_ArrayOfPoints, np_colors: np.ndarray) -> None:
        """
        Populates the RGB vertex colors of an existing Graphic3d_ArrayOfPoints from a NumPy array.

        :param occ_array: Target OpenCASCADE instance (must be initialized with color flag!).
        :param np_colors: NumPy array of shape (N, 3), dtype=float64, values between 0.0 and 1.0.
        """
        if not isinstance(np_colors, np.ndarray) or np_colors.dtype != np.float64:
            np_colors = np.ascontiguousarray(np_colors, dtype=np.float64)

        cpp_ptr = cls._get_cpp_pointer(occ_array)
        occ_bridge.graphic3d.fill_array_of_points_colors(cpp_ptr, np_colors)

    @classmethod
    def create_from_numpy(
        cls,
        np_coords: np.ndarray,
        np_colors: np.ndarray = None,
        with_normals: bool = False
    ) -> Graphic3d_ArrayOfPoints:
        """
        Factory method: Allocates and fully populates a Graphic3d_ArrayOfPoints in a single step.

        :param np_coords: NumPy array of shape (N, 3) containing XYZ coordinates.
        :param np_colors: Optional NumPy array of shape (N, 3) containing RGB colors [0.0 - 1.0].
        :param with_normals: Whether to allocate memory for normal vectors.
        :return: Fully populated Graphic3d_ArrayOfPoints instance ready for rendering.
        """
        num_points = len(np_coords)

        # 1. Assemble bitmask flags for constructor signature: Graphic3d_ArrayOfPoints(int, int)
        array_flags = cls.FLAG_NONE
        if np_colors is not None:
            array_flags |= cls.FLAG_VERTEX_COLOR
        if with_normals:
            array_flags |= cls.FLAG_VERTEX_NORMAL

        # 2. Allocate OpenCASCADE array on the C++ heap
        occ_array = Graphic3d_ArrayOfPoints(num_points, array_flags)

        # 3. Perform zero-copy data transfer
        cls.set_coordinates(occ_array, np_coords)

        if np_colors is not None:
            if len(np_colors) != num_points:
                raise ValueError("Coordinate and color arrays must have identical row counts!")
            cls.set_colors(occ_array, np_colors)

        return occ_array

    @classmethod
    def create_graphic3d_mesh(
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
        coords = np.ascontiguousarray(np_coords, dtype=np.float64)
        indices = np.ascontiguousarray(np_indices, dtype=np.int32)

        num_nodes = len(coords)
        num_triangles = len(indices)

        # In Graphic3d, total edge allocations must equal 3 * number of triangles
        num_edges = num_triangles * 3

        # Resolve bitmask flags (0: None, 1: Normals, 2: Colors)
        flags = 0
        if np_normals is not None:
            flags |= 1
        if np_colors is not None:
            flags |= 2

        # 1. Allocate mesh buffer in OpenCASCADE
        occ_mesh = Graphic3d_ArrayOfTriangles(num_nodes, num_edges, flags)
        cpp_ptr = cls._get_cpp_pointer(occ_mesh)

        # 2. Transfer coordinates and triangle indices natively
        occ_bridge.graphic3d.fill_array_of_triangles_coords(cpp_ptr, coords)
        occ_bridge.graphic3d.fill_array_of_triangles_indices(cpp_ptr, indices)

        # 3. Transfer optional attributes
        if np_normals is not None:
            normals = np.ascontiguousarray(np_normals, dtype=np.float64)
            occ_bridge.graphic3d.fill_array_of_triangles_normals(cpp_ptr, normals)

        if np_colors is not None:
            colors = np.ascontiguousarray(np_colors, dtype=np.float64)
            # Reusing our existing point cloud color adapter
            occ_bridge.graphic3d.fill_array_of_points_colors(cpp_ptr, colors)

        return occ_mesh