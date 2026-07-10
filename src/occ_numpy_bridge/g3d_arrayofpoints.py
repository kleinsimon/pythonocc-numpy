import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfPoints
from . import occ_bridge


class Graphic3d_ArrayOfPoints_Helper:
    """
    Helper class for data transfer between NumPy arrays and
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
        if not isinstance(occ_array, Graphic3d_ArrayOfPoints):
            raise TypeError("Provided object is not a valid Graphic3d_ArrayOfPoints instance!")

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
    def get_coordinates(cls, occ_array: Graphic3d_ArrayOfPoints) -> np.ndarray:
        """
        Gets the coordinates from a given Graphic3d_ArrayOfPoints instance.

        The method retrieves the internal representation of the provided `Graphic3d_ArrayOfPoints`
        to extract its data structure into a NumPy array of coordinates. This allows for efficient
        and structured handling of the points' data.

        :param occ_array: An instance of Graphic3d_ArrayOfPoints containing the points data
            for which coordinates need to be retrieved.
        :return: A NumPy array representing the coordinates extracted from the provided
            `Graphic3d_ArrayOfPoints`.
        :rtype: np.ndarray Nx3, float64
        """

        cpp_ptr = cls._get_cpp_pointer(occ_array)
        return occ_bridge.graphic3d.read_array_of_points_coords(cpp_ptr)

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
    def get_colors(cls, occ_array: Graphic3d_ArrayOfPoints) -> np.ndarray:
        """
        Extracts the colors from an array of points and returns them in a numpy array.

        This class method interfaces with the graphic3d library to read a color array
        from the provided `Graphic3d_ArrayOfPoints` instance.

        :param occ_array: An instance of Graphic3d_ArrayOfPoints containing the points
                          whose colors need to be retrieved.
        :return: A numpy array containing the colors of the points.
        :rtype: np.ndarray Nx3, float64
        """

        cpp_ptr = cls._get_cpp_pointer(occ_array)
        return occ_bridge.graphic3d.read_array_of_points_colors(cpp_ptr)

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

