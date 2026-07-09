"""
pythonocc_helper.py

Ein sauberes Wrapper-Modul für den schnellen Datenaustausch zwischen
pythonocc-core und NumPy mithilfe des nativen C++ Adapters (occ_bridge).
"""

import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfPoints

# Versuche den nativen C++ Adapter zu laden
try:
    import occ_bridge
except ImportError:
    try:
        from . import occ_bridge
    except ImportError:

        raise ImportError(
            "Der native C++ Adapter 'occ_bridge' konnte nicht gefunden werden! "
            "Bitte stelle sicher, dass das Modul kompiliert wurde und im PYTHONPATH liegt."
        )


class Graphic3d_ArrayOfPoints_Helper:
    """
    Provides helper functionalities for efficient data manipulation and memory
    management of `Graphic3d_ArrayOfPoints` in the OpenCASCADE framework.

    This class includes static and class methods to handle point coordinates,
    colors, and optional normals using NumPy arrays. It bridges the gap
    between Python and the underlying OpenCASCADE C++ API for optimized
    performance and seamless data transfer.

    :ivar FLAG_NONE: No flags are set for the array.
    :type FLAG_NONE: int
    :ivar FLAG_VERTEX_NORMAL: Flag for enabling normals in the array.
    :type FLAG_VERTEX_NORMAL: int
    :ivar FLAG_VERTEX_COLOR: Flag for enabling color data in the array.
    :type FLAG_VERTEX_COLOR: int
    :ivar FLAG_VERTEX_TEXEL: Flag for enabling texel data in the array.
    :type FLAG_VERTEX_TEXEL: int
    """
    FLAG_NONE = 0
    FLAG_VERTEX_NORMAL = 1
    FLAG_VERTEX_COLOR = 2
    FLAG_VERTEX_TEXEL = 4

    @staticmethod
    def _get_cpp_pointer(occ_array: Graphic3d_ArrayOfPoints) -> int:
        """
        Returns the pointer representing the given OCC array as an integer. This method
        is designed for fetching the internal representation of a SWIG-wrapped OpenCASCADE
        object. The OCC array must be a valid type to successfully retrieve the pointer.

        :param occ_array: The OCC array object of type Graphic3d_ArrayOfPoints for which
            the pointer is to be retrieved.
        :type occ_array: Graphic3d_ArrayOfPoints

        :return: The integer pointer representing the internal SWIG-based structure
            associated with the given OCC array.
        :rtype: int

        :raises TypeError: If the provided object is not a valid SWIG-wrapped OpenCASCADE
            object, an error is raised indicating an invalid input.
        """
        try:
            return int(occ_array.this.this)
        except AttributeError:
            raise TypeError(
                "Das übergebene Objekt ist kein gültiges SWIG-Wrapped OpenCASCADE Objekt!"
            )

    @staticmethod
    def set_coordinates(occ_array: Graphic3d_ArrayOfPoints, np_coords: np.ndarray) -> None:
        """
        Sets the coordinates of the given `Graphic3d_ArrayOfPoints` instance using a
        NumPy array, ensuring the array is a contiguous array with dtype `float64`.

        :param occ_array: The `Graphic3d_ArrayOfPoints` instance to modify.
        :type occ_array: Graphic3d_ArrayOfPoints
        :param np_coords: A NumPy array containing the new coordinates to be set. The array
            must have dtype `float64`. If a non-NumPy array or array with a different dtype
            is provided, it will be converted to a contiguous NumPy array of dtype `float64`.
        :type np_coords: np.ndarray
        :return: None
        """
        if not isinstance(np_coords, np.ndarray) or np_coords.dtype != np.float64:
            np_coords = np.ascontiguousarray(np_coords, dtype=np.float64)

        cpp_ptr = Graphic3d_ArrayOfPoints_Helper._get_cpp_pointer(occ_array)
        occ_bridge.graphic3d.fill_array_of_points_coords(cpp_ptr, np_coords)

    @staticmethod
    def set_colors(occ_array: Graphic3d_ArrayOfPoints, np_colors: np.ndarray) -> None:
        """
        Sets the colors for a Graphic3d_ArrayOfPoints object using a NumPy array.

        This method updates the colors of the given Graphic3d_ArrayOfPoints object by
        mapping the provided NumPy array of color data. The NumPy array must have
        a floating-point data type to match the expected format.

        :param occ_array: The target Graphic3d_ArrayOfPoints object whose colors
            will be set.
        :param np_colors: A NumPy array containing color data to apply. The array
            must be of dtype `float64`.
        :return: None
        """
        if not isinstance(np_colors, np.ndarray) or np_colors.dtype != np.float64:
            np_colors = np.ascontiguousarray(np_colors, dtype=np.float64)

        cpp_ptr = Graphic3d_ArrayOfPoints_Helper._get_cpp_pointer(occ_array)
        occ_bridge.graphic3d.fill_array_of_points_colors(cpp_ptr, np_colors)

    @classmethod
    def create_from_numpy(
        cls, np_coords: np.ndarray, np_colors: np.ndarray = None, with_normals: bool = False
    ) -> Graphic3d_ArrayOfPoints:
        """
        Creates an instance of Graphic3d_ArrayOfPoints from numpy arrays for coordinates
        and optionally colors and normals. This method manages memory allocation and data
        transfer to the OpenCASCADE object `Graphic3d_ArrayOfPoints`.

        :param np_coords: A numpy array containing the coordinates of the points. The array
            should have shape (n, 3), where n is the number of points and 3 corresponds to the
            x, y, and z components of each coordinate.
        :param np_colors: A numpy array containing the colors of each point (optional). The
            array should have shape (n, 3), where each row represents RGB values. If provided,
            the length of `np_colors` must match the length of `np_coords`.
        :param with_normals: A boolean indicating whether normal vectors should be allocated and
            considered in memory. If True, the FLAG_VERTEX_NORMAL is set during allocation.
        :return: An instance of `Graphic3d_ArrayOfPoints` populated with the data from the
            given numpy arrays.
        """
        num_points = len(np_coords)

        # 1. Flag bestimmen und Speicher in OpenCASCADE allokieren
        array_flags = cls.FLAG_NONE
        if np_colors is not None:
            array_flags |= cls.FLAG_VERTEX_COLOR  # Setzt Bit für Farbe (2)
        if with_normals:
            array_flags |= cls.FLAG_VERTEX_NORMAL  # Setzt Bit für Normalen (1)

        occ_array = Graphic3d_ArrayOfPoints(num_points, array_flags)

        # 2. Koordinaten übertragen
        cls.set_coordinates(occ_array, np_coords)

        # 3. Falls vorhanden, Farben übertragen
        if np_colors is not None:
            if len(np_colors) != num_points:
                raise ValueError("Koordinaten- und Farb-Arrays müssen die gleiche Länge haben!")
            cls.set_colors(occ_array, np_colors)

        return occ_array