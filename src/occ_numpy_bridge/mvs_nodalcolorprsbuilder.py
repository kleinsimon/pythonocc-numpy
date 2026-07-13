import numpy as np
from OCC.Core.MeshVS import MeshVS_NodalColorPrsBuilder
from typing import Tuple

from .base import OCC_Wrapper_Base
from . import occ_bridge


class MeshVS_NodalColorPrsBuilder_Helper(OCC_Wrapper_Base):
    """
    Helper class for data transfer between NumPy arrays and
    OpenCASCADE Graphic3d_ArrayOfPoints objects.
    """

    _OCC_CLS = MeshVS_NodalColorPrsBuilder

    @classmethod
    def set_colors(cls, occ_obj: MeshVS_NodalColorPrsBuilder, np_colors: np.ndarray, np_indices=None) -> None:
        """
        Sets colors for a given OCC MeshVS_NodalColorPrsBuilder instance. This method allows
        updating nodal colors in the MeshVS instance with the provided color and index arrays.

        :param occ_obj: The instance of MeshVS_NodalColorPrsBuilder on which colors are to
                        be set.
        :type occ_obj: MeshVS_NodalColorPrsBuilder
        :param np_colors: A NumPy array of float64 type containing the nodal color specifications.
        :type np_colors: numpy.ndarray
        :param np_indices: (Optional) A NumPy array of int32 type containing nodal indices that
                           correspond to the provided colors. If None, colors will apply
                           in ascending order. 0-based indexing is assumed.
        :type np_indices: numpy.ndarray, optional
        :return: This method does not return a value.
        :rtype: None
        """

        np_colors = np.ascontiguousarray(np_colors, dtype=np.float64)

        if np_indices is not None:
            np_indices = np.ascontiguousarray(np_indices, dtype=np.int32)

        cpp_ptr = cls._get_cpp_pointer(occ_obj)
        occ_bridge.meshvs.fill_meshvs_nodal_color_prs_builder_colors(cpp_ptr, np_colors, np_indices)

    @classmethod
    def get_colors(cls, occ_obj: MeshVS_NodalColorPrsBuilder) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the nodal colors from the given MeshVS_NodalColorPrsBuilder object.

        This method retrieves two numpy arrays representing the nodal color data. The
        first array corresponds to the node indices, and the second array contains
        their associated color values. It interacts with a C++ bridge to fetch the
        data from the given OCC object efficiently.

        :param occ_obj: An instance of MeshVS_NodalColorPrsBuilder from which nodal
                        colors are to be fetched.
        :type occ_obj: MeshVS_NodalColorPrsBuilder
        :return: A tuple containing two numpy arrays: the first array represents node
                 indices (0-bases), and the second one represents associated color values.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        cpp_ptr = cls._get_cpp_pointer(occ_obj)
        return occ_bridge.meshvs.read_meshvs_nodal_color_prs_builder_colors(cpp_ptr)


