"""
occ_numpy_bridge
Zero-Copy NumPy Adapter für OpenCASCADE und pythonocc-core.
"""

from importlib.metadata import version, PackageNotFoundError

from . import occ_bridge

from .g3d_arrayofpoints import Graphic3d_ArrayOfPoints_Helper
from .g3d_arrayoftriangles import Graphic3d_ArrayOfTriangles_Helper
from .poly_triangulation import Poly_Triangulation_Helper
from .mvs_nodalcolorprsbuilder import MeshVS_NodalColorPrsBuilder_Helper
from .mvs_mesh import MeshVS_Mesh_Helper

try:
    __version__ = version("occ-numpy-bridge")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "occ_bridge",
    "Graphic3d_ArrayOfPoints_Helper",
    "Poly_Triangulation_Helper",
    "Graphic3d_ArrayOfTriangles_Helper",
    "MeshVS_NodalColorPrsBuilder_Helper",
    "MeshVS_Mesh_Helper"
]