"""
occ_numpy_bridge
Zero-Copy NumPy Adapter für OpenCASCADE und pythonocc-core.
"""

# Exportiere den nativen C++ Adapter
from . import occ_bridge

# Exportiere unsere Helper-Klasse für direkten Zugriff
from .g3d_arrayofpoints import Graphic3d_ArrayOfPoints_Helper
from .g3d_arrayoftriangles import Graphic3d_ArrayOfTriangles_Helper
from .poly_triangulation import Poly_Triangulation_Helper
from .mvs_nodalcolorprsbuilder import MeshVS_NodalColorPrsBuilder_Helper
from .mvs_mesh import MeshVS_Mesh_Helper

__version__ = "0.1.4.dev1"
__all__ = [
    "occ_bridge",
    "Graphic3d_ArrayOfPoints_Helper",
    "Poly_Triangulation_Helper",
    "Graphic3d_ArrayOfTriangles_Helper",
    "MeshVS_NodalColorPrsBuilder_Helper",
    "MeshVS_Mesh_Helper"
]