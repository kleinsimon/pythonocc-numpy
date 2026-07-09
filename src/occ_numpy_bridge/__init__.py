"""
occ_numpy_bridge
Zero-Copy NumPy Adapter für OpenCASCADE und pythonocc-core.
"""

# Exportiere den nativen C++ Adapter
from . import occ_bridge

# Exportiere unsere Helper-Klasse für direkten Zugriff
from .graphics3d import Graphic3d_ArrayOfPoints_Helper
from .poly import Poly_Triangulation_Helper

__version__ = "0.1.1"
__all__ = ["occ_bridge", "Graphic3d_ArrayOfPoints_Helper", "Poly_Triangulation_Helper"]