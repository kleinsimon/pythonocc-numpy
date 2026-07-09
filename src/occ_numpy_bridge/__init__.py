"""
occ_numpy_bridge
Zero-Copy NumPy Adapter für OpenCASCADE und pythonocc-core.
"""

# Exportiere den nativen C++ Adapter
from . import occ_bridge

# Exportiere unsere Helper-Klasse für direkten Zugriff
from .helper import Graphic3d_ArrayOfPoints_Helper

__version__ = "0.1.0"
__all__ = ["occ_bridge", "Graphic3d_ArrayOfPoints_Helper"]