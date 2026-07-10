"""
tests/test_mesh.py

Automated unit tests for the zero-copy NumPy <-> OpenCASCADE mesh bridge.
Validates 3D rendering meshes (Graphic3d_ArrayOfTriangles) and CAD geometry meshes (Poly_Triangulation).
"""

import pytest
import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfTriangles
from OCC.Core.Poly import Poly_Triangulation
from occ_numpy_bridge import occ_bridge, Graphic3d_ArrayOfTriangles_Helper


def get_cpp_ptr(occ_obj) -> int:
    """Helper to reliably extract the native C++ memory address for testing."""
    return int(occ_obj.this.this)


def test_g3d_mesh_coordinates_and_indices_transfer():
    """Verifies accuracy of coordinate transfer and 0-based to 1-based index shifting."""
    # Create a simple 4-vertex quad (2 triangles)
    np_coords = np.array([
        [0.0, 0.0, 0.0],  # Index 0 (OCCT: 1)
        [10.0, 0.0, 0.0],  # Index 1 (OCCT: 2)
        [10.0, 10.0, 0.0],  # Index 2 (OCCT: 3)
        [0.0, 10.0, 0.0]  # Index 3 (OCCT: 4)
    ], dtype=np.float64)

    np_indices = np.array([
        [0, 1, 2],  # First triangle
        [0, 2, 3]  # Second triangle
    ], dtype=np.int32)

    # Allocate OCC array: 4 vertices, 6 edges (2 triangles * 3 edges)
    occ_mesh = Graphic3d_ArrayOfTriangles(4, 6, 0)
    cpp_ptr = get_cpp_ptr(occ_mesh)

    # Transfer data via C++ bridge
    occ_bridge.graphic3d.fill_array_of_triangles_coords(cpp_ptr, np_coords)
    occ_bridge.graphic3d.fill_array_of_triangles_indices(cpp_ptr, np_indices)

    # Validate coordinates (OCCT is 1-based)
    p3 = occ_mesh.Vertice(3)  # Corresponds to np_coords[2]
    assert np.isclose(p3.X(), 10.0), "X coordinate mismatch at vertex 3"
    assert np.isclose(p3.Y(), 10.0), "Y coordinate mismatch at vertex 3"
    assert np.isclose(p3.Z(), 0.0), "Z coordinate mismatch at vertex 3"

    # Validate indices (Edges in Graphic3d store the 1-based vertex indices)
    # First triangle (0, 1, 2 in NumPy -> 1, 2, 3 in OCCT)
    assert occ_mesh.Edge(1) == 1, "Edge 1 should point to vertex 1"
    assert occ_mesh.Edge(2) == 2, "Edge 2 should point to vertex 2"
    assert occ_mesh.Edge(3) == 3, "Edge 3 should point to vertex 3"

    # Second triangle (0, 2, 3 in NumPy -> 1, 3, 4 in OCCT)
    assert occ_mesh.Edge(4) == 1, "Edge 4 should point to vertex 1"
    assert occ_mesh.Edge(5) == 3, "Edge 5 should point to vertex 3"
    assert occ_mesh.Edge(6) == 4, "Edge 6 should point to vertex 4"


def test_g3d_mesh_normals_and_colors():
    """Verifies optional normal vector and vertex color transfers."""
    num_nodes = 3
    np_coords = np.zeros((num_nodes, 3), dtype=np.float64)
    np_indices = np.array([[0, 1, 2]], dtype=np.int32)

    # Normal pointing straight up along Z axis
    np_normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Pure red, green, and blue colors
    np_colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Use Poly_Triangulation_Helper to build the mesh with flags 1 (Normals) | 2 (Colors) = 3
    occ_mesh = Graphic3d_ArrayOfTriangles_Helper.create_from_numpy(
        np_coords, np_indices, np_normals=np_normals, np_colors=np_colors
    )

    assert occ_mesh.HasVertexNormals() is True, "Mesh should have normal flags enabled"
    assert occ_mesh.HasVertexColors() is True, "Mesh should have color flags enabled"

    # Validate normal of second vertex (Index 2 in OCCT)
    n2 = occ_mesh.VertexNormal(2)
    assert np.isclose(n2.X(), 0.0) and np.isclose(n2.Z(), 1.0), "Normal vector mismatch!"

    # Validate color of third vertex (Index 3 in OCCT)
    # Note: OCCT internal 8-bit quantization requires atol=1e-2
    c3 = occ_mesh.VertexColor(3)
    assert np.isclose(c3.Red(), 0.0, atol=1e-2), "Red channel mismatch!"
    assert np.isclose(c3.Blue(), 1.0, atol=1e-2), "Blue channel mismatch!"


def test_g3d_mesh_read():
    """Verifies optional normal vector and vertex color transfers."""
    num_nodes = 3
    np_coords = np.zeros((num_nodes, 3), dtype=np.float64)
    np_indices = np.array([[0, 1, 2]], dtype=np.int32)

    # Normal pointing straight up along Z axis
    np_normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Pure red, green, and blue colors
    np_colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Use Poly_Triangulation_Helper to build the mesh with flags 1 (Normals) | 2 (Colors) = 3
    occ_mesh = Graphic3d_ArrayOfTriangles_Helper.create_from_numpy(
        np_coords, np_indices, np_normals=np_normals, np_colors=np_colors
    )

    coords = Graphic3d_ArrayOfTriangles_Helper.read_coordinates(occ_mesh)
    indices = Graphic3d_ArrayOfTriangles_Helper.read_indices(occ_mesh)
    colors = Graphic3d_ArrayOfTriangles_Helper.read_colors(occ_mesh)
    normals = Graphic3d_ArrayOfTriangles_Helper.read_normals(occ_mesh)

    assert np.isclose(coords, np_coords).all, "Coords mismatch"
    assert np.equal(indices, np_indices).all, "Indices mismatch"
    assert np.isclose(colors, np_colors, atol=1e-2).all, "Color mismatch"
    assert np.isclose(normals, np_normals).all, "Normals mismatch"


def test_g3d_mesh_insufficient_edge_memory_raises_error():
    """Ensures C++ adapter aborts if triangle count exceeds allocated edge capacity."""
    np_coords = np.zeros((10, 3), dtype=np.float64)
    # 5 triangles require 15 edges
    np_indices = np.zeros((5, 3), dtype=np.int32)

    # Deliberately allocate space for only 3 edges (1 triangle)
    occ_mesh = Graphic3d_ArrayOfTriangles(10, 3, 0)
    cpp_ptr = get_cpp_ptr(occ_mesh)

    with pytest.raises(RuntimeError, match="Insufficient index memory"):
        occ_bridge.graphic3d.fill_array_of_triangles_indices(cpp_ptr, np_indices)


def test_poly_triangulation_dimension_mismatch_raises_error():
    """Ensures C++ adapter throws an exception if NumPy shape does not match allocation."""
    np_coords = np.zeros((100, 3), dtype=np.float64)
    np_indices = np.zeros((50, 3), dtype=np.int32)

    # Allocate Poly_Triangulation expecting 200 nodes instead of 100
    poly = Poly_Triangulation(200, 50, False)
    cpp_ptr = get_cpp_ptr(poly)

    with pytest.raises(RuntimeError, match="do not match NumPy array shapes"):
        occ_bridge.mesh.fill_poly_triangulation(cpp_ptr, np_coords, np_indices)


def test_invalid_index_dtype_raises_error():
    """Ensures passing float64 or int64 indices instead of int32 throws an exception."""
    np_coords = np.zeros((10, 3), dtype=np.float64)
    # Deliberately use int64 instead of required int32
    np_indices_int64 = np.zeros((5, 3), dtype=np.int64)

    with pytest.raises(RuntimeError, match="int32 dtype"):
        occ_mesh = Graphic3d_ArrayOfTriangles(10, 15, 0)
        cpp_ptr = get_cpp_ptr(occ_mesh)
        occ_bridge.graphic3d.fill_array_of_triangles_indices(cpp_ptr, np_indices_int64)