"""
tests/test_mesh.py

Automated unit tests for the zero-copy NumPy <-> OpenCASCADE mesh bridge.
Validates 3D rendering meshes (Graphic3d_ArrayOfTriangles) and CAD geometry meshes (Poly_Triangulation).
"""

import pytest
import numpy as np
from OCC.Core.Poly import Poly_Triangulation
from occ_numpy_bridge import occ_bridge, Poly_Triangulation_Helper


def get_cpp_ptr(occ_obj) -> int:
    """Helper to reliably extract the native C++ memory address for testing."""
    return int(occ_obj.this.this)

def test_poly_triangulation_transfer():
    """Verifies that CAD Poly_Triangulation meshes receive correct nodes and triangles."""
    num_nodes = 500
    num_triangles = 800

    np_coords = np.random.rand(num_nodes, 3).astype(np.float64) * 50.0
    # Generate random valid triangle indices within [0, num_nodes - 1]
    np_indices = np.random.randint(0, num_nodes, size=(num_triangles, 3), dtype=np.int32)

    poly = Poly_Triangulation_Helper.create_poly_triangulation(np_coords, np_indices)

    assert poly.NbNodes() == num_nodes, "Node count mismatch in Poly_Triangulation"
    assert poly.NbTriangles() == num_triangles, "Triangle count mismatch in Poly_Triangulation"

    # Validate last node coordinates
    last_node = poly.Node(num_nodes)
    assert np.isclose(last_node.X(), np_coords[-1, 0]), "Node X coordinate mismatch"
    assert np.isclose(last_node.Z(), np_coords[-1, 2]), "Node Z coordinate mismatch"

    # Validate triangle index shifting (0-based -> 1-based)
    # In pythonocc, Triangle(i).Get() returns a tuple (n1, n2, n3)
    tri_1 = poly.Triangle(1).Get()
    expected_n1 = np_indices[0, 0] + 1
    assert tri_1[0] == expected_n1, f"Expected node index {expected_n1}, got {tri_1[0]}"


def test_poly_triangulation_fill():
    """Verifies that CAD Poly_Triangulation meshes receive correct nodes and triangles."""
    num_nodes = 500
    num_triangles = 800

    np_coords = np.random.rand(num_nodes, 3).astype(np.float64) * 50.0
    # Generate random valid triangle indices within [0, num_nodes - 1]
    np_indices = np.random.randint(0, num_nodes, size=(num_triangles, 3), dtype=np.int32)

    poly = Poly_Triangulation(num_nodes, num_triangles, False)

    Poly_Triangulation_Helper.fill_poly_triangulation_coords(poly, np_coords)
    Poly_Triangulation_Helper.fill_poly_triangulation_indices(poly, np_indices)

    assert poly.NbNodes() == num_nodes, "Node count mismatch in Poly_Triangulation"
    assert poly.NbTriangles() == num_triangles, "Triangle count mismatch in Poly_Triangulation"

    # Validate last node coordinates
    last_node = poly.Node(num_nodes)
    assert np.isclose(last_node.X(), np_coords[-1, 0]), "Node X coordinate mismatch"
    assert np.isclose(last_node.Z(), np_coords[-1, 2]), "Node Z coordinate mismatch"

    # Validate triangle index shifting (0-based -> 1-based)
    # In pythonocc, Triangle(i).Get() returns a tuple (n1, n2, n3)
    tri_1 = poly.Triangle(1).Get()
    expected_n1 = np_indices[0, 0] + 1
    assert tri_1[0] == expected_n1, f"Expected node index {expected_n1}, got {tri_1[0]}"


def test_poly_triangulation_read():
    """Verifies that CAD Poly_Triangulation meshes receive correct nodes and triangles."""
    num_nodes = 500
    num_triangles = 800

    np_coords = np.random.rand(num_nodes, 3).astype(np.float64) * 50.0
    # Generate random valid triangle indices within [0, num_nodes - 1]
    np_indices = np.random.randint(0, num_nodes, size=(num_triangles, 3), dtype=np.int32)

    poly = Poly_Triangulation(num_nodes, num_triangles, False)

    Poly_Triangulation_Helper.fill_poly_triangulation_coords(poly, np_coords)
    Poly_Triangulation_Helper.fill_poly_triangulation_indices(poly, np_indices)

    coords = Poly_Triangulation_Helper.get_poly_triangulation_coords(poly)
    indices = Poly_Triangulation_Helper.get_poly_triangulation_indices(poly)

    assert np.isclose(coords, np_coords).all(), "Coords read mismatch"
    assert np.equal(indices, np_indices).all(), "Indices read mismatch"


def test_poly_triangulation_dimension_mismatch_raises_error():
    """Ensures C++ adapter throws an exception if NumPy shape does not match allocation."""
    np_coords = np.zeros((100, 3), dtype=np.float64)
    np_indices = np.zeros((50, 3), dtype=np.int32)

    # Allocate Poly_Triangulation expecting 200 nodes instead of 100
    poly = Poly_Triangulation(200, 50, False)
    cpp_ptr = get_cpp_ptr(poly)

    with pytest.raises(RuntimeError, match="do not match NumPy array shapes"):
        occ_bridge.poly.fill_poly_triangulation(cpp_ptr, np_coords, np_indices)

