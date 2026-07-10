"""
tests/test_bridge.py

Automatisierte Unit-Tests für die Zero-Copy NumPy <-> OpenCASCADE Brücke.
"""

import pytest
import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfPoints

from occ_numpy_bridge import occ_bridge, Graphic3d_ArrayOfPoints_Helper


# --- HILFSFUNKTIONEN FÜR DIE TESTS ---

def get_cpp_ptr(occ_array: Graphic3d_ArrayOfPoints) -> int:
    """Holt die native C++ Speicheradresse für Testzwecke."""
    return int(occ_array.this.this)


# --- 1. TESTS FÜR DIE REINE KOORDINATEN-ÜBERTRAGUNG ---

def test_coords_transfer_accuracy():
    """Prüft, ob 3D-Koordinaten exakt und ohne Präzisionsverlust übertragen werden."""
    num_points = 1_000
    np_coords = np.random.rand(num_points, 3).astype(np.float64) * 100.0

    occ_array = Graphic3d_ArrayOfPoints(num_points, 0)
    cpp_ptr = get_cpp_ptr(occ_array)

    # C++ Brücke aufrufen
    occ_bridge.graphic3d.fill_array_of_points_coords(cpp_ptr, np_coords)

    # Validierung: Erster, mittlerer und letzter Punkt (OCCT ist 1-basiert!)
    for idx in [0, num_points // 2, num_points - 1]:
        p_occ = occ_array.Vertice(idx + 1)
        assert np.isclose(p_occ.X(), np_coords[idx, 0]), f"Fehler bei X (Index {idx})"
        assert np.isclose(p_occ.Y(), np_coords[idx, 1]), f"Fehler bei Y (Index {idx})"
        assert np.isclose(p_occ.Z(), np_coords[idx, 2]), f"Fehler bei Z (Index {idx})"


# --- 2. TESTS FÜR VERTEX-FARBEN ---

def test_colors_transfer_and_clamping():
    """Prüft RGB-Farbübertragung und ob Werte außerhalb [0.0, 1.0] korrekt geklammert werden."""
    num_points = 5
    # Wir testen gezielt Werte unter 0.0 und über 1.0 für das Clamping
    np_colors = np.array([
        [0.0, 0.5, 1.0],
        [-0.5, 1.5, 0.5],  # Sollte zu [0.0, 1.0, 0.5] geklammert werden
        [0.2, 0.3, 0.4],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float64)

    occ_array = Graphic3d_ArrayOfPoints(num_points, 2)
    cpp_ptr = get_cpp_ptr(occ_array)

    occ_bridge.graphic3d.fill_array_of_points_colors(cpp_ptr, np_colors)

    # Prüfung des geklammerten Punkts (Index 1 in NumPy -> Index 2 in OCCT)
    c_clamped = occ_array.VertexColor(2)
    assert np.isclose(c_clamped.Red(), 0., atol=0.01), "Red not clamped to 0.0!"
    assert np.isclose(c_clamped.Green(), 1.0, atol=0.01), "Green not clamped to 1.0!"
    assert np.isclose(c_clamped.Blue(), 0.5, atol=0.01), "Blau changed!"


# --- 3. TESTS FÜR FEHLERBEHANDLUNG & SAFETY CHECKS ---

def test_invalid_numpy_shape_raises_error():
    """Eine falschen NumPy-Form (z. B. N x 4 statt N x 3) muss eine Exception werfen."""
    np_wrong_shape = np.random.rand(100, 4).astype(np.float64)
    occ_array = Graphic3d_ArrayOfPoints(100, 0)
    cpp_ptr = get_cpp_ptr(occ_array)

    with pytest.raises(RuntimeError, match="shape \(N, 3\)"):
        occ_bridge.graphic3d.fill_array_of_points_coords(cpp_ptr, np_wrong_shape)


def test_buffer_overflow_protection():
    """Wenn das NumPy-Array größer ist als der OCCT-Puffer, muss abgebrochen werden."""
    np_large = np.random.rand(500, 3).astype(np.float64)
    # Wir allokieren Speicher für nur 100 Punkte
    occ_array = Graphic3d_ArrayOfPoints(100, 0)
    cpp_ptr = get_cpp_ptr(occ_array)

    with pytest.raises(RuntimeError, match="Insufficient vertex memory"):
        occ_bridge.graphic3d.fill_array_of_points_coords(cpp_ptr, np_large)


def test_colors_without_flag_raises_error():
    """Farben in ein Array ohne Farb-Flag zu schreiben, muss eine Exception auslösen."""
    np_colors = np.random.rand(50, 3).astype(np.float64)
    # WICHTIG: Kein Farbflag gesetzt!
    occ_array = Graphic3d_ArrayOfPoints(50, 0)
    cpp_ptr = get_cpp_ptr(occ_array)

    with pytest.raises(RuntimeError, match="without color support"):
        occ_bridge.graphic3d.fill_array_of_points_colors(cpp_ptr, np_colors)


# --- 4. TESTS FÜR DIE PYTHON HELPER KLASSE ---

def test_helper_create_from_numpy_complete():
    """Testet den End-to-End Workflow über die Factory-Methode des Helpers."""
    num_points = 2_000
    np_coords = np.random.rand(num_points, 3).astype(np.float64)
    np_colors = np.random.rand(num_points, 3).astype(np.float64)

    # Factory-Aufruf
    occ_array = Graphic3d_ArrayOfPoints_Helper.create_from_numpy(np_coords, np_colors)

    assert occ_array.VertexNumberAllocated() == num_points
    assert occ_array.HasVertexColors() is True

    # Stichprobe am Ende
    p_last = occ_array.Vertice(num_points)
    assert np.isclose(p_last.Z(), np_coords[-1, 2])

def test_helper_populate_from_numpy():
    """Testet den End-to-End Workflow über die Factory-Methode des Helpers."""
    num_points = 2_000
    np_coords = np.random.rand(num_points, 3).astype(np.float64)
    np_colors = np.random.rand(num_points, 3).astype(np.float64)

    occ_array = Graphic3d_ArrayOfPoints(num_points, 2)
    Graphic3d_ArrayOfPoints_Helper.set_coordinates(occ_array, np_coords)
    Graphic3d_ArrayOfPoints_Helper.set_colors(occ_array, np_colors)

    assert occ_array.VertexNumberAllocated() == num_points
    assert occ_array.HasVertexColors() is True

    # Stichprobe am Ende
    p_last = occ_array.Vertice(num_points)
    assert np.isclose(p_last.Z(), np_coords[-1, 2])


def test_helper_read_from_numpy():
    """Testet den End-to-End Workflow über die Factory-Methode des Helpers."""
    num_points = 2_000
    np_coords = np.random.rand(num_points, 3).astype(np.float64)
    np_colors = np.random.rand(num_points, 3).astype(np.float64)

    occ_array = Graphic3d_ArrayOfPoints(num_points, 2)
    Graphic3d_ArrayOfPoints_Helper.set_coordinates(occ_array, np_coords)
    Graphic3d_ArrayOfPoints_Helper.set_colors(occ_array, np_colors)

    np_coords_read = Graphic3d_ArrayOfPoints_Helper.get_coordinates(occ_array)
    np_colors_read = Graphic3d_ArrayOfPoints_Helper.get_colors(occ_array)

    assert np.isclose(np_coords, np_coords_read).all()
    assert np.isclose(np_colors, np_colors_read, atol=0.01).all()


def test_helper_handles_non_contiguous_arrays():
    """Der Helper muss auch mit zerschnittenen (slicing) oder float32 Arrays klarkommen."""
    # Erstelle ein 1000x3 Array, nimm aber via Slicing nur jede zweite Zeile
    raw_coords = np.random.rand(1000, 3).astype(np.float32)  # Absichtlich float32!
    sliced_coords = raw_coords[::2, :]  # Ist im Speicher nicht mehr zusammenhängend

    # Das darf nicht abstürzen, da der Helper ascontiguousarray und float64 erzwingt
    occ_array = Graphic3d_ArrayOfPoints_Helper.create_from_numpy(sliced_coords)

    assert occ_array.VertexNumberAllocated() == 500
    p1 = occ_array.Vertice(1)
    assert np.isclose(p1.X(), float(sliced_coords[0, 0]), atol=1e-6)
