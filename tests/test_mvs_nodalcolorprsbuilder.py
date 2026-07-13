import pytest
import numpy as np

from OCC.Core.MeshVS import MeshVS_Mesh, MeshVS_NodalColorPrsBuilder

from occ_numpy_bridge.occ_bridge import meshvs
from occ_numpy_bridge.mvs_nodalcolorprsbuilder import MeshVS_NodalColorPrsBuilder_Helper



@pytest.fixture
def empty_meshvs_builder():
    """
    Erstellt einen leeren MeshVS_NodalColorPrsBuilder für die Tests.
    Ein Builder benötigt in OpenCASCADE zwingend ein Parent-Mesh.
    """
    mesh = MeshVS_Mesh()
    # 0 sind die Standard-Flags für den Builder
    builder = MeshVS_NodalColorPrsBuilder(mesh, 0)
    return builder


def test_write_and_read_colors_with_explicit_indices(empty_meshvs_builder):
    """
    Testet das Schreiben und Lesen von Farben mit völlig willkürlichen Node-IDs.
    """
    # 1. Testdaten vorbereiten
    # Willkürliche IDs, unkorreliert zur Array-Länge
    input_ids = np.array([42, 1024, 7], dtype=np.int32) 
    
    input_colors = np.array([
        [1.0, 0.0, 0.0],  # Rot für ID 42
        [0.0, 1.0, 0.0],  # Grün für ID 1024
        [0.0, 0.0, 1.0]   # Blau für ID 7
    ], dtype=np.float64)

    ptr = int(empty_meshvs_builder.this.this)

    # 2. Schreiben in C++ (Dein neues Feature mit Indizes)
    meshvs.fill_meshvs_nodal_color_prs_builder_colors(ptr, input_colors, input_ids)

    # 3. Lesen aus C++ (Rückgabe als Tuple)
    out_ids, out_colors = meshvs.read_meshvs_nodal_color_prs_builder_colors(ptr)

    # 4. Validierung
    assert out_ids.shape == (3,), "Falsche Anzahl an IDs zurückgegeben!"
    assert out_colors.shape == (3, 3), "Falsche Shape für das Farben-Array!"

    # ACHTUNG: Da C++ eine Hash-Map nutzt, ist die Reihenfolge unbestimmt!
    # Wir müssen beide Arrays anhand der Node-IDs aufsteigend sortieren.
    sort_indices_out = np.argsort(out_ids)
    sorted_out_ids = out_ids[sort_indices_out]
    sorted_out_colors = out_colors[sort_indices_out]

    sort_indices_in = np.argsort(input_ids)
    sorted_in_ids = input_ids[sort_indices_in]
    sorted_in_colors = input_colors[sort_indices_in]

    # IDs müssen exakt übereinstimmen
    assert np.array_equal(sorted_out_ids, sorted_in_ids), "Die Node-IDs wurden fehlerhaft gespeichert!"
    
    # Farben müssen wegen 8-Bit-OCCT-Quantisierung mit Toleranz geprüft werden
    assert np.allclose(sorted_out_colors, sorted_in_colors, atol=1e-2), "Die Farben weichen zu stark ab!"


def test_write_and_read_colors_without_indices_fallback(empty_meshvs_builder):
    """
    Testet das Schreiben von Farben ohne explizite Indizes. 
    Die C++ Logik soll automatisch 1-basierte Indizes (1, 2, 3...) vergeben.
    """
    input_colors = np.array([
        [0.1, 0.2, 0.3],
        [0.9, 0.8, 0.7]
    ], dtype=np.float64)

    ptr = int(empty_meshvs_builder.this.this)

    # Schreiben OHNE das optionale Index-Array
    meshvs.fill_meshvs_nodal_color_prs_builder_colors(ptr, input_colors, None)

    # Lesen
    out_ids, out_colors = meshvs.read_meshvs_nodal_color_prs_builder_colors(ptr)

    # Sortieren für konsistenten Check
    sort_indices_out = np.argsort(out_ids)
    sorted_out_ids = out_ids[sort_indices_out]
    sorted_out_colors = out_colors[sort_indices_out]

    # Erwartete Indizes: OpenCASCADE ist 1-basiert, also [1, 2]
    expected_ids = np.array([0, 1], dtype=np.int32)

    assert np.array_equal(sorted_out_ids, expected_ids), "Fallback auf 1-basierte Indizes ist fehlgeschlagen!"
    assert np.allclose(sorted_out_colors, input_colors, atol=1e-2), "Farben weichen ab!"


def test_meshvs_invalid_arrays_throw_errors(empty_meshvs_builder):
    """
    Testet, ob die C++ Typ-Wächter (Dtype-Guards) korrekt greifen.
    """
    ptr = int(empty_meshvs_builder.this.this)
    
    # Falscher Datentyp für Farben (float32 statt float64)
    bad_colors = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="float64"):
        meshvs.fill_meshvs_nodal_color_prs_builder_colors(ptr, bad_colors, None)

    # Falscher Datentyp für Indizes (float64 statt int32)
    good_colors = np.zeros((5, 3), dtype=np.float64)
    bad_indices = np.zeros(5, dtype=np.float64)
    with pytest.raises(RuntimeError, match="int32"):
        meshvs.fill_meshvs_nodal_color_prs_builder_colors(ptr, good_colors, bad_indices)

    # Längen-Mismatch: 5 Farben, aber nur 4 Indizes
    mismatch_indices = np.array([1, 2, 3, 4], dtype=np.int32)
    with pytest.raises(RuntimeError, match="match the number of colors"):
        meshvs.fill_meshvs_nodal_color_prs_builder_colors(ptr, good_colors, mismatch_indices)


def test_meshvs_wrapper(empty_meshvs_builder):
    builder = empty_meshvs_builder

    colors = np.random.random((2000, 3))
    idx = np.arange(2000)

    MeshVS_NodalColorPrsBuilder_Helper.set_colors(builder, colors)

    cpp_ids, cpp_colors = MeshVS_NodalColorPrsBuilder_Helper.get_colors(builder)

    order = np.argsort(cpp_ids)

    assert np.equal(cpp_ids[order], idx[order]).all(), "Index mismatch"
    assert np.allclose(cpp_colors[order, :], colors, atol=0.02), "Color mismatch"


def test_meshvs_wrapper_idx(empty_meshvs_builder):
    builder = empty_meshvs_builder

    colors = np.random.random((2000, 3))
    idx = np.arange(2000)

    MeshVS_NodalColorPrsBuilder_Helper.set_colors(builder, colors, idx)

    cpp_ids, cpp_colors = MeshVS_NodalColorPrsBuilder_Helper.get_colors(builder)

    order = np.argsort(cpp_ids)

    assert np.equal(cpp_ids[order], idx).all(), "Index mismatch"
    assert np.allclose(cpp_colors[order, :], colors, atol=0.02), "Color mismatch"
