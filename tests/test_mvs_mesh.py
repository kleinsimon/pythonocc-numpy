import pytest
import numpy as np
from OCC.Core.MeshVS import MeshVS_Mesh
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger

from occ_numpy_bridge import occ_bridge, MeshVS_Mesh_Helper


def generate_terrain_mesh(grid_size=100):
    """
    Generiert ein großes 3D-Terrain-Mesh (Grid) rein vektorisiert über NumPy.

    Args:
        grid_size (int): Anzahl der Knoten pro Kante.
                         grid_size=100 -> 10.000 Knoten, 19.602 Dreiecke.
                         grid_size=500 -> 250.000 Knoten, fast 500.000 Dreiecke.

    Returns:
        nodes (np.ndarray): (N, 3) Array mit Koordinaten (float64)
        elements (np.ndarray): (M, 3) Array mit Dreiecks-Indizes (int32)
        colors (np.ndarray): (N, 3) Array mit RGB-Werten von 0.0 bis 1.0 (float64)
    """
    print(f"Generiere Terrain mit {grid_size}x{grid_size} Grid...")

    # 1. Koordinaten generieren (float64 ist zwingend für die OCC-Brücke)
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # Z-Koordinaten als mathematische Funktion (Berg- und Tallandschaft)
    zv = np.sin(xv) * np.cos(yv) * 2.0

    # Zusammenbauen und sicherstellen, dass der Speicher flach im RAM liegt
    nodes = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)

    # 2. Elemente (Triangles) vektorisiert generieren (int32 für OCC)
    # Wir erstellen 2D-Gitter für die Indizes (ohne die letzte Spalte/Zeile)
    i, j = np.meshgrid(np.arange(grid_size - 1), np.arange(grid_size - 1), indexing='ij')

    # Indizes der vier Ecken jedes Grid-Quadrats berechnen (0-basiert für NumPy)
    top_left = i * grid_size + j
    top_right = top_left + 1
    bottom_left = (i + 1) * grid_size + j
    bottom_right = bottom_left + 1

    # Jedes Quadrat wird in zwei Dreiecke geteilt
    triangles1 = np.stack([top_left.ravel(), bottom_left.ravel(), top_right.ravel()], axis=1)
    triangles2 = np.stack([top_right.ravel(), bottom_left.ravel(), bottom_right.ravel()], axis=1)

    elements = np.vstack([triangles1, triangles2])
    elements = np.ascontiguousarray(elements, dtype=np.int32)

    # 3. Farben (Heatmap) generieren
    # Normalisiere die Z-Höhe auf den Bereich [0.0, 1.0]
    z_min, z_max = zv.min(), zv.max()
    z_normalized = (zv.ravel() - z_min) / (z_max - z_min)

    # Blau (Tal) zu Rot (Berg). Grün bleibt 0.2 für etwas Kontrast.
    colors = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    colors[:, 0] = z_normalized  # Roter Kanal (hoch = Berg)
    colors[:, 1] = 0.2  # Grüner Kanal (konstant)
    colors[:, 2] = 1.0 - z_normalized  # Blauer Kanal (tief = Tal)
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    print(f"Fertig! {len(nodes)} Knoten und {len(elements)} Dreiecke generiert.")
    return nodes, elements, colors


@pytest.fixture
def valid_mesh_data():
    """Stellt ein minimales, gültiges Mesh (1 Dreieck, 3 Knoten) bereit."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)

    elements = np.array([
        [0, 1, 2]
    ], dtype=np.int32)

    return nodes, elements


def test_datasource_assignment(valid_mesh_data):
    """Testet, ob die Zuweisung in C++ ohne Absturz funktioniert und das Handle existiert."""
    nodes, elements = valid_mesh_data

    mesh = MeshVS_Mesh()
    ptr = int(mesh.this.this)

    # Brücke aufrufen
    occ_bridge.meshvs.assign_numpy_datasource_to_mesh(ptr, nodes, elements, None)

    # Prüfen, ob OpenCASCADE die DataSource intern registriert hat
    ds_handle = mesh.GetDataSource()
    assert ds_handle.DynamicType().Name() == "NumpyMeshDataSource", "Die DataSource wurde dem Mesh nicht korrekt zugewiesen!"


def test_id_mapping_1_based(valid_mesh_data):
    """Testet, ob die C++ Klasse die 0-basierten NumPy Indizes korrekt auf 1-basierte OCC IDs umrechnet."""
    nodes, elements = valid_mesh_data

    mesh = MeshVS_Mesh()
    occ_bridge.meshvs.assign_numpy_datasource_to_mesh(int(mesh.this.this), nodes, elements, None)
    ds = mesh.GetDataSource()

    # 1. Knoten-IDs prüfen
    node_map = ds.GetAllNodes()
    assert node_map.Extent() == 3, "Es müssen exakt 3 Knoten existieren."
    assert node_map.Contains(1), "OCC Node-ID 1 fehlt (NumPy Index 0)."
    assert node_map.Contains(3), "OCC Node-ID 3 fehlt (NumPy Index 2)."
    assert not node_map.Contains(0), "OCC Node-IDs dürfen niemals 0 sein!"

    # 2. Element-IDs prüfen
    elem_map = ds.GetAllElements()
    assert elem_map.Extent() == 1, "Es muss exakt 1 Element (Dreieck) existieren."
    assert elem_map.Contains(1), "OCC Element-ID 1 fehlt."


def test_get_geom_coordinates(valid_mesh_data):
    """Testet, ob OpenCASCADE die korrekten XYZ Koordinaten aus dem NumPy Array lesen kann."""
    nodes, elements = valid_mesh_data

    mesh = MeshVS_Mesh()
    occ_bridge.meshvs.assign_numpy_datasource_to_mesh(int(mesh.this.this), nodes, elements, None)
    ds = mesh.GetDataSource()

    # OCC Arrays sind zwingend 1-basiert
    coords = TColStd_Array1OfReal(1, 3)
    nb_nodes = 0
    geom_type = 0

    # Abfrage des zweiten Knotens (OCC ID 2 -> NumPy Index 1: [1.0, 0.0, 0.0])
    success = ds.GetGeom(2, False, coords)

    assert success[0] is True
    assert np.isclose(coords.Value(1), 1.0), "X-Koordinate falsch"
    assert np.isclose(coords.Value(2), 0.0), "Y-Koordinate falsch"
    assert np.isclose(coords.Value(3), 0.0), "Z-Koordinate falsch"


def test_invalid_nodes_shape():
    """Sicherheits-Check: Falsche Dimensionen für Knoten müssen einen Fehler werfen."""
    mesh = MeshVS_Mesh()
    ptr = int(mesh.this.this)

    # Falsche Shape: (3, 2) statt (N, 3) für 3D Koordinaten
    bad_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    elements = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="Nodes must be shape"):
        occ_bridge.meshvs.assign_numpy_datasource_to_mesh(ptr, bad_nodes, elements, None)


def test_invalid_elements_shape(valid_mesh_data):
    """Sicherheits-Check: Falsche Dimensionen für Elemente müssen einen Fehler werfen."""
    nodes, _ = valid_mesh_data
    mesh = MeshVS_Mesh()
    ptr = int(mesh.this.this)

    # Falsche Shape: Vierecke (N, 4) statt Dreiecke (N, 3)
    bad_elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="Elements must be shape"):
        occ_bridge.meshvs.assign_numpy_datasource_to_mesh(ptr, nodes, bad_elements, None)

def test_mesh_helper():
    verts, faces, colors = generate_terrain_mesh(100)

    mesh = MeshVS_Mesh_Helper.create_mesh_with_numpy_source(verts, faces)
    ds = mesh.GetDataSource()

    # 1. Knoten-IDs prüfen
    node_map = ds.GetAllNodes()
    assert node_map.Extent() == verts.shape[0], f"Es müssen exakt {verts.shape[0]} Knoten existieren."

    # 2. Element-IDs prüfen
    elem_map = ds.GetAllElements()
    assert elem_map.Extent() == faces.shape[0], f"Es müssen exakt {faces.shape[0]} Elemente existieren."

    np_verts = MeshVS_Mesh_Helper.read_numpy_data_source_vertices(ds)
    np_faces = MeshVS_Mesh_Helper.read_numpy_data_source_faces(ds)
    np_normals = MeshVS_Mesh_Helper.read_numpy_data_source_normals(ds)

    assert np.allclose(verts, np_verts), "Vertices Read mismatch"
    assert np_normals is None, "Normals not None"
    assert np.equal(np_faces, faces).all(), "Faces Read mismatch"
