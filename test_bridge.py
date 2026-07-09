import time
import numpy as np
from OCC.Core.Graphic3d import Graphic3d_ArrayOfPoints

# Unser neu kompiliertes C++ Modul importieren
import occ_bridge

def test_point_cloud_bridge():
    num_points = 500_000  # 500.000 Punkte zum Testen
    print(f"1. Generiere {num_points} zufällige 3D-Punkte in NumPy...")
    np_data = np.random.rand(num_points, 3).astype(np.float64) * 100.0

    print("2. Erstelle leeres Graphic3d_ArrayOfPoints in pythonocc...")
    # Speicher in OpenCASCADE allokieren (maximale Punktzahl angeben)
    occ_array = Graphic3d_ArrayOfPoints(num_points)

    print("3. Extrahiere C++ Speicheradresse aus SWIG...")
    # Bei pythonocc verweist '.this' auf den SWIG-Wrapper. 
    # int() konvertiert die eigentliche C++ Speicheradresse in einen Integer.
    cpp_pointer_int = int(occ_array.this)
    print(f"   -> C++ Speicheradresse: {hex(cpp_pointer_int)}")

    print("4. Starte Datenübertragung via C++ Adapter...")
    start_time = time.time()
    
    # Der magische Aufruf: Zero-Copy Pointer-Übergabe!
    occ_bridge.fill_points_from_numpy(cpp_pointer_int, np_data)
    
    duration = time.time() - start_time
    print(f"   -> Fertig in {duration:.4f} Sekunden!")

    # 5. Validierung: Wir prüfen den 1. und den letzten Punkt in OCCT
    p1 = occ_array.Vertice(1)  # Achtung: 1-basiert in OCCT
    p_last = occ_array.Vertice(num_points)
    
    print("\n--- Validierung ---")
    print(f"NumPy  [0]:   x={np_data[0,0]:.4f}, y={np_data[0,1]:.4f}, z={np_data[0,2]:.4f}")
    print(f"OCCT   [1]:   x={p1.X():.4f}, y={p1.Y():.4f}, z={p1.Z():.4f}")
    print(f"NumPy  [-1]:  x={np_data[-1,0]:.4f}, y={np_data[-1,1]:.4f}, z={np_data[-1,2]:.4f}")
    print(f"OCCT   [end]: x={p_last.X():.4f}, y={p_last.Y():.4f}, z={p_last.Z():.4f}")
    
    assert np.isclose(p1.X(), np_data[0, 0]), "Fehler beim ersten Punkt!"
    assert np.isclose(p_last.Z(), np_data[-1, 2]), "Fehler beim letzten Punkt!"
    print("\nSUCCESS: Alle Daten wurden perfekt synchronisiert!")

if __name__ == "__main__":
    test_point_cloud_bridge()