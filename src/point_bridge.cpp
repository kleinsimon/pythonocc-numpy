#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Graphic3d_ArrayOfPoints.hxx>
#include <iostream>

namespace py = pybind11;

/**
 * Füllt ein vorhandenes Graphic3d_ArrayOfPoints mit Daten aus einem 2D-NumPy-Array (N x 3).
 * * @param occ_ptr_address Speicheradresse des C++ Objekts als Integer aus pythonocc
 * @param numpy_array     2D NumPy Array mit shape (N, 3) und dtype float64
 */
void fill_points_from_numpy(uintptr_t occ_ptr_address, py::array_t<double> numpy_array) {
    // 1. Validierung des NumPy-Arrays
    py::buffer_info buf = numpy_array.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Das NumPy-Array muss die Form (N, 3) haben!");
    }

    int num_points = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    // 2. Cast: Speicheradresse zurück in den OCCT-Pointer verwandeln
    // Hinweis: Wir gehen davon aus, dass wir den Pointer auf das Array-Objekt erhalten
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Ungültiger OpenCASCADE Pointer!");
    }

    // 3. Sicherheitscheck: Hat das Array genug Speicherplatz?
    if (occ_array->VertexNumber() < num_points) {
        throw std::runtime_error("Das Graphic3d_ArrayOfPoints ist zu klein für das NumPy-Array!");
    }

    // 4. Daten extrem schnell in C++ übertragen (ohne Python GIL)
    // OCCT Indizes sind 1-basiert!
    for (int i = 0; i < num_points; ++i) {
        double x = ptr[i * 3 + 0];
        double y = ptr[i * 3 + 1];
        double z = ptr[i * 3 + 2];
        
        occ_array->SetVertice(i + 1, x, y, z);
    }
}

// Pybind11 Modul-Definition
PYBIND11_MODULE(occ_bridge, m) {
    m.doc() = "Native C++ Brücke zwischen pythonocc-core und NumPy";
    m.def("fill_points_from_numpy", &fill_points_from_numpy, 
          "Füllt Graphic3d_ArrayOfPoints extrem schnell aus NumPy (N, 3)");
}