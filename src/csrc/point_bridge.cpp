#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Graphic3d_ArrayOfPoints.hxx>
#include <iostream>

namespace py = pybind11;

void fill_graphic3d_array_of_points_coords(uintptr_t occ_ptr_address, py::array_t<double> numpy_array) {
    py::buffer_info buf = numpy_array.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("The NumPy array for coordinates must have shape (N, 3)!");
    }

    int num_points = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid OpenCASCADE pointer (expected Graphic3d_ArrayOfPoints)!");
    }

    if (occ_array->VertexNumberAllocated() < num_points) {
        throw std::runtime_error(
            "Graphic3d_ArrayOfPoints: Not enough memory allocated (" +
            std::to_string(occ_array->VertexNumberAllocated()) + ") for " +
            std::to_string(num_points) + " points!"
        );
    }

    for (int i = 0; i < num_points; ++i) {
        occ_array->SetVertice(i + 1, ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
    }
}

void fill_graphic3d_array_of_points_colors(uintptr_t occ_ptr_address, py::array_t<double> numpy_array) {
    py::buffer_info buf = numpy_array.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("The NumPy array for colors must have shape (N, 3)!");
    }

    int num_colors = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid OpenCASCADE pointer!");
    }

    if (!occ_array->HasVertexColors()) {
        throw std::runtime_error(
            "Graphic3d_ArrayOfPoints was created without color support! "
            "Use Graphic3d_ArrayFlags_VertexColor during initialization in Python."
        );
    }

    if (occ_array->VertexNumberAllocated() < num_colors) {
        throw std::runtime_error("Graphic3d_ArrayOfPoints: Not enough memory allocated for color values!");
    }

    for (int i = 0; i < num_colors; ++i) {
        double r = std::clamp(ptr[i * 3 + 0], 0.0, 1.0);
        double g = std::clamp(ptr[i * 3 + 1], 0.0, 1.0);
        double b = std::clamp(ptr[i * 3 + 2], 0.0, 1.0);

        occ_array->SetVertexColor(i + 1, r, g, b);
    }
}

PYBIND11_MODULE(occ_bridge, m) {
    m.doc() = "Native C++ bridge between pythonocc-core and NumPy";

    py::module_ g3d = m.def_submodule("graphic3d", "Bridge for OCC.Core.Graphic3d");

    g3d.def("fill_array_of_points_coords", &fill_graphic3d_array_of_points_coords,
            "Fills coordinates in an existing Graphic3d_ArrayOfPoints (expects an N x 3 float64 array)");

    g3d.def("fill_array_of_points_colors", &fill_graphic3d_array_of_points_colors,
            "Fills RGB colors in an existing Graphic3d_ArrayOfPoints (expects an N x 3 float64 array, values from 0.0 to 1.0). The array must be created with the 'Vertex Color' flag (2)");

    m.def("fill_graphic3d_array_of_points_coords", &fill_graphic3d_array_of_points_coords);
    m.def("fill_graphic3d_array_of_points_colors", &fill_graphic3d_array_of_points_colors);
}