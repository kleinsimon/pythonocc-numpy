#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Graphic3d_ArrayOfPoints.hxx>
#include <Graphic3d_ArrayOfTriangles.hxx>
#include <iostream>
#include <algorithm>
#include <string>

namespace py = pybind11;

// ============================================================================
// 1. GRAPHIC3D_ARRAYOFPOINTS (Point Clouds & Vertex Colors)
// ============================================================================

/**
 * Fills XYZ coordinates into an existing Graphic3d_ArrayOfPoints from a (N, 3) float64 NumPy array.
 */
void fill_array_of_points_coords(uintptr_t occ_ptr_address, py::array np_coords) {
    if (!np_coords.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Coordinate array must have float64 dtype!");
    }

    py::buffer_info buf = np_coords.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Coordinate array must have shape (N, 3) with float64 dtype!");
    }

    int num_points = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfPoints pointer!");
    }

    if (occ_array->VertexNumberAllocated() < num_points) {
        throw std::runtime_error(
            "Graphic3d_ArrayOfPoints: Insufficient vertex memory allocated (" +
            std::to_string(occ_array->VertexNumberAllocated()) + ") for " +
            std::to_string(num_points) + " points!"
        );
    }

    // OCCT vertex indices are 1-based
    for (int i = 0; i < num_points; ++i) {
        occ_array->SetVertice(
            i + 1,
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
}

/**
 * Reads XYZ coordinates from an existing Graphic3d_ArrayOfPoints to a (N, 3) float64 NumPy array.
 */
static py::array_t<double> read_array_of_points_coords(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfPoints pointer!");
    }

    int num_verts = occ_array->VertexNumber();

    auto numpy_array = py::array_t<double>({ num_verts, 3 });
    py::buffer_info buf = numpy_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < num_verts; i++) {
        occ_array->Vertice(
            i + 1,
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
    return numpy_array;
}

/**
 * Fills RGB vertex colors (values 0.0 - 1.0) into an existing Graphic3d_ArrayOfPoints from a (N, 3) float64 NumPy array.
 */
void fill_array_of_points_colors(uintptr_t occ_ptr_address, py::array np_colors) {
    if (!np_colors.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Coordinate array must have float64 dtype!");
    }

    py::buffer_info buf = np_colors.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Color array must have shape (N, 3) with float64 dtype!");
    }

    int num_colors = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfPoints pointer!");
    }

    if (!occ_array->HasVertexColors()) {
        throw std::runtime_error(
            "Graphic3d_ArrayOfPoints was created without color support! "
            "Ensure Graphic3d_ArrayFlags_VertexColor is passed during initialization."
        );
    }

    if (occ_array->VertexNumberAllocated() < num_colors) {
        throw std::runtime_error("Graphic3d_ArrayOfPoints: Insufficient vertex memory allocated for colors!");
    }

    for (int i = 0; i < num_colors; i++) {
        occ_array->SetVertexColor(
            i + 1, 
            std::clamp(ptr[i * 3 + 0], 0.0, 1.0),
            std::clamp(ptr[i * 3 + 1], 0.0, 1.0),
            std::clamp(ptr[i * 3 + 2], 0.0, 1.0)
        );
    }
}

/**
 * Reads RGB vertex colors (values 0.0 - 1.0) from an existing Graphic3d_ArrayOfPoints to a (N, 3) float64 NumPy array.
 */
static py::array_t<double> read_array_of_points_colors(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfPoints*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfPoints pointer!");
    }

    int num_verts = occ_array->VertexNumber();

    auto numpy_array = py::array_t<double>({ num_verts, 3 });
    py::buffer_info buf = numpy_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < num_verts; i++) {
        occ_array->VertexColor(
            i + 1,
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
    return numpy_array;
}


// ============================================================================
// 2. GRAPHIC3D_ARRAYOFTRIANGLES (High-Performance Rendering Meshes)
// ============================================================================

/**
 * Fills XYZ coordinates into a Graphic3d_ArrayOfTriangles from a (N, 3) float64 NumPy array.
 */
void fill_g3d_triangles_coords(uintptr_t occ_ptr_address, py::array np_coords) {
    if (!np_coords.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Coordinate array must have float64 dtype!");
    }

    py::buffer_info buf = np_coords.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Coordinate array must have shape (N, 3)!");
    }

    int num_nodes = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");

    if (occ_array->VertexNumberAllocated() < num_nodes) {
        throw std::runtime_error("Insufficient vertex memory allocated in Graphic3d_ArrayOfTriangles!");
    }

    for (int i = 0; i < num_nodes; i++) {
        occ_array->SetVertice(
            i + 1,
            ptr[i * 3 + 0], 
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
}

/**
 * Reads XYZ coordinates into a Graphic3d_ArrayOfTriangles from a (N, 3) float64 NumPy array.
 */
static py::array_t<double> read_g3d_triangles_coords(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");
    }

    int num_verts = occ_array->VertexNumber();
    int num_coords = num_verts * 3;

    auto numpy_array = py::array_t<double>({ num_verts, 3 });
    py::buffer_info buf = numpy_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < num_verts; i++) {
        occ_array->Vertice(
            i + 1,
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
    return numpy_array;
}

/**
 * Fills vertex normal vectors (NX, NY, NZ) for lighting calculations in a (N, 3) float64 array.
 */
void fill_g3d_triangles_normals(uintptr_t occ_ptr_address, py::array_t<double> np_norms) {
    if (!np_norms.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Coordinate array must have float64 dtype!");
    }

    py::buffer_info buf = np_norms.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Normal vector array must have shape (N, 3)!");
    }

    int num_nodes = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");

    if (!occ_array->HasVertexNormals()) {
        throw std::runtime_error("Graphic3d_ArrayOfTriangles was initialized without vertex normal support!");
    }

    for (int i = 0; i < num_nodes; i++) {
        occ_array->SetVertexNormal(
            i + 1, 
            ptr[i * 3 + 0], 
            ptr[i * 3 + 1], 
            ptr[i * 3 + 2]
        );
    }
}

/**
 * Reads vertex normal vectors (NX, NY, NZ) for lighting calculations in (N, 3) float64 array.
 */
static py::array_t<double> read_g3d_triangles_normals(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");
    }

    if (!occ_array->HasVertexNormals()) {
        throw std::runtime_error("Graphic3d_ArrayOfTriangles was initialized without vertex normal support!");
    }

    int num_verts = occ_array->VertexNumber();
    int num_coords = num_verts * 3;

    auto numpy_array = py::array_t<double>({ num_verts, 3 });
    py::buffer_info buf = numpy_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < num_verts; i++) {
        occ_array->VertexNormal(
            i + 1,
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
    return numpy_array;
}

/**
 * Fills vertex colors (r, g, b) for lighting calculations from a (N, 3) float64 array.
 */
void fill_g3d_triangles_colors(uintptr_t occ_ptr_address, py::array_t<double> np_colors) {
    if (!np_colors.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Coordinate array must have float64 dtype!");
    }

    py::buffer_info buf = np_colors.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Color vector array must have shape (N, 3)!");
    }

    int num_nodes = static_cast<int>(buf.shape[0]);
    double* ptr = static_cast<double*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");

    if (!occ_array->HasVertexColors()) {
        throw std::runtime_error("Graphic3d_ArrayOfTriangles was initialized without vertex color support!");
    }

    for (int i = 0; i < num_nodes; i++) {
        occ_array->SetVertexColor(
            i + 1, 
            std::clamp(ptr[i * 3 + 0], 0.0, 1.0),
            std::clamp(ptr[i * 3 + 1], 0.0, 1.0),
            std::clamp(ptr[i * 3 + 2], 0.0, 1.0)
        );
    }
}

/**
 * Reads vertex colors (r, g, b) for lighting calculations in a (N, 3) float64 array.
 */
static py::array_t<double> read_g3d_triangles_colors(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");
    }

    if (!occ_array->HasVertexColors()) {
        throw std::runtime_error("Graphic3d_ArrayOfTriangles was initialized without vertex color support!");
    }

    int num_verts = occ_array->VertexNumber();

    auto numpy_array = py::array_t<double>({ num_verts, 3 });
    py::buffer_info buf = numpy_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < num_verts; i++) {
        occ_array->VertexColor(
            i + 1,
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
    return numpy_array;
}

/**
 * Fills triangle indices from a (M, 3) int32 NumPy array.
 * Automatically translates 0-based NumPy indices to 1-based OpenCASCADE indices.
 */
void fill_g3d_triangles_indices(uintptr_t occ_ptr_address, py::array numpy_array) {
    if (!numpy_array.dtype().equal(py::dtype::of<int32_t>())) {
        throw std::runtime_error("Triangle index array must have int32 dtype (use .astype(np.int32))!");
    }

    py::buffer_info buf = numpy_array.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Triangle index array must have shape (M, 3)!");
    }

    int num_triangles = static_cast<int>(buf.shape[0]);
    int num_edges = num_triangles * 3;
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);

    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");

    // In Graphic3d, each triangle vertex connection is stored as an "edge" (3 edges per triangle)
    if (occ_array->EdgeNumberAllocated() < num_edges) {
        throw std::runtime_error("Insufficient index memory (edges) allocated in Graphic3d_ArrayOfTriangles!");
    }

    for (int i = 0; i < num_edges; i++) {
        occ_array->AddEdge(ptr[i] + 1);
    }
}

/**
 * Reads triangle indices from a (M, 3) int32 NumPy array.
 * Automatically translates 1-based OpenCASCADE indices to 0-based NumPy indices.
 */
py::array_t<int32_t> read_g3d_triangles_indices(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Graphic3d_ArrayOfTriangles*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Graphic3d_ArrayOfTriangles pointer!");
    }

    int num_triangles = occ_array->EdgeNumber() / 3;

    auto numpy_array = py::array_t<int32_t>({ num_triangles, 3 });
    py::buffer_info buf = numpy_array.request();
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);

    int num_edges = occ_array->EdgeNumber();
    for (int i = 0; i < num_edges; i++) {
        ptr[i] = occ_array->Edge(i + 1) - 1;
    }
    return numpy_array;
}
