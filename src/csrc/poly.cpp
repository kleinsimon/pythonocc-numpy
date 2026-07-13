#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Poly_Triangulation.hxx>
#include <Poly_Triangle.hxx>
#include <iostream>
#include <algorithm>
#include <string>

namespace py = pybind11;

// ============================================================================
// 3. POLY_TRIANGULATION (CAD & Geometry Meshes)
// ============================================================================

/**
 * Fills a Poly_Triangulation (CAD geometry mesh) face indices directly from NumPy array.
 */
void fill_poly_triangulation_indices(uintptr_t occ_ptr_address, py::array np_indices) {
    if (!np_indices.dtype().equal(py::dtype::of<int32_t>())) {
        throw std::runtime_error("Index array must have int32 dtype (use .astype(np.int32))!");
    }

    py::buffer_info buf_indices = np_indices.request();

    if (buf_indices.ndim != 2 || buf_indices.shape[1] != 3) {
        throw std::runtime_error("Index array must have shape (M, 3)!");
    }

    int num_triangles = static_cast<int>(buf_indices.shape[0]);

    int32_t* ptr_indices = static_cast<int32_t*>(buf_indices.ptr);

    auto* poly = reinterpret_cast<Poly_Triangulation*>(occ_ptr_address);
    if (!poly) throw std::runtime_error("Invalid Poly_Triangulation pointer!");

    if (poly->NbTriangles() != num_triangles) {
        throw std::runtime_error("Allocated dimensions of Poly_Triangulation do not match NumPy array shapes!");
    }

    for (int i = 0; i < num_triangles; i++) {
        Poly_Triangle tri(
            ptr_indices[i * 3 + 0] + 1,
            ptr_indices[i * 3 + 1] + 1,
            ptr_indices[i * 3 + 2] + 1
        );
        poly->SetTriangle(i + 1, tri);
    }
}

/**
 * Reads a Poly_Triangulation (CAD geometry mesh) face indices directly from NumPy array.
 */
py::array_t<int32_t> read_poly_triangulation_indices(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Poly_Triangulation*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Poly_Triangulation pointer!");
    }

    int num_triangles = occ_array->NbTriangles();

    auto numpy_array = py::array_t<int32_t>({ num_triangles, 3 });
    py::buffer_info buf = numpy_array.request();
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);


    for (int i = 0; i < num_triangles; i++) {
        auto triangle = occ_array->Triangle(i + 1);
        for (int j = 0; j < 3; j++) {
            ptr[i * 3 + j] = triangle.Value(j + 1) - 1;
        }
    }
    return numpy_array;
}

/**
 * Fills a Poly_Triangulation (CAD geometry mesh) vertex coordinates directly from NumPy array.
 */
void fill_poly_triangulation_coords(uintptr_t occ_ptr_address, py::array np_coords) {
    if (!np_coords.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Coordinate array must have float64 dtype!");
    }

    py::buffer_info buf_coords = np_coords.request();

    if (buf_coords.ndim != 2 || buf_coords.shape[1] != 3) {
        throw std::runtime_error("Coordinate array must have shape (N, 3)!");
    }

    int num_nodes = static_cast<int>(buf_coords.shape[0]);

    double* ptr_coords = static_cast<double*>(buf_coords.ptr);

    auto* poly = reinterpret_cast<Poly_Triangulation*>(occ_ptr_address);
    if (!poly) throw std::runtime_error("Invalid Poly_Triangulation pointer!");

    if (poly->NbNodes() != num_nodes) {
        throw std::runtime_error("Allocated dimensions of Poly_Triangulation do not match NumPy array shapes!");
    }

    for (int i = 0; i < num_nodes; i++) {
        gp_Pnt pnt(
            ptr_coords[i * 3 + 0],
            ptr_coords[i * 3 + 1],
            ptr_coords[i * 3 + 2]
        );
        poly->SetNode(i + 1, pnt);
    }
}

/**
 * Reads Poly_Triangulation (CAD geometry mesh) vertex coordinates directly to NumPy array.
 */
static py::array_t<double> read_poly_triangulation_coords(uintptr_t occ_ptr_address) {
    auto* occ_array = reinterpret_cast<Poly_Triangulation*>(occ_ptr_address);
    if (!occ_array) {
        throw std::runtime_error("Invalid Poly_Triangulation pointer!");
    }

    int num_verts = occ_array->NbNodes();

    auto numpy_array = py::array_t<double>({ num_verts, 3 });
    py::buffer_info buf = numpy_array.request();
    double* ptr = static_cast<double*>(buf.ptr);

    for (int i = 0; i < num_verts; i++) {
        auto node = occ_array->Node(i + 1);
        node.Coord(
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }
    return numpy_array;
}

/**
 * Fills a Poly_Triangulation (CAD geometry mesh) directly from NumPy arrays.
 */
void fill_poly_triangulation(uintptr_t occ_ptr_address, py::array np_coords, py::array np_indices) {
    fill_poly_triangulation_coords(occ_ptr_address, np_coords);
    fill_poly_triangulation_indices(occ_ptr_address, np_indices);
}

