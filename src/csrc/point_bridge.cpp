#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Graphic3d_ArrayOfPoints.hxx>
#include <Graphic3d_ArrayOfTriangles.hxx>
#include <Poly_Triangulation.hxx>
#include <Poly_Triangle.hxx>
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


// ============================================================================
// MODULE REGISTRATION
// ============================================================================
PYBIND11_MODULE(occ_bridge, m) {
    m.doc() = "High-performance zero-copy C++ bridge between pythonocc-core and NumPy";

    // Submodule: graphic3d (For OpenGL visualization & AIS_Shape/AIS_PointCloud)
    py::module_ g3d = m.def_submodule("graphic3d", "Adapters for OCC.Core.Graphic3d classes");
    g3d.def("fill_array_of_points_coords", &fill_array_of_points_coords, "Fills point coordinates (N x 3 float64)");
    g3d.def("fill_array_of_points_colors", &fill_array_of_points_colors, "Fills RGB vertex colors (N x 3 float64, range 0.0-1.0)");
    g3d.def("read_array_of_points_coords", &read_array_of_points_coords, "Reads point coordinates (N x 3 float64)");
    g3d.def("read_array_of_points_colors", &read_array_of_points_colors, "Reads RGB vertex colors (N x 3 float64, range 0.0-1.0)");

    g3d.def("fill_array_of_triangles_coords", &fill_g3d_triangles_coords, "Fills mesh vertex coordinates (N x 3 float64)");
    g3d.def("fill_array_of_triangles_colors", &fill_g3d_triangles_colors, "Fills vertex colors (N x 3 float64)");
    g3d.def("fill_array_of_triangles_normals", &fill_g3d_triangles_normals, "Fills vertex normals (N x 3 float64)");
    g3d.def("fill_array_of_triangles_indices", &fill_g3d_triangles_indices, "Fills triangle indices (M x 3 int32, auto 1-based shift)");
    g3d.def("read_array_of_triangles_indices", &read_g3d_triangles_indices, "Reads triangle indices (M x 3 int32, 0-based)");
    g3d.def("read_array_of_triangles_coords", &read_g3d_triangles_coords, "Reads vertex coordinates (M x 3 double)");
    g3d.def("read_array_of_triangles_colors", &read_g3d_triangles_colors, "Reads vertex colors (M x 3 double)");
    g3d.def("read_array_of_triangles_normals", &read_g3d_triangles_normals, "Reads vertex normals (M x 3 double)");

    // Submodule: mesh (For CAD geometry, BRep operations & STL export)
    py::module_ mesh = m.def_submodule("mesh", "Adapters for OCC.Core.Poly classes");
    mesh.def("fill_poly_triangulation", &fill_poly_triangulation, "Fills Poly_Triangulation nodes and triangles from NumPy");
    mesh.def("fill_poly_triangulation_coords", &fill_poly_triangulation_coords, "Fills Poly_Triangulation vertex coordinates (N x 3 float64)");
    mesh.def("read_poly_triangulation_coords", &read_poly_triangulation_coords, "Read Poly_Triangulation vertex coordinates (N x 3 float64)");
    mesh.def("fill_poly_triangulation_indices", &fill_poly_triangulation_indices, "Fills Poly_Triangulation face indices (M x 3 int32, auto 1-based shift)");
    mesh.def("read_poly_triangulation_indices", &read_poly_triangulation_indices, "Read Poly_Triangulation face indices (M x 3 int32, auto 0-based shift)");
}