#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "g3d.cpp"
#include "poly.cpp"
#include "mvs.cpp"
#include "mds.cpp"


PYBIND11_MODULE(occ_bridge, m) {
    m.doc() = "High-performance C++ bridge between pythonocc-core and NumPy";

    // Submodule: graphic3d
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

    // Submodule: poly
    py::module_ poly = m.def_submodule("poly", "Adapters for OCC.Core.Poly classes");
    poly.def("fill_poly_triangulation", &fill_poly_triangulation, "Fills Poly_Triangulation nodes and triangles from NumPy");
    poly.def("fill_poly_triangulation_coords", &fill_poly_triangulation_coords, "Fills Poly_Triangulation vertex coordinates (N x 3 float64)");
    poly.def("read_poly_triangulation_coords", &read_poly_triangulation_coords, "Read Poly_Triangulation vertex coordinates (N x 3 float64)");
    poly.def("fill_poly_triangulation_indices", &fill_poly_triangulation_indices, "Fills Poly_Triangulation face indices (M x 3 int32, auto 1-based shift)");
    poly.def("read_poly_triangulation_indices", &read_poly_triangulation_indices, "Read Poly_Triangulation face indices (M x 3 int32, auto 0-based shift)");

    // Submodule: meshVS
    py::module_ meshvs = m.def_submodule("meshvs", "Adapters for OCC.Core.MeshVS classes");
    meshvs.def("fill_meshvs_nodal_color_prs_builder_colors", &fill_nodal_color_prs_builder_colors, "Fills MeshVS_NodalColorPrsBuilder colors from NumPy");
    meshvs.def("read_meshvs_nodal_color_prs_builder_colors", &read_nodal_color_prs_builder_colors, "Reads MeshVS_NodalColorPrsBuilder colors to NumPy");

    meshvs.def("assign_numpy_datasource_to_mesh", &assign_numpy_datasource_to_mesh, "Creates a Datasource from numpy arrays and assigns it to a MeshVS_Mesh");
    meshvs.def("read_numpy_datasource_vertices", &read_numpy_datasource_vertices, "Read the vertices / nodes from a numpy datasource");
    meshvs.def("read_numpy_datasource_faces", &read_numpy_datasource_faces, "Read the faces / elements from a numpy datasource");
    meshvs.def("read_numpy_datasource_normals", &read_numpy_datasource_normals, "Read the normals from a numpy datasource");
}