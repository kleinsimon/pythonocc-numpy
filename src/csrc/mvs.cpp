#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Quantity_Color.hxx>
#include <MeshVS_NodalColorPrsBuilder.hxx>
#include <iostream>
#include <algorithm>
#include <optional>
#include <string>
#include <tuple>

namespace py = pybind11;

/**
 * Fills RGB vertex colors (values 0.0 - 1.0) into an existing MeshVS_NodalColorPrsBuilder.
 * Optionally maps them to specific Node IDs via an (N,) int32 numpy array.
 */
void fill_nodal_color_prs_builder_colors(uintptr_t occ_ptr_address,
    py::array np_colors,
    std::optional<py::array> np_indices = std::nullopt) {

    if (!np_colors.dtype().equal(py::dtype::of<double>())) {
        throw std::runtime_error("Color array must have float64 dtype!");
    }

    py::buffer_info buf_colors = np_colors.request();
    if (buf_colors.ndim != 2 || buf_colors.shape[1] != 3) {
        throw std::runtime_error("Color array must have shape (N, 3) with float64 dtype!");
    }

    int num_colors = static_cast<int>(buf_colors.shape[0]);
    double* ptr_colors = static_cast<double*>(buf_colors.ptr);

    int32_t* ptr_indices = nullptr;
    if (np_indices.has_value()) {
        if (!np_indices->dtype().equal(py::dtype::of<int32_t>())) {
            throw std::runtime_error("Indices array must have int32 dtype!");
        }

        py::buffer_info buf_indices = np_indices->request();
        if (buf_indices.ndim != 1 || buf_indices.shape[0] != num_colors) {
            throw std::runtime_error("Indices array must be 1D and match the number of colors (N,)!");
        }
        ptr_indices = static_cast<int32_t*>(buf_indices.ptr);
    }

    auto* builder = reinterpret_cast<MeshVS_NodalColorPrsBuilder*>(occ_ptr_address);
    if (!builder) {
        throw std::runtime_error("Invalid MeshVS_NodalColorPrsBuilder pointer!");
    }

    for (int i = 0; i < num_colors; i++) {
        auto color = Quantity_Color(
            std::clamp(ptr_colors[i * 3 + 0], 0.0, 1.0),
            std::clamp(ptr_colors[i * 3 + 1], 0.0, 1.0),
            std::clamp(ptr_colors[i * 3 + 2], 0.0, 1.0),
            Quantity_TOC_RGB
        );

        int node_id = (ptr_indices ? ptr_indices[i] : i) + 1;
        builder->SetColor(node_id, color);
    }
}

/**
 * Reads RGB vertex colors (values 0.0 - 1.0) from an existing MeshVS_NodalColorPrsBuilder to a (N, 3) float64 NumPy array.
 */
static std::tuple<py::array_t<int32_t>, py::array_t<double>> read_nodal_color_prs_builder_colors(uintptr_t occ_ptr_address) {
    auto* builder = reinterpret_cast<MeshVS_NodalColorPrsBuilder*>(occ_ptr_address);
    if (!builder) {
        throw std::runtime_error("Invalid MeshVS_NodalColorPrsBuilder pointer!");
    }

    auto map = builder->GetColors();
    int num_verts = map.Size();

    auto np_colors = py::array_t<double>({ num_verts, 3 });
    auto np_node_ids = py::array_t<int32_t>({ num_verts });

    py::buffer_info buf_colors = np_colors.request();
    double* ptr_colors = static_cast<double*>(buf_colors.ptr);

    py::buffer_info buf_ids = np_node_ids.request();
    int32_t* ptr_ids = static_cast<int32_t*>(buf_ids.ptr);

    MeshVS_DataMapIteratorOfDataMapOfIntegerColor it(map);

    for (int i = 0; it.More(); it.Next(), i++) {
        ptr_ids[i] = it.Key() - 1;
        Quantity_Color color = it.Value();

        ptr_colors[i * 3 + 0] = color.Red();
        ptr_colors[i * 3 + 1] = color.Green();
        ptr_colors[i * 3 + 2] = color.Blue();
    }
    return std::make_tuple(np_node_ids, np_colors);
}