#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MeshVS_DataSource.hxx>
#include <MeshVS_EntityType.hxx>
#include <MeshVS_Mesh.hxx>
#include <TColStd_PackedMapOfInteger.hxx>
#include <NCollection_Array1.hxx>
#include <Standard_Version.hxx>
#include <Standard_Type.hxx>
#include <optional>

namespace py = pybind11;

// OCCT 8 replaced the MeshVS_HArray1OfSequenceOfInteger class by a (deprecated) NCollection typedef
#if OCC_VERSION_HEX >= 0x080000
#include <NCollection_HArray1.hxx>
#include <NCollection_Sequence.hxx>
using NumpyDS_HArray1OfSequenceOfInteger = NCollection_HArray1<NCollection_Sequence<int>>;
#else
#include <MeshVS_HArray1OfSequenceOfInteger.hxx>
using NumpyDS_HArray1OfSequenceOfInteger = MeshVS_HArray1OfSequenceOfInteger;
#endif

class NumpyMeshDataSource : public MeshVS_DataSource {
    DEFINE_STANDARD_RTTIEXT(NumpyMeshDataSource, MeshVS_DataSource)

private:
    py::array_t<double> my_nodes;
    py::array_t<int32_t> my_elements;
    std::optional<py::array_t<double>> my_normals;

    std::vector<double> my_node_normals;
    std::vector<double> my_elem_normals;
    std::vector<double> my_elem_normals_raw;
    std::vector<double> my_smooth_normals;
    bool use_computed_normals = false;

    double my_crease_angle_cos = 0.0;

    TColStd_PackedMapOfInteger my_node_ids;
    TColStd_PackedMapOfInteger my_element_ids;

    void compute_normals(double crease_angle_deg) {
        auto nodes_proxy = my_nodes.unchecked<2>();
        auto elems_proxy = my_elements.unchecked<2>();

        Standard_Integer nb_nodes = my_nodes.shape(0);
        Standard_Integer nb_elems = my_elements.shape(0);

        my_crease_angle_cos = std::cos(crease_angle_deg * M_PI / 180.0);

        my_elem_normals.resize(nb_elems * 3);
        my_elem_normals_raw.resize(nb_elems * 3);

        // ── Phase 1: Compute face normals ──
        for (Standard_Integer i = 0; i < nb_elems; ++i) {
            Standard_Integer idx0 = elems_proxy(i, 0);
            Standard_Integer idx1 = elems_proxy(i, 1);
            Standard_Integer idx2 = elems_proxy(i, 2);

            gp_Pnt aP1(nodes_proxy(idx0, 0), nodes_proxy(idx0, 1), nodes_proxy(idx0, 2));
            gp_Pnt aP2(nodes_proxy(idx1, 0), nodes_proxy(idx1, 1), nodes_proxy(idx1, 2));
            gp_Pnt aP3(nodes_proxy(idx2, 0), nodes_proxy(idx2, 1), nodes_proxy(idx2, 2));

            gp_Vec aN = gp_Vec(aP1, aP2).Crossed(gp_Vec(aP1, aP3));

            // Store raw (area-weighted) normal for smooth averaging
            aN.Coord(
                my_elem_normals_raw[i * 3 + 0],
                my_elem_normals_raw[i * 3 + 1],
                my_elem_normals_raw[i * 3 + 2]
            );

            // Store normalized face normal
            if (aN.SquareMagnitude() > Precision::SquareConfusion())
                aN.Normalize();
            else
                aN.SetCoord(0.0, 0.0, 0.0);

            aN.Coord(
                my_elem_normals[i * 3 + 0],
                my_elem_normals[i * 3 + 1],
                my_elem_normals[i * 3 + 2]
            );
        }

        // ── Phase 2: Build node → elements adjacency ──
        std::vector<std::vector<Standard_Integer>> node_to_elems(nb_nodes);
        for (Standard_Integer i = 0; i < nb_elems; ++i) {
            for (int j = 0; j < 3; ++j) {
                node_to_elems[elems_proxy(i, j)].push_back(i);
            }
        }

        // ── Phase 3: Compute per-element-vertex smooth normals ──
        // For each (element, local_vertex) pair, average only those
        // neighbor face normals whose angle to the current face normal
        // is below the crease threshold.
        my_smooth_normals.resize(nb_elems * 9);

        for (Standard_Integer ei = 0; ei < nb_elems; ++ei) {
            double fnx = my_elem_normals[ei * 3 + 0];
            double fny = my_elem_normals[ei * 3 + 1];
            double fnz = my_elem_normals[ei * 3 + 2];

            for (int lv = 0; lv < 3; ++lv) {
                Standard_Integer node_idx = elems_proxy(ei, lv);
                const auto& adj_elems = node_to_elems[node_idx];

                double sx = 0.0, sy = 0.0, sz = 0.0;

                for (Standard_Integer ej : adj_elems) {
                    double onx = my_elem_normals[ej * 3 + 0];
                    double ony = my_elem_normals[ej * 3 + 1];
                    double onz = my_elem_normals[ej * 3 + 2];

                    // dot product of normalized face normals = cos(angle)
                    double dot = fnx * onx + fny * ony + fnz * onz;

                    if (dot >= my_crease_angle_cos) {
                        // Within crease angle → include area-weighted normal
                        sx += my_elem_normals_raw[ej * 3 + 0];
                        sy += my_elem_normals_raw[ej * 3 + 1];
                        sz += my_elem_normals_raw[ej * 3 + 2];
                    }
                }

                double len = std::sqrt(sx * sx + sy * sy + sz * sz);
                if (len > 1e-10) {
                    sx /= len; sy /= len; sz /= len;
                }
                else {
                    // Fallback to face normal
                    sx = fnx; sy = fny; sz = fnz;
                }

                my_smooth_normals[(ei * 3 + lv) * 3 + 0] = sx;
                my_smooth_normals[(ei * 3 + lv) * 3 + 1] = sy;
                my_smooth_normals[(ei * 3 + lv) * 3 + 2] = sz;
            }
        }
    }

public:
    NumpyMeshDataSource(
        py::array_t<double> nodes, 
        py::array_t<int32_t> elements, 
        std::optional<py::array_t<double>> normals,
        double crease_angle_deg = 30.0
    )
        : my_nodes(nodes), my_elements(elements), my_normals(normals) {

        if (nodes.ndim() != 2 || nodes.shape(1) != 3)
            throw std::runtime_error("Nodes must be shape (N, 3)");
        if (elements.ndim() != 2 || elements.shape(1) != 3)
            throw std::runtime_error("Elements must be shape (M, 3) for triangles");

        if (normals.has_value()) {
            if (normals->ndim() != 2 || normals->shape(1) != 3) {
                throw std::runtime_error("Normals must be shape (N, 3)");
            }
            if (normals->shape(0) != my_nodes.shape(0)) {
                throw std::runtime_error("Number of normals must match number of nodes");
            }
        }
        else {
            compute_normals(crease_angle_deg);
            use_computed_normals = true;
        }

        // register IDs
        for (Standard_Integer i = 1; i <= nodes.shape(0); ++i) {
            my_node_ids.Add(i);
        }
        for (Standard_Integer i = 1; i <= elements.shape(0); ++i) {
            my_element_ids.Add(i);
        }
    }

    Standard_Boolean GetGeom(const Standard_Integer ID,
        const Standard_Boolean IsElement,
        NCollection_Array1<double>& Coords,
        Standard_Integer& NbNodes,
        MeshVS_EntityType& Type) const override {

        auto nodes_proxy = my_nodes.unchecked<2>();

        if (!IsElement) {
            if (!my_node_ids.Contains(ID)) return Standard_False;

            Standard_Integer np_idx = ID - 1;

            Coords(1) = nodes_proxy(np_idx, 0);
            Coords(2) = nodes_proxy(np_idx, 1);
            Coords(3) = nodes_proxy(np_idx, 2);

            NbNodes = 1;
            Type = MeshVS_ET_Node;
            return Standard_True;
        }
        else {
            if (!my_element_ids.Contains(ID)) return Standard_False;

            auto elems_proxy = my_elements.unchecked<2>();
            Standard_Integer elem_idx = ID - 1;

            for (int i = 0; i < 3; ++i) {
                Standard_Integer node_idx = elems_proxy(elem_idx, i);

                Coords(i * 3 + 1) = nodes_proxy(node_idx, 0);
                Coords(i * 3 + 2) = nodes_proxy(node_idx, 1);
                Coords(i * 3 + 3) = nodes_proxy(node_idx, 2);
            }

            NbNodes = 3;
            Type = MeshVS_ET_Face;
            return Standard_True;
        }
    }

    Standard_Boolean GetGeomType(const Standard_Integer ID,
        const Standard_Boolean IsElement,
        MeshVS_EntityType& Type) const override {
        if (IsElement) {
            if (!my_element_ids.Contains(ID)) return Standard_False;
            Type = MeshVS_ET_Face;
            return Standard_True;
        }
        else {
            if (!my_node_ids.Contains(ID)) return Standard_False;
            Type = MeshVS_ET_Node;
            return Standard_True;
        }
    }

    Standard_Boolean GetNodesByElement(const Standard_Integer ID,
        NCollection_Array1<int>& NodeIDs,
        Standard_Integer& NbNodes) const override {
        if (!my_element_ids.Contains(ID)) return Standard_False;

        auto elems_proxy = my_elements.unchecked<2>();
        Standard_Integer elem_idx = ID - 1;

        NodeIDs(1) = elems_proxy(elem_idx, 0) + 1;
        NodeIDs(2) = elems_proxy(elem_idx, 1) + 1;
        NodeIDs(3) = elems_proxy(elem_idx, 2) + 1;

        NbNodes = 3;
        return Standard_True;
    }

    const TColStd_PackedMapOfInteger& GetAllNodes() const override {
        return my_node_ids;
    }

    const TColStd_PackedMapOfInteger& GetAllElements() const override {
        return my_element_ids;
    }

    Standard_Boolean Get3DGeom(
        const Standard_Integer /*ID*/,
        Standard_Integer&,
        Handle(NumpyDS_HArray1OfSequenceOfInteger)& /*Data*/) const override {
        return Standard_False;
    }

    Standard_Address GetAddr(
        const Standard_Integer /*ID*/,
        const Standard_Boolean /*IsElement*/) const override {
        return nullptr;
    }

    Standard_Boolean GetNormal(
        const Standard_Integer ID,
        const Standard_Integer /*Max*/,
        Standard_Real& nx,
        Standard_Real& ny,
        Standard_Real& nz) const override {

        if (!my_element_ids.Contains(ID))
            return Standard_False;

        Standard_Integer idx = ID - 1;

        if (use_computed_normals) {
            nx = my_elem_normals[idx * 3 + 0];
            ny = my_elem_normals[idx * 3 + 1];
            nz = my_elem_normals[idx * 3 + 2];
            return Standard_True;
        }

        return Standard_False;
    }

    Standard_Boolean GetNodeNormal(
        const Standard_Integer ranknode,
        const Standard_Integer ElementId,
        Standard_Real& nx,
        Standard_Real& ny,
        Standard_Real& nz) const override {

        if (!my_element_ids.Contains(ElementId))
            return Standard_False;

        Standard_Integer elem_idx = ElementId - 1;
        Standard_Integer local_idx = ranknode - 1;  // ranknode is 1-based

        if (local_idx < 0 || local_idx >= 3)
            return Standard_False;

        if (use_computed_normals) {
            // Use precomputed crease-angle-aware smooth normals
            Standard_Integer base = (elem_idx * 3 + local_idx) * 3;
            nx = my_smooth_normals[base + 0];
            ny = my_smooth_normals[base + 1];
            nz = my_smooth_normals[base + 2];
            return Standard_True;
        }
        else if (my_normals.has_value()) {
            // Use externally provided per-vertex normals
            auto elems_proxy = my_elements.unchecked<2>();
            Standard_Integer node_idx = elems_proxy(elem_idx, local_idx);

            auto norms_proxy = my_normals->unchecked<2>();
            nx = norms_proxy(node_idx, 0);
            ny = norms_proxy(node_idx, 1);
            nz = norms_proxy(node_idx, 2);
            return Standard_True;
        }

        return Standard_False;
    }

    py::array_t<double> GetNodesArray() {
        return my_nodes;
    }

    py::array_t<int32_t> GetElementsArray() {
        return my_elements;
    }

    std::optional<py::array_t<double>> GetNormalsArray() {
        return my_normals;
    }
};

IMPLEMENT_STANDARD_RTTIEXT(NumpyMeshDataSource, MeshVS_DataSource)


void assign_numpy_datasource_to_mesh(
    uintptr_t occ_mesh_ptr,
    py::array_t<double> nodes,
    py::array_t<int32_t> elements,
    std::optional<py::array_t<double>> normals,
    double crease_angle_deg
) {

    auto* mesh = reinterpret_cast<MeshVS_Mesh*>(occ_mesh_ptr);
    if (!mesh) throw std::runtime_error("Invalid MeshVS_Mesh pointer!");

    Handle(NumpyMeshDataSource) custom_ds = new NumpyMeshDataSource(nodes, elements, normals, crease_angle_deg);

    mesh->SetDataSource(custom_ds);
}


py::array_t<double> read_numpy_datasource_vertices(uintptr_t occ_ds_ptr) {

    auto* ds = reinterpret_cast<NumpyMeshDataSource*>(occ_ds_ptr);
    if (!ds) throw std::runtime_error("Invalid NumpyMeshDataSource pointer!");

    return ds->GetNodesArray();
}


py::array_t<int32_t> read_numpy_datasource_faces(uintptr_t occ_ds_ptr) {

    auto* ds = reinterpret_cast<NumpyMeshDataSource*>(occ_ds_ptr);
    if (!ds) throw std::runtime_error("Invalid NumpyMeshDataSource pointer!");

    return ds->GetElementsArray();
}


std::optional<py::array_t<double>> read_numpy_datasource_normals(uintptr_t occ_ds_ptr) {

    auto* ds = reinterpret_cast<NumpyMeshDataSource*>(occ_ds_ptr);
    if (!ds) throw std::runtime_error("Invalid NumpyMeshDataSource pointer!");

    return ds->GetNormalsArray();
}
