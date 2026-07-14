#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MeshVS_DataSource.hxx>
#include <MeshVS_EntityType.hxx>
#include <MeshVS_Mesh.hxx>
#include <TColStd_PackedMapOfInteger.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <Standard_Type.hxx>
#include <optional>

namespace py = pybind11;

class NumpyMeshDataSource : public MeshVS_DataSource {
    DEFINE_STANDARD_RTTIEXT(NumpyMeshDataSource, MeshVS_DataSource)

private:
    py::array_t<double> my_nodes;
    py::array_t<int32_t> my_elements;
    std::optional<py::array_t<double>> my_normals;

    std::vector<double> my_computed_normals;
    bool use_computed_normals = false;

    TColStd_PackedMapOfInteger my_node_ids;
    TColStd_PackedMapOfInteger my_element_ids;

    void compute_normals() {
        auto nodes_proxy = my_nodes.unchecked<2>();
        auto elems_proxy = my_elements.unchecked<2>();

        Standard_Integer nb_nodes = my_nodes.shape(0);
        Standard_Integer nb_elems = my_elements.shape(0);

        my_computed_normals.assign(nb_nodes * 3, 0.0);

        for (Standard_Integer i = 0; i < nb_elems; ++i) {
            Standard_Integer idx0 = elems_proxy(i, 0);
            Standard_Integer idx1 = elems_proxy(i, 1);
            Standard_Integer idx2 = elems_proxy(i, 2);

            // fetch coords
            double p0[3] = { nodes_proxy(idx0, 0), nodes_proxy(idx0, 1), nodes_proxy(idx0, 2) };
            double p1[3] = { nodes_proxy(idx1, 0), nodes_proxy(idx1, 1), nodes_proxy(idx1, 2) };
            double p2[3] = { nodes_proxy(idx2, 0), nodes_proxy(idx2, 1), nodes_proxy(idx2, 2) };

            // vecs of edges (U = P1 - P0, V = P2 - P0)
            double u[3] = { p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2] };
            double v[3] = { p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2] };

            // cross product
            double nx = u[1] * v[2] - u[2] * v[1];
            double ny = u[2] * v[0] - u[0] * v[2];
            double nz = u[0] * v[1] - u[1] * v[0];

            // sum over vertices
            for (int j = 0; j < 3; ++j) {
                Standard_Integer v_idx = elems_proxy(i, j);
                my_computed_normals[v_idx * 3 + 0] += nx;
                my_computed_normals[v_idx * 3 + 1] += ny;
                my_computed_normals[v_idx * 3 + 2] += nz;
            }
        }

        // Normalize
        for (Standard_Integer i = 0; i < nb_nodes; ++i) {
            double nx = my_computed_normals[i * 3 + 0];
            double ny = my_computed_normals[i * 3 + 1];
            double nz = my_computed_normals[i * 3 + 2];

            double len = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-10) {
                my_computed_normals[i * 3 + 0] /= len;
                my_computed_normals[i * 3 + 1] /= len;
                my_computed_normals[i * 3 + 2] /= len;
            }
        }
    }

public:
    NumpyMeshDataSource(py::array_t<double> nodes, py::array_t<int32_t> elements, std::optional<py::array_t<double>> normals)
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
            compute_normals();
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
        TColStd_Array1OfReal& Coords,
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
        TColStd_Array1OfInteger& NodeIDs,
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

    Standard_Boolean Get3DGeom(const Standard_Integer /*ID*/,
        Standard_Integer&,
        Handle(MeshVS_HArray1OfSequenceOfInteger)& /*Data*/) const override {
        return Standard_False;
    }

    Standard_Address GetAddr(const Standard_Integer /*ID*/,
        const Standard_Boolean /*IsElement*/) const override {
        return nullptr;
    }

    Standard_Boolean GetNormal(const Standard_Integer ID,
        const Standard_Integer Max,
        Standard_Real& nx,
        Standard_Real& ny,
        Standard_Real& nz) const override {

        if (!my_node_ids.Contains(ID)) return Standard_False;

        Standard_Integer np_idx = ID - 1;

        if (use_computed_normals) {
            nx = my_computed_normals[np_idx * 3 + 0];
            ny = my_computed_normals[np_idx * 3 + 1];
            nz = my_computed_normals[np_idx * 3 + 2];
            return Standard_True;
        }
        else if (my_normals.has_value()) {
            auto norms_proxy = my_normals->unchecked<2>();
            nx = norms_proxy(np_idx, 0);
            ny = norms_proxy(np_idx, 1);
            nz = norms_proxy(np_idx, 2);
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


void assign_numpy_datasource_to_mesh(uintptr_t occ_mesh_ptr,
    py::array_t<double> nodes,
    py::array_t<int32_t> elements,
    std::optional<py::array_t<double>> normals) {

    auto* mesh = reinterpret_cast<MeshVS_Mesh*>(occ_mesh_ptr);
    if (!mesh) throw std::runtime_error("Invalid MeshVS_Mesh pointer!");

    Handle(NumpyMeshDataSource) custom_ds = new NumpyMeshDataSource(nodes, elements, normals);

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
