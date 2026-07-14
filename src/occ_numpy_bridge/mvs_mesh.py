import numpy as np
from OCC.Core.MeshVS import MeshVS_Mesh, MeshVS_DataSource

from . import occ_bridge
from .base import OCC_Wrapper_Base


class MeshVS_Mesh_Helper(OCC_Wrapper_Base):
    """
    Provides helper methods for working with `MeshVS_Mesh` objects.

    This class offers functionality to assign numpy data sources to `MeshVS_Mesh`
    objects, read numpy-compatible data back, and create mesh objects using
    numpy arrays as data sources. It utilizes OCC functions to facilitate
    this process and bridges the connection between numpy arrays and OCC mesh
    data structures.

    """
    _OCC_CLS = MeshVS_Mesh

    @classmethod
    def assign_numpy_data_source(cls, mesh: MeshVS_Mesh, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray = None) -> MeshVS_Mesh:
        """
        Assigns a data source based on NumPy arrays to the provided MeshVS_Mesh instance.

        This method takes NumPy arrays representing vertices, faces, and optionally normals,
        and assigns them as data sources to the given mesh. All input arrays are converted
        to their appropriate contiguous and dtype formats for compatibility with the underlying
        C++ library.

        :param mesh: The MeshVS_Mesh instance to which the data source should be assigned.
        :type mesh: MeshVS_Mesh
        :param vertices: A NumPy array of vertex coordinates.
        :type vertices: np.ndarray
        :param faces: A NumPy array defining triangular or polygonal faces.
        :type faces: np.ndarray
        :param normals: An optional NumPy array of normals for the vertices. Defaults to None.
        :type normals: np.ndarray, optional
        :return: The updated MeshVS_Mesh instance with the assigned data source.
        :rtype: MeshVS_Mesh
        """
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        if normals is not None:
            normals = np.ascontiguousarray(normals, dtype=np.float64)

        mesh_ptr_int = cls._get_cpp_pointer(mesh)

        occ_bridge.meshvs.assign_numpy_datasource_to_mesh(mesh_ptr_int, vertices, faces, normals)

        return mesh

    @classmethod
    def read_numpy_data_source_vertices(cls, ds: MeshVS_DataSource) -> np.ndarray:
        """
        Reads the vertices from a given ``MeshVS_DataSource`` as a numpy array.
        The Datasource must have been created with ``assign_numpy_data_source``.

        :param ds: The data source from which to read the vertices.
                   This must be an instance of ``MeshVS_DataSource``.
        :return: A numpy array containing the vertices extracted from
                 the data source.
        :rtype: np.ndarray
        """
        ds_ptr_int = cls._get_cpp_pointer(ds, MeshVS_DataSource)

        return occ_bridge.meshvs.read_numpy_datasource_vertices(ds_ptr_int)

    @classmethod
    def read_numpy_data_source_faces(cls, ds: MeshVS_DataSource) -> np.ndarray:
        """
        Reads face data from a given MeshVS data source and returns it as a NumPy array.
        The Datasource must have been created with ``assign_numpy_data_source``.

        :param ds: The MeshVS_DataSource object containing the data to extract.
        :type ds: MeshVS_DataSource
        :return: A NumPy array containing face-related data extracted from the input
            data source.
        :rtype: np.ndarray
        """
        ds_ptr_int = cls._get_cpp_pointer(ds, MeshVS_DataSource)

        return occ_bridge.meshvs.read_numpy_datasource_faces(ds_ptr_int)

    @classmethod
    def read_numpy_data_source_normals(cls, ds: MeshVS_DataSource) -> np.ndarray:
        """
        Reads normals data from a MeshVS_DataSource and returns it as a numpy array.

        The Datasource must have been created with ``assign_numpy_data_source``.
        If no normals are available, returns None.

        :param ds: MeshVS_DataSource object representing the source of mesh data.
        :return: Numpy array containing normalized vector data extracted from the
                 data source.
        :rtype: np.ndarray or None
        """
        ds_ptr_int = cls._get_cpp_pointer(ds, MeshVS_DataSource)

        return occ_bridge.meshvs.read_numpy_datasource_normals(ds_ptr_int)

    @classmethod
    def create_mesh_with_numpy_source(cls, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray = None) -> MeshVS_Mesh:
        """
        Create a mesh object with data sourced from numpy arrays.

        This method constructs a mesh using vertices, faces, and optionally normals
        defined in numpy arrays.

        :param vertices: A numpy array containing vertex coordinate data.
        :param faces: A numpy array representing the face definitions, where each
            face is described by indices pointing to vertices.
        :param normals: Optional; A numpy array specifying normals for the vertices
            or faces. If not provided, the mesh will be constructed without normals.
        :return: An instance of MeshVS_Mesh containing the mesh created from the
            provided data.
        """
        return cls.assign_numpy_data_source(MeshVS_Mesh(), vertices, faces, normals)