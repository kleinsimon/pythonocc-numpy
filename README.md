# pythonocc-numpy

This project aims to provide some helpers for [pythonocc-core](https://github.com/tpaviot/pythonocc-core) to efficiently
handle large numpy arrays. By now it is rather a proof-of-concept.

It allows the fast transfer of vertices and colors for selected OpenCascade class instances.
It links via pybind11 directly against OCC, so the official pythonocc-core binaries can be used.

# Install
There are some wheels provided under "Releases". Install them via pip.

# Building

since pythonocc-core is only available via conda-forge, you need conda / miniforge to build the wheel.

```bash
# Create env:
conda env create -f environment.yml

# Activate env:
conda activate occ-bridge-dev

# build wheel 
python -m build --wheel

# install wheel
pip install dist/occ_numpy_bridge*.whl
```

# Usage

## Supported Classes

Read and Write access using numpy arrays

- Graphic3d_ArrayOfPoints
  - vertex coordinates
  - vertex colors
  

- Graphic3d_ArrayOfTriangles
  - vertex coordinates
  - vertex colors
  - vertex normals
  - face indices


- Poly_Triangulation
  - vertex coords
  - face indices

```python
import numpy as np
from occ_numpy_bridge import Graphic3d_ArrayOfPoints_Helper
from OCC.Core.Graphic3d import Graphic3d_ArrayOfPoints

# create with factory
num_points = 2_000
np_coords = np.random.rand(num_points, 3).astype(np.float64)
np_colors = np.random.rand(num_points, 3).astype(np.float64)

occ_array = Graphic3d_ArrayOfPoints_Helper.create_from_numpy(np_coords, np_colors)

# populate existing
occ_array2 = Graphic3d_ArrayOfPoints(num_points, 2)
Graphic3d_ArrayOfPoints_Helper.set_coordinates(occ_array2, np_coords)
Graphic3d_ArrayOfPoints_Helper.set_colors(occ_array2, np_colors)

# read values
coords = Graphic3d_ArrayOfPoints_Helper.get_coordinates(occ_array2)
colors = Graphic3d_ArrayOfPoints_Helper.get_colors(occ_array2)
```