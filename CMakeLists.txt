cmake_minimum_required(VERSION 3.18)
project(OccNumpyBridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Python & pybind11 finden
# Tipp: Wenn du Conda nutzt, aktiviere deine Environment VOR dem cmake Aufruf!
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Versuche pybind11 über cmake zu finden (oft via conda install pybind11 installiert)
find_package(pybind11 CONFIG REQUIRED)

# 2. OpenCASCADE finden
# Falls cmake OpenCASCADE nicht automatisch findet, setze beim Build: 
# -DOpenCASCADE_DIR=/pfad/zu/conda/env/lib/cmake/opencascade
find_package(OpenCASCADE REQUIRED)

# 3. Das Python-Modul erstellen
pybind11_add_module(occ_bridge src/point_bridge.cpp)

# 4. OpenCASCADE Header und Bibliotheken verlinken
target_include_directories(occ_bridge PRIVATE ${OpenCASCADE_INCLUDE_DIR})

target_link_directories(occ_bridge PRIVATE 
    ${OpenCASCADE_LIBRARY_DIR}
    "$ENV{CONDA_PREFIX}/Library/lib" # Fallback für Windows Conda
)

target_link_libraries(occ_bridge PRIVATE 
    TKMath 
    TKKernel 
    TKService 
    TKV3d 
    TKOpenGl
)