cmake_minimum_required(VERSION 3.18)
project(OccNumpyBridge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# OpenCASCADE
find_package(OpenCASCADE REQUIRED)

if(WIN32)
    link_directories("$ENV{CONDA_PREFIX}/Library/lib")
    if(OpenCASCADE_LIBRARY_DIR)
        link_directories(${OpenCASCADE_LIBRARY_DIR})
    endif()
endif()

pybind11_add_module(occ_bridge src/csrc/main.cpp)

target_include_directories(occ_bridge PRIVATE
    ${OpenCASCADE_INCLUDE_DIR}
    "$ENV{CONDA_PREFIX}/Library/include/opencascade"
)

target_link_libraries(occ_bridge PRIVATE
    TKMath
    TKernel
    TKService
    TKV3d
    TKOpenGl
    TKMeshVS
)

install(TARGETS occ_bridge DESTINATION occ_numpy_bridge COMPONENT python_modules)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_command(TARGET occ_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:occ_bridge> 
            ${Python3_EXECUTABLE} -m pybind11_stubgen occ_bridge -o $<TARGET_FILE_DIR:occ_bridge>
    WORKING_DIRECTORY $<TARGET_FILE_DIR:occ_bridge>
    COMMENT "Generating Python Stubs (.pyi)..."
)