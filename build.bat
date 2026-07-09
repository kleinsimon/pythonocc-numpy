mkdir build
cd build
cmake ..
cmake --build .
copy /y Debug\*.pyd ..\src\occ_numpy_bridge
cd ..