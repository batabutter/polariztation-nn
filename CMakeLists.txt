cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(scatter-best-fit)

set(Torch_DIR "/home/batabutter/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(scatter-best-fit src/main.cpp)
target_link_libraries(scatter-best-fit "${TORCH_LIBRARIES}")
set_property(TARGET scatter-best-fit PROPERTY CXX_STANDARD 17)
