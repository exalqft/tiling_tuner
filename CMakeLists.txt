cmake_minimum_required(VERSION 3.21.0)
project(tiling-tuner VERSION 1.0.0 LANGUAGES CXX)

# Honor already set CMAKE_CXX_STANDARD.
# Otherwise default to C++17
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

#
# Kokkos
#
# General Kokkos settings
set(Kokkos_ENABLE_COMPILER_WARNINGS "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_COMPILER_WARNINGS" FORCE)
set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION "ON" CACHE BOOL "CMakeLists: Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/kokkos/kokkos ${CMAKE_CURRENT_BINARY_DIR}/kokkos)

add_executable(tune_su3xsu3 tune_su3xsu3.cpp)
target_link_libraries(tune_su3xsu3 PRIVATE Kokkos::kokkos)

# add_executable(tune_plaq tune_plaq.cpp)
# target_link_libraries(tune_plaq PRIVATE Kokkos::kokkos)