cmake_minimum_required(VERSION 3.16)
project(soa_cpp_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Eigen is header-only. Do NOT use FetchContent_MakeAvailable(eigen), because
# that configures Eigen's full test project and can inject unwanted global flags
# on MSYS2/MinGW. We only fetch/include the headers.
set(LOCAL_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-3.4.0")

if(EXISTS "${LOCAL_EIGEN_DIR}/Eigen/Dense")
    message(STATUS "Using local Eigen headers: ${LOCAL_EIGEN_DIR}")
    set(EIGEN_INCLUDE_DIR "${LOCAL_EIGEN_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        eigen_headers
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(eigen_headers)
    if(NOT eigen_headers_POPULATED)
        FetchContent_Populate(eigen_headers)
    endif()
    message(STATUS "Using fetched Eigen headers: ${eigen_headers_SOURCE_DIR}")
    set(EIGEN_INCLUDE_DIR "${eigen_headers_SOURCE_DIR}")
endif()

add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN_INCLUDE_DIR}")

add_library(soa_cpp INTERFACE)
target_include_directories(soa_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(soa_cpp INTERFACE
    EIGEN_DONT_VECTORIZE
    EIGEN_MAX_ALIGN_BYTES=0
    EIGEN_NO_DEBUG
)
target_link_libraries(soa_cpp INTERFACE Eigen3::Eigen)

add_executable(example_simulation examples/example_simulation.cpp)
target_link_libraries(example_simulation PRIVATE soa_cpp)
