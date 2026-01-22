cmake_minimum_required(VERSION 4.0)
project(test_Donut)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use static runtime library to match DLSS SDK
# /MT for Release, /MTd for Debug
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(3rd_party)

add_subdirectory(src)
