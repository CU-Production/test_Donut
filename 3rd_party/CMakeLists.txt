# donut
add_subdirectory(Donut)

# pugixml
add_subdirectory(pugixml)

# hmm
add_library(hmm INTERFACE)
target_include_directories(hmm INTERFACE HandmadeMath)

# tinyobj
add_library(tinyobj INTERFACE)
target_include_directories(tinyobj INTERFACE tinyobjloader-c)
