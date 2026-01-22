# Enable DLSS before adding Donut
# Note: DLSS static library uses /MT runtime which conflicts with /MD in Debug mode
# Use Release mode to avoid linker errors
set(DONUT_WITH_DLSS OFF CACHE BOOL "Enable DLSS" FORCE)

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
