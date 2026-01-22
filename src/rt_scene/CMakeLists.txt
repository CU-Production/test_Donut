include(../../3rd_party/donut/compileshaders.cmake)
file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")

set(project rt_scene)
set(folder "Examples/Ray Traced Scene")

donut_compile_shaders(
    TARGET ${project}_shaders
    PROJECT_NAME "Ray Traced Scene"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXIL ${CMAKE_CURRENT_BINARY_DIR}/shaders/${project}/dxil
    SPIRV_DXC ${CMAKE_CURRENT_BINARY_DIR}/shaders/${project}/spirv
)

add_executable(${project} WIN32 ${sources})
target_link_libraries(${project} donut_app donut_engine donut_render pugixml tinyobj hmm)
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

# Ensure RuntimeLibrary matches DLSS static library (/MT for Release, /MTd for Debug)
# This is critical for DLSS to work - NGX DLL must match the runtime library
if(MSVC)
    set_target_properties(${project} PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
endif()

# Link DLSS if available
if(TARGET DLSS)
    target_link_libraries(${project} DLSS)
endif()

# Include paths for stb_image and tinyexr
target_include_directories(${project} PRIVATE 
    ${CMAKE_SOURCE_DIR}/3rd_party/Donut/thirdparty/stb
    ${CMAKE_SOURCE_DIR}/3rd_party/Donut/thirdparty/tinyexr
)
