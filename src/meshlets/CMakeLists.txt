include(../../3rd_party/donut/compileshaders.cmake)
file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")

set(project meshlets)
set(folder "Examples/Meshlets")

donut_compile_shaders(
    TARGET ${project}_shaders
    PROJECT_NAME "Meshlets"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    DXIL ${CMAKE_CURRENT_BINARY_DIR}/shaders/${project}/dxil
    SPIRV_DXC ${CMAKE_CURRENT_BINARY_DIR}/shaders/${project}/spirv
    SHADERMAKE_OPTIONS_SPIRV "--spirvExt SPV_EXT_mesh_shader")

add_executable(${project} WIN32 ${sources})
target_link_libraries(${project} donut_app donut_engine)
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP")
endif()
