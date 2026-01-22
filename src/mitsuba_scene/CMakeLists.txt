include(../../3rd_party/donut/compileshaders.cmake)
file(GLOB shaders "*.hlsl")
file(GLOB sources "*.cpp" "*.h")

set(project mitsuba_scene)
set(folder "Examples/Mitsuba Scene Rasterizer")

donut_compile_shaders(
    TARGET ${project}_shaders
    PROJECT_NAME "Mitsuba Scene Rasterizer"
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    SOURCES ${shaders}
    FOLDER ${folder}
    DXIL ${CMAKE_CURRENT_BINARY_DIR}/shaders/${project}/dxil
    SPIRV_DXC ${CMAKE_CURRENT_BINARY_DIR}/shaders/${project}/spirv
)

add_executable(${project} WIN32 ${sources})
target_link_libraries(${project} donut_app donut_engine pugixml tinyobj hmm)
add_dependencies(${project} ${project}_shaders)
set_target_properties(${project} PROPERTIES FOLDER ${folder})
