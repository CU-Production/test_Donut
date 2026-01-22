#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>

#include <pugixml.hpp>

#define HANDMADE_MATH_USE_RADIANS
#include "HandmadeMath.h"

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include <tinyobj_loader_c.h>

#include <filesystem>
#include <unordered_map>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace donut;

static const char* g_WindowTitle = "Mitsuba Scene Rasterizer";

// ============================================================================
// GPU Structures (must match HLSL) - using plain floats for GPU compatibility
// ============================================================================
struct GPUVertex
{
    float position[3];
    float normal[3];
    float texcoord[2];
};

struct PerObjectConstants
{
    float worldViewProj[16];  // column-major 4x4 matrix
    float world[16];          // column-major 4x4 matrix
    float baseColor[3];
    float roughness;
    float emission[3];
    uint32_t isEmitter;
};

struct LightConstants
{
    float lightDir[3];
    float pad0;
    float lightColor[3];
    float pad1;
    float ambientColor[3];
    float pad2;
    float cameraPos[3];
    float pad3;
};

// ============================================================================
// Mitsuba Scene Parser - using HandmadeMath (column-vector convention)
// ============================================================================
class MitsubaSceneParser
{
public:
    struct Camera
    {
        HMM_Mat4 transform = HMM_M4D(1.0f);  // Identity matrix
        float fov = 45.0f;
        int width = 1280;
        int height = 720;
    };

    struct Material
    {
        std::string id;
        HMM_Vec3 baseColor = HMM_V3(0.5f, 0.5f, 0.5f);
        float roughness = 0.5f;
    };

    struct Shape
    {
        std::string type;
        std::string filename;
        std::string materialRef;
        HMM_Mat4 transform = HMM_M4D(1.0f);  // Identity matrix
        bool isEmitter = false;
        HMM_Vec3 emission = HMM_V3(0.0f, 0.0f, 0.0f);
        
        Material inlineMaterial;
        bool hasInlineMaterial = false;
    };

    Camera camera;
    std::unordered_map<std::string, Material> materials;
    std::vector<Shape> shapes;
    std::filesystem::path sceneDirectory;

    bool Parse(const std::filesystem::path& xmlPath)
    {
        sceneDirectory = xmlPath.parent_path();

        pugi::xml_document doc;
        pugi::xml_parse_result result = doc.load_file(xmlPath.c_str());

        if (!result)
        {
            log::error("Failed to parse XML file: %s", result.description());
            return false;
        }

        pugi::xml_node sceneNode = doc.child("scene");
        if (!sceneNode)
        {
            log::error("No <scene> node found in XML");
            return false;
        }

        for (pugi::xml_node node : sceneNode.children())
        {
            std::string nodeName = node.name();

            if (nodeName == "sensor")
            {
                ParseSensor(node);
            }
            else if (nodeName == "bsdf")
            {
                Material mat = ParseBSDF(node);
                if (!mat.id.empty())
                {
                    materials[mat.id] = mat;
                }
            }
            else if (nodeName == "shape")
            {
                ParseShape(node);
            }
        }

        log::info("Parsed %zu materials and %zu shapes", materials.size(), shapes.size());
        return true;
    }

private:
    // Parse Mitsuba matrix for HMM row-vector convention
    // Mitsuba uses column-vector: p' = M * p, where p'.x = m00*p.x + m01*p.y + m02*p.z + m03
    // HMM uses row-vector: p' = p * M, where p'.x = p.x*col0.x + p.y*col1.x + p.z*col2.x + p.w*col3.x
    // For equivalence: col0.x=m00, col1.x=m01, col2.x=m02, col3.x=m03
    //                  col0.y=m10, col1.y=m11, col2.y=m12, col3.y=m13, etc.
    // Mitsuba XML text order: m00 m01 m02 m03  m10 m11 m12 m13  m20 m21 m22 m23  m30 m31 m32 m33
    HMM_Mat4 ParseMatrix(const std::string& matrixStr)
    {
        std::istringstream iss(matrixStr);
        float values[16];
        for (int i = 0; i < 16; i++)
        {
            iss >> values[i];
        }
        
        // Map Mitsuba matrix to HMM columns:
        // Column j gets all elements m[i][j] for i=0..3
        HMM_Mat4 result;
        result.Columns[0] = HMM_V4(values[0], values[4], values[8],  values[12]); // m00,m10,m20,m30
        result.Columns[1] = HMM_V4(values[1], values[5], values[9],  values[13]); // m01,m11,m21,m31
        result.Columns[2] = HMM_V4(values[2], values[6], values[10], values[14]); // m02,m12,m22,m32
        result.Columns[3] = HMM_V4(values[3], values[7], values[11], values[15]); // m03,m13,m23,m33
        return result;
    }

    HMM_Vec3 ParseRGB(const std::string& rgbStr)
    {
        HMM_Vec3 color = HMM_V3(0.0f, 0.0f, 0.0f);
        std::string cleaned = rgbStr;
        for (char& c : cleaned)
        {
            if (c == ',') c = ' ';
        }
        std::istringstream iss(cleaned);
        iss >> color.X >> color.Y >> color.Z;
        return color;
    }

    void ParseSensor(pugi::xml_node sensorNode)
    {
        for (pugi::xml_node child : sensorNode.children("float"))
        {
            std::string name = child.attribute("name").value();
            if (name == "fov")
            {
                camera.fov = child.attribute("value").as_float(45.0f);
            }
        }

        pugi::xml_node transformNode = sensorNode.child("transform");
        if (transformNode)
        {
            pugi::xml_node matrixNode = transformNode.child("matrix");
            if (matrixNode)
            {
                camera.transform = ParseMatrix(matrixNode.attribute("value").value());
            }
        }

        pugi::xml_node filmNode = sensorNode.child("film");
        if (filmNode)
        {
            for (pugi::xml_node child : filmNode.children("integer"))
            {
                std::string name = child.attribute("name").value();
                if (name == "width")
                {
                    camera.width = child.attribute("value").as_int(1280);
                }
                else if (name == "height")
                {
                    camera.height = child.attribute("value").as_int(720);
                }
            }
        }
    }

    Material ParseBSDF(pugi::xml_node bsdfNode, bool nested = false)
    {
        Material mat;
        
        if (!nested)
        {
            mat.id = bsdfNode.attribute("id").value();
        }

        std::string type = bsdfNode.attribute("type").value();

        if (type == "twosided")
        {
            pugi::xml_node innerBsdf = bsdfNode.child("bsdf");
            if (innerBsdf)
            {
                Material innerMat = ParseBSDF(innerBsdf, true);
                innerMat.id = mat.id;
                return innerMat;
            }
        }

        for (pugi::xml_node child : bsdfNode.children())
        {
            std::string childName = child.name();
            std::string propName = child.attribute("name").value();

            if (childName == "rgb")
            {
                HMM_Vec3 color = ParseRGB(child.attribute("value").value());
                if (propName == "reflectance" || propName == "diffuse_reflectance" || propName == "specular_reflectance")
                {
                    mat.baseColor = color;
                }
            }
            else if (childName == "float")
            {
                float value = child.attribute("value").as_float();
                if (propName == "alpha")
                {
                    mat.roughness = value;
                }
            }
        }

        return mat;
    }

    void ParseShape(pugi::xml_node shapeNode)
    {
        Shape shape;
        shape.type = shapeNode.attribute("type").value();

        pugi::xml_node transformNode = shapeNode.child("transform");
        if (transformNode)
        {
            pugi::xml_node matrixNode = transformNode.child("matrix");
            if (matrixNode)
            {
                shape.transform = ParseMatrix(matrixNode.attribute("value").value());
            }
        }

        for (pugi::xml_node child : shapeNode.children("string"))
        {
            std::string name = child.attribute("name").value();
            if (name == "filename")
            {
                shape.filename = child.attribute("value").value();
            }
        }

        pugi::xml_node refNode = shapeNode.child("ref");
        if (refNode)
        {
            shape.materialRef = refNode.attribute("id").value();
        }

        pugi::xml_node inlineBsdf = shapeNode.child("bsdf");
        if (inlineBsdf)
        {
            shape.inlineMaterial = ParseBSDF(inlineBsdf, true);
            shape.hasInlineMaterial = true;
        }

        pugi::xml_node emitterNode = shapeNode.child("emitter");
        if (emitterNode)
        {
            shape.isEmitter = true;
            for (pugi::xml_node child : emitterNode.children("rgb"))
            {
                std::string name = child.attribute("name").value();
                if (name == "radiance")
                {
                    shape.emission = ParseRGB(child.attribute("value").value());
                }
            }
        }

        shapes.push_back(shape);
    }
};

// ============================================================================
// OBJ Loader Callback
// ============================================================================
static void FileReaderCallback(void* ctx, const char* filename, int is_mtl, 
                               const char* obj_filename, char** buf, size_t* len)
{
    std::filesystem::path* basePath = static_cast<std::filesystem::path*>(ctx);
    std::filesystem::path fullPath = *basePath / filename;

    FILE* file = nullptr;
#ifdef _WIN32
    fopen_s(&file, fullPath.string().c_str(), "rb");
#else
    file = fopen(fullPath.string().c_str(), "rb");
#endif

    if (!file)
    {
        *buf = nullptr;
        *len = 0;
        return;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    *buf = static_cast<char*>(malloc(fileSize + 1));
    *len = fread(*buf, 1, fileSize, file);
    (*buf)[*len] = '\0';

    fclose(file);
}

// ============================================================================
// Mesh Data for Rendering
// ============================================================================
struct RenderMesh
{
    nvrhi::BufferHandle vertexBuffer;
    nvrhi::BufferHandle indexBuffer;
    uint32_t indexCount;
    HMM_Mat4 worldTransform;  // Model matrix (local to world)
    HMM_Vec3 baseColor;
    float roughness;
    HMM_Vec3 emission;
    bool isEmitter;
};

// ============================================================================
// Rasterized Scene Application
// ============================================================================
class MitsubaSceneRasterizer : public app::IRenderPass
{
private:
    nvrhi::ShaderHandle m_VertexShader;
    nvrhi::ShaderHandle m_PixelShader;
    nvrhi::GraphicsPipelineHandle m_Pipeline;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::InputLayoutHandle m_InputLayout;
    nvrhi::BufferHandle m_PerObjectBuffer;
    nvrhi::BufferHandle m_LightBuffer;

    // Depth buffer and framebuffer
    nvrhi::TextureHandle m_DepthTexture;
    nvrhi::BindingSetHandle m_BindingSet;

    std::vector<RenderMesh> m_Meshes;
    
    MitsubaSceneParser m_SceneParser;
    std::filesystem::path m_ScenePath;
    
    // Camera state (using HandmadeMath)
    HMM_Vec3 m_CameraPosition;
    HMM_Vec3 m_CameraTarget;
    HMM_Vec3 m_CameraUp;
    float m_CameraYaw = 0.0f;
    float m_CameraPitch = 0.0f;
    float m_CameraSpeed = 10.0f;
    
    // Mouse state
    bool m_MouseDown = false;
    float m_LastMouseX = 0.0f;
    float m_LastMouseY = 0.0f;
    
    // Keyboard state
    bool m_KeyW = false, m_KeyS = false, m_KeyA = false, m_KeyD = false;
    bool m_KeyQ = false, m_KeyE = false;

public:
    MitsubaSceneRasterizer(app::DeviceManager* deviceManager, const std::filesystem::path& scenePath)
        : IRenderPass(deviceManager)
        , m_ScenePath(scenePath)
    {
    }

    bool Init()
    {
        // Parse scene
        if (!m_SceneParser.Parse(m_ScenePath))
        {
            log::error("Failed to parse scene file: %s", m_ScenePath.string().c_str());
            return false;
        }

        // Load shaders
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/mitsuba_scene" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::ShaderFactory shaderFactory(GetDevice(), nativeFS, appShaderPath);

        m_VertexShader = shaderFactory.CreateShader("shaders.hlsl", "main_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_PixelShader = shaderFactory.CreateShader("shaders.hlsl", "main_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_VertexShader || !m_PixelShader)
        {
            log::error("Failed to create shaders");
            return false;
        }

        // Create input layout
        nvrhi::VertexAttributeDesc attributes[] = {
            nvrhi::VertexAttributeDesc()
                .setName("POSITION")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(offsetof(GPUVertex, position))
                .setElementStride(sizeof(GPUVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("NORMAL")
                .setFormat(nvrhi::Format::RGB32_FLOAT)
                .setOffset(offsetof(GPUVertex, normal))
                .setElementStride(sizeof(GPUVertex)),
            nvrhi::VertexAttributeDesc()
                .setName("TEXCOORD")
                .setFormat(nvrhi::Format::RG32_FLOAT)
                .setOffset(offsetof(GPUVertex, texcoord))
                .setElementStride(sizeof(GPUVertex))
        };
        m_InputLayout = GetDevice()->createInputLayout(attributes, 3, m_VertexShader);

        // Create binding layout
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::ConstantBuffer(0),  // PerObject
            nvrhi::BindingLayoutItem::ConstantBuffer(1)   // Light
        };
        m_BindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);

        // Create constant buffers
        nvrhi::BufferDesc cbDesc;
        cbDesc.byteSize = sizeof(PerObjectConstants);
        cbDesc.isConstantBuffer = true;
        cbDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        cbDesc.keepInitialState = true;
        cbDesc.debugName = "PerObjectBuffer";
        m_PerObjectBuffer = GetDevice()->createBuffer(cbDesc);

        cbDesc.byteSize = sizeof(LightConstants);
        cbDesc.debugName = "LightBuffer";
        m_LightBuffer = GetDevice()->createBuffer(cbDesc);

        m_CommandList = GetDevice()->createCommandList();

        // Load meshes from scene
        LoadSceneMeshes();

        // Initialize camera from scene
        InitializeCamera();

        return true;
    }

    void LoadSceneMeshes()
    {
        m_CommandList->open();

        for (auto& shape : m_SceneParser.shapes)
        {
            if (shape.type == "obj")
            {
                LoadOBJMesh(shape);
            }
            else if (shape.type == "rectangle")
            {
                CreateRectangleMesh(shape);
            }
        }

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        log::info("Loaded %zu meshes", m_Meshes.size());
    }

    void LoadOBJMesh(MitsubaSceneParser::Shape& shape)
    {
        std::filesystem::path objPath = m_SceneParser.sceneDirectory / shape.filename;

        tinyobj_attrib_t attrib;
        tinyobj_shape_t* shapes = nullptr;
        size_t numShapes = 0;
        tinyobj_material_t* materials = nullptr;
        size_t numMaterials = 0;

        std::filesystem::path basePath = objPath.parent_path();

        int ret = tinyobj_parse_obj(&attrib, &shapes, &numShapes, &materials, &numMaterials,
            objPath.string().c_str(), FileReaderCallback, &basePath, TINYOBJ_FLAG_TRIANGULATE);

        if (ret != TINYOBJ_SUCCESS)
        {
            log::warning("Failed to load OBJ: %s", objPath.string().c_str());
            return;
        }

        std::vector<GPUVertex> vertices;
        std::vector<uint32_t> indices;
        std::unordered_map<uint64_t, uint32_t> vertexMap;

        for (unsigned int f = 0; f < attrib.num_faces; f++)
        {
            tinyobj_vertex_index_t idx = attrib.faces[f];
            uint64_t key = (uint64_t(idx.v_idx) << 40) | (uint64_t(idx.vn_idx) << 20) | uint64_t(idx.vt_idx);

            auto it = vertexMap.find(key);
            if (it != vertexMap.end())
            {
                indices.push_back(it->second);
            }
            else
            {
                GPUVertex vertex;

                // Keep vertices in local/model space (don't pre-transform)
                vertex.position[0] = attrib.vertices[3 * idx.v_idx + 0];
                vertex.position[1] = attrib.vertices[3 * idx.v_idx + 1];
                vertex.position[2] = attrib.vertices[3 * idx.v_idx + 2];

                if (idx.vn_idx >= 0 && attrib.normals)
                {
                    HMM_Vec3 n = HMM_NormV3(HMM_V3(
                        attrib.normals[3 * idx.vn_idx + 0],
                        attrib.normals[3 * idx.vn_idx + 1],
                        attrib.normals[3 * idx.vn_idx + 2]
                    ));
                    vertex.normal[0] = n.X;
                    vertex.normal[1] = n.Y;
                    vertex.normal[2] = n.Z;
                }
                else
                {
                    vertex.normal[0] = 0.0f;
                    vertex.normal[1] = 1.0f;
                    vertex.normal[2] = 0.0f;
                }

                if (idx.vt_idx >= 0 && attrib.texcoords)
                {
                    vertex.texcoord[0] = attrib.texcoords[2 * idx.vt_idx + 0];
                    vertex.texcoord[1] = attrib.texcoords[2 * idx.vt_idx + 1];
                }
                else
                {
                    vertex.texcoord[0] = 0.0f;
                    vertex.texcoord[1] = 0.0f;
                }

                uint32_t newIndex = static_cast<uint32_t>(vertices.size());
                vertexMap[key] = newIndex;
                vertices.push_back(vertex);
                indices.push_back(newIndex);
            }
        }

        if (vertices.empty() || indices.empty())
        {
            tinyobj_attrib_free(&attrib);
            tinyobj_shapes_free(shapes, numShapes);
            tinyobj_materials_free(materials, numMaterials);
            return;
        }

        // Create GPU buffers
        RenderMesh mesh;

        nvrhi::BufferDesc vbDesc;
        vbDesc.byteSize = sizeof(GPUVertex) * vertices.size();
        vbDesc.isVertexBuffer = true;
        vbDesc.initialState = nvrhi::ResourceStates::VertexBuffer;
        vbDesc.keepInitialState = true;
        mesh.vertexBuffer = GetDevice()->createBuffer(vbDesc);
        m_CommandList->writeBuffer(mesh.vertexBuffer, vertices.data(), vbDesc.byteSize);

        nvrhi::BufferDesc ibDesc;
        ibDesc.byteSize = sizeof(uint32_t) * indices.size();
        ibDesc.isIndexBuffer = true;
        ibDesc.initialState = nvrhi::ResourceStates::IndexBuffer;
        ibDesc.keepInitialState = true;
        mesh.indexBuffer = GetDevice()->createBuffer(ibDesc);
        m_CommandList->writeBuffer(mesh.indexBuffer, indices.data(), ibDesc.byteSize);

        mesh.indexCount = static_cast<uint32_t>(indices.size());
        mesh.worldTransform = shape.transform;  // Store model matrix

        // Get material
        if (!shape.materialRef.empty() && m_SceneParser.materials.count(shape.materialRef))
        {
            auto& mat = m_SceneParser.materials[shape.materialRef];
            mesh.baseColor = mat.baseColor;
            mesh.roughness = mat.roughness;
        }
        else if (shape.hasInlineMaterial)
        {
            mesh.baseColor = shape.inlineMaterial.baseColor;
            mesh.roughness = shape.inlineMaterial.roughness;
        }
        else
        {
            mesh.baseColor = HMM_V3(0.5f, 0.5f, 0.5f);
            mesh.roughness = 0.5f;
        }

        mesh.isEmitter = shape.isEmitter;
        mesh.emission = shape.emission;

        m_Meshes.push_back(mesh);

        tinyobj_attrib_free(&attrib);
        tinyobj_shapes_free(shapes, numShapes);
        tinyobj_materials_free(materials, numMaterials);
    }

    void CreateRectangleMesh(MitsubaSceneParser::Shape& shape)
    {
        HMM_Vec3 positions[4] = {
            HMM_V3(-1.0f, -1.0f, 0.0f),
            HMM_V3( 1.0f, -1.0f, 0.0f),
            HMM_V3( 1.0f,  1.0f, 0.0f),
            HMM_V3(-1.0f,  1.0f, 0.0f)
        };

        HMM_Vec3 normal = HMM_V3(0.0f, 0.0f, 1.0f);
        HMM_Vec2 texcoords[4] = {
            HMM_V2(0.0f, 0.0f),
            HMM_V2(1.0f, 0.0f),
            HMM_V2(1.0f, 1.0f),
            HMM_V2(0.0f, 1.0f)
        };

        // Keep vertices in local space (don't pre-transform)
        std::vector<GPUVertex> vertices(4);
        for (int i = 0; i < 4; i++)
        {
            vertices[i].position[0] = positions[i].X;
            vertices[i].position[1] = positions[i].Y;
            vertices[i].position[2] = positions[i].Z;
            vertices[i].normal[0] = normal.X;
            vertices[i].normal[1] = normal.Y;
            vertices[i].normal[2] = normal.Z;
            vertices[i].texcoord[0] = texcoords[i].X;
            vertices[i].texcoord[1] = texcoords[i].Y;
        }

        std::vector<uint32_t> indices = { 0, 1, 2, 0, 2, 3 };

        RenderMesh mesh;

        nvrhi::BufferDesc vbDesc;
        vbDesc.byteSize = sizeof(GPUVertex) * vertices.size();
        vbDesc.isVertexBuffer = true;
        vbDesc.initialState = nvrhi::ResourceStates::VertexBuffer;
        vbDesc.keepInitialState = true;
        mesh.vertexBuffer = GetDevice()->createBuffer(vbDesc);
        m_CommandList->writeBuffer(mesh.vertexBuffer, vertices.data(), vbDesc.byteSize);

        nvrhi::BufferDesc ibDesc;
        ibDesc.byteSize = sizeof(uint32_t) * indices.size();
        ibDesc.isIndexBuffer = true;
        ibDesc.initialState = nvrhi::ResourceStates::IndexBuffer;
        ibDesc.keepInitialState = true;
        mesh.indexBuffer = GetDevice()->createBuffer(ibDesc);
        m_CommandList->writeBuffer(mesh.indexBuffer, indices.data(), ibDesc.byteSize);

        mesh.indexCount = 6;
        mesh.worldTransform = shape.transform;  // Store model matrix

        if (shape.hasInlineMaterial)
        {
            mesh.baseColor = shape.inlineMaterial.baseColor;
            mesh.roughness = shape.inlineMaterial.roughness;
        }
        else
        {
            mesh.baseColor = HMM_V3(0.5f, 0.5f, 0.5f);
            mesh.roughness = 0.5f;
        }

        mesh.isEmitter = shape.isEmitter;
        mesh.emission = shape.emission;

        m_Meshes.push_back(mesh);
    }

    void InitializeCamera()
    {
        // Extract camera position and orientation from scene transform matrix
        // After ParseMatrix mapping for HMM:
        // Columns[j] = (m0j, m1j, m2j, m3j) where mij is element at row i, col j of Mitsuba matrix
        HMM_Mat4& camTransform = m_SceneParser.camera.transform;
        
        // Position: In Mitsuba column-vector, translation is column 3 = (m03, m13, m23)
        // After our mapping: Columns[3] = (m03, m13, m23, m33)
        m_CameraPosition = HMM_V3(
            camTransform.Columns[3].X,
            camTransform.Columns[3].Y,
            camTransform.Columns[3].Z
        );
        
        // Forward direction: Column 2 is local Z axis in world space
        // The to_world matrix's Z column already points in the view direction for Mitsuba
        // (Mitsuba's camera convention may differ from OpenGL's -Z convention)
        HMM_Vec3 forward = HMM_V3(
            camTransform.Columns[2].X,
            camTransform.Columns[2].Y,
            camTransform.Columns[2].Z
        );
        
        // Up vector: column 1 = (m01, m11, m21)
        m_CameraUp = HMM_V3(
            camTransform.Columns[1].X,
            camTransform.Columns[1].Y,
            camTransform.Columns[1].Z
        );
        
        m_CameraTarget = HMM_AddV3(m_CameraPosition, forward);
        
        // Calculate initial yaw and pitch from forward direction
        m_CameraPitch = asinf(-forward.Y);
        m_CameraYaw = atan2f(forward.X, forward.Z);
        
        log::info("Camera position: (%.2f, %.2f, %.2f)", m_CameraPosition.X, m_CameraPosition.Y, m_CameraPosition.Z);
        log::info("Camera forward: (%.2f, %.2f, %.2f)", forward.X, forward.Y, forward.Z);
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        bool pressed = (action == 1 || action == 2);  // GLFW_PRESS or GLFW_REPEAT
        
        switch (key)
        {
            case 'W': m_KeyW = pressed; break;
            case 'S': m_KeyS = pressed; break;
            case 'A': m_KeyA = pressed; break;
            case 'D': m_KeyD = pressed; break;
            case 'Q': m_KeyQ = pressed; break;
            case 'E': m_KeyE = pressed; break;
        }
        return true;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        float dx = float(xpos) - m_LastMouseX;
        float dy = float(ypos) - m_LastMouseY;
        m_LastMouseX = float(xpos);
        m_LastMouseY = float(ypos);
        
        if (m_MouseDown)
        {
            float sensitivity = 0.003f;
            m_CameraYaw += dx * sensitivity;
            m_CameraPitch -= dy * sensitivity;
            
            // Clamp pitch to avoid gimbal lock
            float maxPitch = HMM_PI32 / 2.0f - 0.01f;
            if (m_CameraPitch > maxPitch) m_CameraPitch = maxPitch;
            if (m_CameraPitch < -maxPitch) m_CameraPitch = -maxPitch;
        }
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        if (button == 1)  // Right mouse button
        {
            m_MouseDown = (action == 1);  // GLFW_PRESS
        }
        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        // Calculate forward and right vectors from yaw/pitch
        HMM_Vec3 forward = HMM_V3(
            sinf(m_CameraYaw) * cosf(m_CameraPitch),
            sinf(m_CameraPitch),
            cosf(m_CameraYaw) * cosf(m_CameraPitch)
        );
        HMM_Vec3 right = HMM_NormV3(HMM_Cross(forward, HMM_V3(0.0f, 1.0f, 0.0f)));
        HMM_Vec3 up = HMM_V3(0.0f, 1.0f, 0.0f);
        
        // Movement
        float speed = m_CameraSpeed * fElapsedTimeSeconds;
        if (m_KeyW) m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(forward, speed));
        if (m_KeyS) m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(forward, -speed));
        if (m_KeyA) m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(right, -speed));
        if (m_KeyD) m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(right, speed));
        if (m_KeyE) m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(up, speed));
        if (m_KeyQ) m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(up, -speed));
        
        m_CameraTarget = HMM_AddV3(m_CameraPosition, forward);
        
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    { 
        m_Pipeline = nullptr;
        m_DepthTexture = nullptr;
        m_BindingSet = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        // Create depth texture if needed
        if (!m_DepthTexture)
        {
            nvrhi::TextureDesc depthDesc;
            depthDesc.width = fbinfo.width;
            depthDesc.height = fbinfo.height;
            depthDesc.format = nvrhi::Format::D32;
            depthDesc.isRenderTarget = true;
            depthDesc.initialState = nvrhi::ResourceStates::DepthWrite;
            depthDesc.keepInitialState = true;
            depthDesc.debugName = "DepthBuffer";
            m_DepthTexture = GetDevice()->createTexture(depthDesc);
        }

        // Create framebuffer with current swapchain texture and our depth buffer
        // This must be done each frame since swapchain texture changes
        nvrhi::FramebufferDesc fbDesc;
        fbDesc.addColorAttachment(framebuffer->getDesc().colorAttachments[0].texture);
        fbDesc.setDepthAttachment(m_DepthTexture);
        nvrhi::FramebufferHandle renderFramebuffer = GetDevice()->createFramebuffer(fbDesc);

        // Create pipeline if needed
        if (!m_Pipeline)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_VertexShader;
            psoDesc.PS = m_PixelShader;
            psoDesc.inputLayout = m_InputLayout;
            psoDesc.bindingLayouts = { m_BindingLayout };
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.depthStencilState.depthTestEnable = true;
            psoDesc.renderState.depthStencilState.depthWriteEnable = true;
            psoDesc.renderState.depthStencilState.depthFunc = nvrhi::ComparisonFunc::Less;
            psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;

            m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, renderFramebuffer->getFramebufferInfo());
        }

        // Create binding set if needed (cached)
        if (!m_BindingSet)
        {
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::ConstantBuffer(0, m_PerObjectBuffer),
                nvrhi::BindingSetItem::ConstantBuffer(1, m_LightBuffer)
            };
            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
        }

        // Calculate view/projection matrices using HandmadeMath
        float aspect = float(fbinfo.width) / float(fbinfo.height);
        
        // Mitsuba uses horizontal FOV by default, convert to vertical FOV
        float horizontalFovRadians = m_SceneParser.camera.fov * (HMM_PI32 / 180.0f);
        float verticalFovRadians = 2.0f * atanf(tanf(horizontalFovRadians * 0.5f) / aspect);
        
        // Use HMM right-handed perspective with [0,1] depth range for D3D/Vulkan
        // RH: camera looks down -Z, which matches Mitsuba convention
        HMM_Mat4 view = HMM_LookAt_RH(m_CameraPosition, m_CameraTarget, HMM_V3(0.0f, 1.0f, 0.0f));
        HMM_Mat4 proj = HMM_Perspective_RH_ZO(verticalFovRadians, aspect, 0.1f, 10000.0f);
        
        // For column-vector convention: VP = Proj * View
        // HMM uses column-vector math: v' = M * v
        // So MVP = Proj * View * Model, applied as v_clip = MVP * v_local
        HMM_Mat4 viewProj = HMM_MulM4(proj, view);  // P * V

        m_CommandList->open();

        // Clear framebuffer
        nvrhi::utils::ClearColorAttachment(m_CommandList, renderFramebuffer, 0, nvrhi::Color(0.1f, 0.2f, 0.3f, 1.0f));
        
        // Clear depth
        m_CommandList->clearDepthStencilTexture(m_DepthTexture, 
            nvrhi::AllSubresources, true, 1.0f, false, 0);

        // Update light constants (once per frame)
        LightConstants lightConstants;
        HMM_Vec3 lightDir = HMM_NormV3(HMM_V3(0.5f, 1.0f, 0.3f));
        lightConstants.lightDir[0] = lightDir.X;
        lightConstants.lightDir[1] = lightDir.Y;
        lightConstants.lightDir[2] = lightDir.Z;
        lightConstants.lightColor[0] = 1.0f;
        lightConstants.lightColor[1] = 0.98f;
        lightConstants.lightColor[2] = 0.95f;
        lightConstants.ambientColor[0] = 0.15f;
        lightConstants.ambientColor[1] = 0.15f;
        lightConstants.ambientColor[2] = 0.2f;
        lightConstants.cameraPos[0] = m_CameraPosition.X;
        lightConstants.cameraPos[1] = m_CameraPosition.Y;
        lightConstants.cameraPos[2] = m_CameraPosition.Z;
        m_CommandList->writeBuffer(m_LightBuffer, &lightConstants, sizeof(LightConstants));

        // Debug: print first frame info
        static bool firstFrame = true;
        if (firstFrame && !m_Meshes.empty())
        {
            log::info("=== DEBUG MVP ===");
            log::info("Camera pos: (%.2f, %.2f, %.2f)", m_CameraPosition.X, m_CameraPosition.Y, m_CameraPosition.Z);
            log::info("Camera target: (%.2f, %.2f, %.2f)", m_CameraTarget.X, m_CameraTarget.Y, m_CameraTarget.Z);
            log::info("View matrix:");
            log::info("  col0: %.3f %.3f %.3f %.3f", view.Columns[0].X, view.Columns[0].Y, view.Columns[0].Z, view.Columns[0].W);
            log::info("  col1: %.3f %.3f %.3f %.3f", view.Columns[1].X, view.Columns[1].Y, view.Columns[1].Z, view.Columns[1].W);
            log::info("  col2: %.3f %.3f %.3f %.3f", view.Columns[2].X, view.Columns[2].Y, view.Columns[2].Z, view.Columns[2].W);
            log::info("  col3: %.3f %.3f %.3f %.3f", view.Columns[3].X, view.Columns[3].Y, view.Columns[3].Z, view.Columns[3].W);
            log::info("Proj matrix:");
            log::info("  col0: %.3f %.3f %.3f %.3f", proj.Columns[0].X, proj.Columns[0].Y, proj.Columns[0].Z, proj.Columns[0].W);
            log::info("  col1: %.3f %.3f %.3f %.3f", proj.Columns[1].X, proj.Columns[1].Y, proj.Columns[1].Z, proj.Columns[1].W);
            log::info("  col2: %.3f %.3f %.3f %.3f", proj.Columns[2].X, proj.Columns[2].Y, proj.Columns[2].Z, proj.Columns[2].W);
            log::info("  col3: %.3f %.3f %.3f %.3f", proj.Columns[3].X, proj.Columns[3].Y, proj.Columns[3].Z, proj.Columns[3].W);
            
            // Test transform a vertex from first mesh
            HMM_Mat4 mvp0 = HMM_MulM4(viewProj, m_Meshes[0].worldTransform);
            HMM_Vec4 testV = HMM_V4(0.0f, 0.0f, 0.0f, 1.0f);  // origin in local space
            HMM_Vec4 clipPos = HMM_MulM4V4(mvp0, testV);
            log::info("Test vertex (0,0,0,1) -> clip: (%.3f, %.3f, %.3f, %.3f)", clipPos.X, clipPos.Y, clipPos.Z, clipPos.W);
            if (clipPos.W != 0.0f)
            {
                log::info("  NDC: (%.3f, %.3f, %.3f)", clipPos.X/clipPos.W, clipPos.Y/clipPos.W, clipPos.Z/clipPos.W);
            }
            firstFrame = false;
        }

        // Render each mesh
        for (auto& mesh : m_Meshes)
        {
            // Calculate MVP = Proj * View * Model (column-vector convention)
            // v_clip = MVP * v_local = P * V * M * v
            HMM_Mat4 mvp = HMM_MulM4(viewProj, mesh.worldTransform);  // (P * V) * M = P * V * M
            
            // Send matrices directly without transpose
            // HMM uses column-major storage, HLSL default is also column-major
            // In shader we use mul(M, v) for column-vector math
            PerObjectConstants perObject;
            memcpy(perObject.worldViewProj, &mvp, sizeof(float) * 16);
            memcpy(perObject.world, &mesh.worldTransform, sizeof(float) * 16);
            perObject.baseColor[0] = mesh.baseColor.X;
            perObject.baseColor[1] = mesh.baseColor.Y;
            perObject.baseColor[2] = mesh.baseColor.Z;
            perObject.roughness = mesh.roughness;
            perObject.emission[0] = mesh.emission.X;
            perObject.emission[1] = mesh.emission.Y;
            perObject.emission[2] = mesh.emission.Z;
            perObject.isEmitter = mesh.isEmitter ? 1 : 0;
            m_CommandList->writeBuffer(m_PerObjectBuffer, &perObject, sizeof(PerObjectConstants));

            nvrhi::GraphicsState state;
            state.pipeline = m_Pipeline;
            state.framebuffer = renderFramebuffer;
            state.bindings = { m_BindingSet };
            state.vertexBuffers = { { mesh.vertexBuffer, 0, 0 } };
            state.indexBuffer = { mesh.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            state.viewport.addViewportAndScissorRect(renderFramebuffer->getFramebufferInfo().getViewport());

            m_CommandList->setGraphicsState(state);

            nvrhi::DrawArguments args;
            args.vertexCount = mesh.indexCount;
            m_CommandList->drawIndexed(args);
        }

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }
};

// ============================================================================
// Entry Point
// ============================================================================
#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    // Enable console output for logging on Windows
    log::EnableOutputToConsole(true);
    
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
    deviceParams.enablePerMonitorDPI = true;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    // Get scene path from command line
    std::filesystem::path scenePath;
    
#ifdef WIN32
    int argc;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    
    for (int i = 1; i < argc; i++)
    {
        std::wstring arg = argv[i];
        if (arg.find(L".xml") != std::wstring::npos)
        {
            scenePath = arg;
            break;
        }
    }
    LocalFree(argv);
#else
    for (int i = 1; i < __argc; i++)
    {
        std::string arg = __argv[i];
        if (arg.find(".xml") != std::string::npos)
        {
            scenePath = arg;
            break;
        }
    }
#endif

    if (scenePath.empty())
    {
        log::info("Usage: mitsuba_scene <scene.xml>");
        scenePath = "E:/SW/CG/mitsuba3/scenes/bathroom2/bathroom2/scene.xml";
        log::info("Trying default path: %s", scenePath.string().c_str());
    }

    if (!std::filesystem::exists(scenePath))
    {
        log::fatal("Scene file does not exist: %s", scenePath.string().c_str());
        return 1;
    }

    {
        MitsubaSceneRasterizer example(deviceManager, scenePath);
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;

    return 0;
}
