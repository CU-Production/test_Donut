
#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>

#include <pugixml.hpp>

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include <tinyobj_loader_c.h>

#include <filesystem>
#include <unordered_map>
#include <sstream>

using namespace donut;
using namespace donut::math;

static const char* g_WindowTitle = "Donut Example: Mitsuba Scene Ray Tracer";

// ============================================================================
// Material Types (matching Mitsuba BSDF types)
// ============================================================================
enum class MaterialType : uint32_t
{
    Diffuse = 0,
    Conductor = 1,
    RoughConductor = 2,
    Dielectric = 3,
    RoughDielectric = 4,
    Plastic = 5,
    RoughPlastic = 6
};

// ============================================================================
// GPU Structures (must match HLSL)
// ============================================================================
struct GPUMaterial
{
    float3 baseColor;
    float roughness;
    
    float3 eta;          // For conductors: complex IOR real part
    float metallic;
    
    float3 k;            // For conductors: complex IOR imaginary part
    uint32_t type;
    
    float intIOR;        // Interior index of refraction
    float extIOR;        // Exterior index of refraction
    float2 padding;
};

struct GPUVertex
{
    float3 position;
    float pad0;
    float3 normal;
    float pad1;
    float2 texcoord;
    float2 pad2;
};

struct GPUInstance
{
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t materialIndex;
    uint32_t isEmitter;
    float3 emission;
    float pad;
};

struct CameraConstants
{
    float4x4 viewInverse;
    float4x4 projInverse;
    float3 cameraPosition;
    uint32_t frameIndex;
    uint32_t samplesPerPixel;
    uint32_t maxBounces;
    float2 padding;
};

// ============================================================================
// Mitsuba Scene Parser
// ============================================================================
class MitsubaSceneParser
{
public:
    struct Camera
    {
        float4x4 transform = float4x4::identity();
        float fov = 45.0f;
        int width = 1280;
        int height = 720;
    };

    struct Material
    {
        std::string id;
        MaterialType type = MaterialType::Diffuse;
        float3 baseColor = float3(0.5f);
        float roughness = 0.5f;
        float3 eta = float3(1.0f);
        float3 k = float3(0.0f);
        float intIOR = 1.5f;
        float extIOR = 1.0f;
    };

    struct Shape
    {
        std::string type;           // "obj" or "rectangle"
        std::string filename;       // OBJ filename
        std::string materialRef;    // Reference to material ID
        float4x4 transform = float4x4::identity();
        bool isEmitter = false;
        float3 emission = float3(0.0f);
        
        // For inline materials
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

        // Parse all children
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
    // Parse a 4x4 matrix from Mitsuba format (row-major, 16 floats)
    float4x4 ParseMatrix(const std::string& matrixStr)
    {
        std::istringstream iss(matrixStr);
        float values[16];
        for (int i = 0; i < 16; i++)
        {
            iss >> values[i];
        }
        
        // Mitsuba uses row-major, we need column-major
        return float4x4(
            values[0], values[4], values[8], values[12],
            values[1], values[5], values[9], values[13],
            values[2], values[6], values[10], values[14],
            values[3], values[7], values[11], values[15]
        );
    }

    // Parse RGB color from "r, g, b" format
    float3 ParseRGB(const std::string& rgbStr)
    {
        float3 color(0.0f);
        std::string cleaned = rgbStr;
        // Remove commas
        for (char& c : cleaned)
        {
            if (c == ',') c = ' ';
        }
        std::istringstream iss(cleaned);
        iss >> color.x >> color.y >> color.z;
        return color;
    }

    void ParseSensor(pugi::xml_node sensorNode)
    {
        // Parse FOV
        for (pugi::xml_node child : sensorNode.children("float"))
        {
            std::string name = child.attribute("name").value();
            if (name == "fov")
            {
                camera.fov = child.attribute("value").as_float(45.0f);
            }
        }

        // Parse transform
        pugi::xml_node transformNode = sensorNode.child("transform");
        if (transformNode)
        {
            pugi::xml_node matrixNode = transformNode.child("matrix");
            if (matrixNode)
            {
                camera.transform = ParseMatrix(matrixNode.attribute("value").value());
            }
        }

        // Parse film (resolution)
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

        // Handle twosided wrapper
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

        // Set material type
        if (type == "diffuse")
        {
            mat.type = MaterialType::Diffuse;
            mat.roughness = 1.0f;
        }
        else if (type == "conductor")
        {
            mat.type = MaterialType::Conductor;
            mat.roughness = 0.0f;
            mat.baseColor = float3(1.0f);
        }
        else if (type == "roughconductor")
        {
            mat.type = MaterialType::RoughConductor;
        }
        else if (type == "dielectric")
        {
            mat.type = MaterialType::Dielectric;
            mat.roughness = 0.0f;
        }
        else if (type == "roughdielectric")
        {
            mat.type = MaterialType::RoughDielectric;
        }
        else if (type == "plastic")
        {
            mat.type = MaterialType::Plastic;
            mat.roughness = 0.0f;
        }
        else if (type == "roughplastic")
        {
            mat.type = MaterialType::RoughPlastic;
        }

        // Parse material properties
        for (pugi::xml_node child : bsdfNode.children())
        {
            std::string childName = child.name();
            std::string propName = child.attribute("name").value();

            if (childName == "rgb")
            {
                float3 color = ParseRGB(child.attribute("value").value());
                if (propName == "reflectance" || propName == "diffuse_reflectance" || propName == "specular_reflectance")
                {
                    mat.baseColor = color;
                }
                else if (propName == "eta")
                {
                    mat.eta = color;
                }
                else if (propName == "k")
                {
                    mat.k = color;
                }
            }
            else if (childName == "float")
            {
                float value = child.attribute("value").as_float();
                if (propName == "alpha")
                {
                    mat.roughness = value;
                }
                else if (propName == "int_ior")
                {
                    mat.intIOR = value;
                }
                else if (propName == "ext_ior")
                {
                    mat.extIOR = value;
                }
            }
        }

        return mat;
    }

    void ParseShape(pugi::xml_node shapeNode)
    {
        Shape shape;
        shape.type = shapeNode.attribute("type").value();

        // Parse transform
        pugi::xml_node transformNode = shapeNode.child("transform");
        if (transformNode)
        {
            pugi::xml_node matrixNode = transformNode.child("matrix");
            if (matrixNode)
            {
                shape.transform = ParseMatrix(matrixNode.attribute("value").value());
            }
        }

        // Parse OBJ filename
        for (pugi::xml_node child : shapeNode.children("string"))
        {
            std::string name = child.attribute("name").value();
            if (name == "filename")
            {
                shape.filename = child.attribute("value").value();
            }
        }

        // Parse material reference
        pugi::xml_node refNode = shapeNode.child("ref");
        if (refNode)
        {
            shape.materialRef = refNode.attribute("id").value();
        }

        // Parse inline BSDF
        pugi::xml_node inlineBsdf = shapeNode.child("bsdf");
        if (inlineBsdf)
        {
            shape.inlineMaterial = ParseBSDF(inlineBsdf, true);
            shape.hasInlineMaterial = true;
        }

        // Parse emitter
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
// OBJ Loader Callback for tinyobjloader-c
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
// Ray Traced Scene Application
// ============================================================================
class RayTracedScene : public app::IRenderPass
{
private:
    // Shader and pipeline handles
    nvrhi::ShaderLibraryHandle m_ShaderLibrary;
    nvrhi::rt::PipelineHandle m_Pipeline;
    nvrhi::rt::ShaderTableHandle m_ShaderTable;
    nvrhi::CommandListHandle m_CommandList;
    nvrhi::BindingLayoutHandle m_BindingLayout;
    nvrhi::BindingSetHandle m_BindingSet;

    // Acceleration structures
    std::vector<nvrhi::rt::AccelStructHandle> m_BottomLevelAS;
    nvrhi::rt::AccelStructHandle m_TopLevelAS;

    // Buffers
    nvrhi::BufferHandle m_VertexBuffer;
    nvrhi::BufferHandle m_IndexBuffer;
    nvrhi::BufferHandle m_MaterialBuffer;
    nvrhi::BufferHandle m_InstanceBuffer;
    nvrhi::BufferHandle m_CameraBuffer;

    // Render target
    nvrhi::TextureHandle m_RenderTarget;
    nvrhi::TextureHandle m_AccumulationTarget;

    // Render passes
    std::shared_ptr<engine::CommonRenderPasses> m_CommonPasses;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    // Scene data
    MitsubaSceneParser m_SceneParser;
    std::vector<GPUVertex> m_Vertices;
    std::vector<uint32_t> m_Indices;
    std::vector<GPUMaterial> m_Materials;
    std::vector<GPUInstance> m_Instances;

    // Camera
    CameraConstants m_CameraConstants;
    float3 m_CameraPosition;
    float3 m_CameraTarget;
    float3 m_CameraUp;
    uint32_t m_FrameIndex = 0;

    // Scene path
    std::filesystem::path m_ScenePath;

public:
    RayTracedScene(app::DeviceManager* deviceManager, const std::filesystem::path& scenePath)
        : IRenderPass(deviceManager)
        , m_ScenePath(scenePath)
    {
    }

    bool Init()
    {
        // Parse the Mitsuba scene
        if (!m_SceneParser.Parse(m_ScenePath))
        {
            log::error("Failed to parse scene file: %s", m_ScenePath.string().c_str());
            return false;
        }

        // Load geometry from scene
        if (!LoadSceneGeometry())
        {
            log::error("Failed to load scene geometry");
            return false;
        }

        // Initialize shader factory
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/rt_scene" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());

        auto rootFS = std::make_shared<vfs::RootFileSystem>();
        rootFS->mount("/shaders/donut", frameworkShaderPath);
        rootFS->mount("/shaders/app", appShaderPath);

        auto shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), rootFS, "/shaders");
        m_ShaderLibrary = shaderFactory->CreateShaderLibrary("app/rt_scene.hlsl", nullptr);

        if (!m_ShaderLibrary)
        {
            log::error("Failed to create shader library");
            return false;
        }

        m_BindingCache = std::make_unique<engine::BindingCache>(GetDevice());
        m_CommonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), shaderFactory);

        // Create binding layout
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),     // t0: TLAS
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),      // t1: Vertices
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),      // t2: Indices
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),      // t3: Materials
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),      // t4: Instances
            nvrhi::BindingLayoutItem::Texture_UAV(0),               // u0: Output
            nvrhi::BindingLayoutItem::Texture_UAV(1),               // u1: Accumulation
            nvrhi::BindingLayoutItem::ConstantBuffer(0)             // b0: Camera
        };
        m_BindingLayout = GetDevice()->createBindingLayout(bindingLayoutDesc);

        // Create ray tracing pipeline
        nvrhi::rt::PipelineDesc pipelineDesc;
        pipelineDesc.globalBindingLayouts = { m_BindingLayout };
        pipelineDesc.shaders = {
            { "", m_ShaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration), nullptr },
            { "", m_ShaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss), nullptr },
            { "", m_ShaderLibrary->getShader("ShadowMiss", nvrhi::ShaderType::Miss), nullptr }
        };

        pipelineDesc.hitGroups = { {
            "HitGroup",
            m_ShaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
            nullptr, // anyHitShader
            nullptr, // intersectionShader
            nullptr, // bindingLayout
            false    // isProceduralPrimitive
        }, {
            "ShadowHitGroup",
            m_ShaderLibrary->getShader("ShadowClosestHit", nvrhi::ShaderType::ClosestHit),
            nullptr,
            nullptr,
            nullptr,
            false
        } };

        pipelineDesc.maxPayloadSize = sizeof(float) * 8;  // HitInfo struct
        pipelineDesc.maxRecursionDepth = 8;

        m_Pipeline = GetDevice()->createRayTracingPipeline(pipelineDesc);

        m_ShaderTable = m_Pipeline->createShaderTable();
        m_ShaderTable->setRayGenerationShader("RayGen");
        m_ShaderTable->addHitGroup("HitGroup");
        m_ShaderTable->addHitGroup("ShadowHitGroup");
        m_ShaderTable->addMissShader("Miss");
        m_ShaderTable->addMissShader("ShadowMiss");

        m_CommandList = GetDevice()->createCommandList();

        // Create GPU buffers and acceleration structures
        CreateGPUResources();

        // Setup camera from scene
        SetupCameraFromScene();

        return true;
    }

    bool LoadSceneGeometry()
    {
        uint32_t currentVertexOffset = 0;
        uint32_t currentIndexOffset = 0;
        uint32_t currentMaterialIndex = 0;

        // Create materials from parsed scene
        for (auto& [id, mat] : m_SceneParser.materials)
        {
            GPUMaterial gpuMat;
            gpuMat.baseColor = mat.baseColor;
            gpuMat.roughness = mat.roughness;
            gpuMat.eta = mat.eta;
            gpuMat.k = mat.k;
            gpuMat.type = static_cast<uint32_t>(mat.type);
            gpuMat.intIOR = mat.intIOR;
            gpuMat.extIOR = mat.extIOR;
            gpuMat.metallic = (mat.type == MaterialType::Conductor || mat.type == MaterialType::RoughConductor) ? 1.0f : 0.0f;
            m_Materials.push_back(gpuMat);
        }

        // Process each shape
        for (auto& shape : m_SceneParser.shapes)
        {
            if (shape.type == "obj")
            {
                LoadOBJShape(shape, currentVertexOffset, currentIndexOffset, currentMaterialIndex);
            }
            else if (shape.type == "rectangle")
            {
                CreateRectangleShape(shape, currentVertexOffset, currentIndexOffset, currentMaterialIndex);
            }
        }

        log::info("Loaded %zu vertices, %zu indices, %zu materials, %zu instances",
            m_Vertices.size(), m_Indices.size(), m_Materials.size(), m_Instances.size());

        return !m_Vertices.empty();
    }

    void LoadOBJShape(MitsubaSceneParser::Shape& shape, uint32_t& vertexOffset, 
                      uint32_t& indexOffset, uint32_t& materialIndex)
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

        uint32_t startVertexIndex = static_cast<uint32_t>(m_Vertices.size());
        uint32_t startIndexOffset = static_cast<uint32_t>(m_Indices.size());

        // Convert vertices with transform
        std::unordered_map<uint64_t, uint32_t> vertexMap;

        for (unsigned int f = 0; f < attrib.num_faces; f++)
        {
            tinyobj_vertex_index_t idx = attrib.faces[f];

            // Create unique key for vertex
            uint64_t key = (uint64_t(idx.v_idx) << 40) | (uint64_t(idx.vn_idx) << 20) | uint64_t(idx.vt_idx);

            auto it = vertexMap.find(key);
            if (it != vertexMap.end())
            {
                m_Indices.push_back(it->second);
            }
            else
            {
                GPUVertex vertex;

                // Position
                float4 pos = float4(
                    attrib.vertices[3 * idx.v_idx + 0],
                    attrib.vertices[3 * idx.v_idx + 1],
                    attrib.vertices[3 * idx.v_idx + 2],
                    1.0f
                );
                pos = pos * shape.transform;
                vertex.position = float3(pos.x, pos.y, pos.z);

                // Normal
                if (idx.vn_idx >= 0 && attrib.normals)
                {
                    float4 normal = float4(
                        attrib.normals[3 * idx.vn_idx + 0],
                        attrib.normals[3 * idx.vn_idx + 1],
                        attrib.normals[3 * idx.vn_idx + 2],
                        0.0f
                    );
                    // Transform normal (use inverse transpose for correct normal transformation)
                    normal = normal * shape.transform;
                    vertex.normal = normalize(float3(normal.x, normal.y, normal.z));
                }
                else
                {
                    vertex.normal = float3(0.0f, 1.0f, 0.0f);
                }

                // Texcoord
                if (idx.vt_idx >= 0 && attrib.texcoords)
                {
                    vertex.texcoord = float2(
                        attrib.texcoords[2 * idx.vt_idx + 0],
                        attrib.texcoords[2 * idx.vt_idx + 1]
                    );
                }
                else
                {
                    vertex.texcoord = float2(0.0f);
                }

                uint32_t newIndex = static_cast<uint32_t>(m_Vertices.size());
                vertexMap[key] = newIndex;
                m_Vertices.push_back(vertex);
                m_Indices.push_back(newIndex);
            }
        }

        // Find material index
        uint32_t matIndex = 0;
        if (!shape.materialRef.empty())
        {
            uint32_t idx = 0;
            for (auto& [id, mat] : m_SceneParser.materials)
            {
                if (id == shape.materialRef)
                {
                    matIndex = idx;
                    break;
                }
                idx++;
            }
        }
        else if (shape.hasInlineMaterial)
        {
            // Add inline material
            GPUMaterial gpuMat;
            gpuMat.baseColor = shape.inlineMaterial.baseColor;
            gpuMat.roughness = shape.inlineMaterial.roughness;
            gpuMat.eta = shape.inlineMaterial.eta;
            gpuMat.k = shape.inlineMaterial.k;
            gpuMat.type = static_cast<uint32_t>(shape.inlineMaterial.type);
            gpuMat.intIOR = shape.inlineMaterial.intIOR;
            gpuMat.extIOR = shape.inlineMaterial.extIOR;
            gpuMat.metallic = 0.0f;
            matIndex = static_cast<uint32_t>(m_Materials.size());
            m_Materials.push_back(gpuMat);
        }

        // Create instance
        GPUInstance instance;
        instance.vertexOffset = startVertexIndex;
        instance.indexOffset = startIndexOffset;
        instance.materialIndex = matIndex;
        instance.isEmitter = shape.isEmitter ? 1 : 0;
        instance.emission = shape.emission;
        m_Instances.push_back(instance);

        // Cleanup
        tinyobj_attrib_free(&attrib);
        tinyobj_shapes_free(shapes, numShapes);
        tinyobj_materials_free(materials, numMaterials);
    }

    void CreateRectangleShape(MitsubaSceneParser::Shape& shape, uint32_t& vertexOffset,
                              uint32_t& indexOffset, uint32_t& materialIndex)
    {
        uint32_t startVertexIndex = static_cast<uint32_t>(m_Vertices.size());
        uint32_t startIndexOffset = static_cast<uint32_t>(m_Indices.size());

        // Create a unit rectangle in XY plane, centered at origin
        float3 positions[4] = {
            float3(-1.0f, -1.0f, 0.0f),
            float3( 1.0f, -1.0f, 0.0f),
            float3( 1.0f,  1.0f, 0.0f),
            float3(-1.0f,  1.0f, 0.0f)
        };

        float3 normal = float3(0.0f, 0.0f, 1.0f);
        float2 texcoords[4] = {
            float2(0.0f, 0.0f),
            float2(1.0f, 0.0f),
            float2(1.0f, 1.0f),
            float2(0.0f, 1.0f)
        };

        // Transform and add vertices
        for (int i = 0; i < 4; i++)
        {
            GPUVertex vertex;
            float4 pos = float4(positions[i], 1.0f);
            pos = pos * shape.transform;
            vertex.position = float3(pos.x, pos.y, pos.z);

            float4 n = float4(normal, 0.0f);
            n = n * shape.transform;
            vertex.normal = normalize(float3(n.x, n.y, n.z));

            vertex.texcoord = texcoords[i];
            m_Vertices.push_back(vertex);
        }

        // Add indices (two triangles)
        uint32_t base = startVertexIndex;
        m_Indices.push_back(base + 0);
        m_Indices.push_back(base + 1);
        m_Indices.push_back(base + 2);
        m_Indices.push_back(base + 0);
        m_Indices.push_back(base + 2);
        m_Indices.push_back(base + 3);

        // Handle material
        uint32_t matIndex = 0;
        if (shape.hasInlineMaterial)
        {
            GPUMaterial gpuMat;
            gpuMat.baseColor = shape.inlineMaterial.baseColor;
            gpuMat.roughness = shape.inlineMaterial.roughness;
            gpuMat.eta = shape.inlineMaterial.eta;
            gpuMat.k = shape.inlineMaterial.k;
            gpuMat.type = static_cast<uint32_t>(shape.inlineMaterial.type);
            gpuMat.intIOR = shape.inlineMaterial.intIOR;
            gpuMat.extIOR = shape.inlineMaterial.extIOR;
            gpuMat.metallic = 0.0f;
            matIndex = static_cast<uint32_t>(m_Materials.size());
            m_Materials.push_back(gpuMat);
        }

        // Create instance
        GPUInstance instance;
        instance.vertexOffset = startVertexIndex;
        instance.indexOffset = startIndexOffset;
        instance.materialIndex = matIndex;
        instance.isEmitter = shape.isEmitter ? 1 : 0;
        instance.emission = shape.emission;
        m_Instances.push_back(instance);
    }

    void CreateGPUResources()
    {
        m_CommandList->open();

        // Create vertex buffer
        nvrhi::BufferDesc vertexBufferDesc;
        vertexBufferDesc.byteSize = sizeof(GPUVertex) * m_Vertices.size();
        vertexBufferDesc.structStride = sizeof(GPUVertex);
        vertexBufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        vertexBufferDesc.keepInitialState = true;
        vertexBufferDesc.isAccelStructBuildInput = true;
        vertexBufferDesc.debugName = "VertexBuffer";
        m_VertexBuffer = GetDevice()->createBuffer(vertexBufferDesc);
        m_CommandList->writeBuffer(m_VertexBuffer, m_Vertices.data(), vertexBufferDesc.byteSize);

        // Create index buffer
        nvrhi::BufferDesc indexBufferDesc;
        indexBufferDesc.byteSize = sizeof(uint32_t) * m_Indices.size();
        indexBufferDesc.structStride = sizeof(uint32_t);
        indexBufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        indexBufferDesc.keepInitialState = true;
        indexBufferDesc.isAccelStructBuildInput = true;
        indexBufferDesc.debugName = "IndexBuffer";
        m_IndexBuffer = GetDevice()->createBuffer(indexBufferDesc);
        m_CommandList->writeBuffer(m_IndexBuffer, m_Indices.data(), indexBufferDesc.byteSize);

        // Create material buffer
        if (!m_Materials.empty())
        {
            nvrhi::BufferDesc materialBufferDesc;
            materialBufferDesc.byteSize = sizeof(GPUMaterial) * m_Materials.size();
            materialBufferDesc.structStride = sizeof(GPUMaterial);
            materialBufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            materialBufferDesc.keepInitialState = true;
            materialBufferDesc.debugName = "MaterialBuffer";
            m_MaterialBuffer = GetDevice()->createBuffer(materialBufferDesc);
            m_CommandList->writeBuffer(m_MaterialBuffer, m_Materials.data(), materialBufferDesc.byteSize);
        }

        // Create instance buffer
        if (!m_Instances.empty())
        {
            nvrhi::BufferDesc instanceBufferDesc;
            instanceBufferDesc.byteSize = sizeof(GPUInstance) * m_Instances.size();
            instanceBufferDesc.structStride = sizeof(GPUInstance);
            instanceBufferDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            instanceBufferDesc.keepInitialState = true;
            instanceBufferDesc.debugName = "InstanceBuffer";
            m_InstanceBuffer = GetDevice()->createBuffer(instanceBufferDesc);
            m_CommandList->writeBuffer(m_InstanceBuffer, m_Instances.data(), instanceBufferDesc.byteSize);
        }

        // Create camera constant buffer
        nvrhi::BufferDesc cameraBufferDesc;
        cameraBufferDesc.byteSize = sizeof(CameraConstants);
        cameraBufferDesc.isConstantBuffer = true;
        cameraBufferDesc.initialState = nvrhi::ResourceStates::ConstantBuffer;
        cameraBufferDesc.keepInitialState = true;
        cameraBufferDesc.debugName = "CameraBuffer";
        m_CameraBuffer = GetDevice()->createBuffer(cameraBufferDesc);

        // Build acceleration structures
        BuildAccelerationStructures();

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }

    void BuildAccelerationStructures()
    {
        std::vector<nvrhi::rt::InstanceDesc> tlasInstances;

        // Build BLAS for each instance
        for (size_t i = 0; i < m_Instances.size(); i++)
        {
            const GPUInstance& instance = m_Instances[i];
            uint32_t indexCount = 0;

            // Calculate index count for this instance
            if (i < m_Instances.size() - 1)
            {
                indexCount = m_Instances[i + 1].indexOffset - instance.indexOffset;
            }
            else
            {
                indexCount = static_cast<uint32_t>(m_Indices.size()) - instance.indexOffset;
            }

            uint32_t vertexCount = 0;
            if (i < m_Instances.size() - 1)
            {
                vertexCount = m_Instances[i + 1].vertexOffset - instance.vertexOffset;
            }
            else
            {
                vertexCount = static_cast<uint32_t>(m_Vertices.size()) - instance.vertexOffset;
            }

            // Create BLAS
            nvrhi::rt::AccelStructDesc blasDesc;
            blasDesc.isTopLevel = false;

            nvrhi::rt::GeometryDesc geometryDesc;
            auto& triangles = geometryDesc.geometryData.triangles;
            triangles.indexBuffer = m_IndexBuffer;
            triangles.indexOffset = instance.indexOffset * sizeof(uint32_t);
            triangles.indexFormat = nvrhi::Format::R32_UINT;
            triangles.indexCount = indexCount;
            triangles.vertexBuffer = m_VertexBuffer;
            triangles.vertexOffset = instance.vertexOffset * sizeof(GPUVertex);
            triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            triangles.vertexStride = sizeof(GPUVertex);
            triangles.vertexCount = vertexCount;
            geometryDesc.geometryType = nvrhi::rt::GeometryType::Triangles;
            geometryDesc.flags = nvrhi::rt::GeometryFlags::Opaque;
            blasDesc.bottomLevelGeometries.push_back(geometryDesc);

            nvrhi::rt::AccelStructHandle blas = GetDevice()->createAccelStruct(blasDesc);
            nvrhi::utils::BuildBottomLevelAccelStruct(m_CommandList, blas, blasDesc);
            m_BottomLevelAS.push_back(blas);

            // Create TLAS instance
            nvrhi::rt::InstanceDesc instanceDesc;
            instanceDesc.bottomLevelAS = blas;
            instanceDesc.instanceMask = 1;
            instanceDesc.instanceID = static_cast<uint32_t>(i);
            instanceDesc.flags = nvrhi::rt::InstanceFlags::TriangleFrontCounterclockwise;
            float3x4 transform = float3x4::identity();
            memcpy(instanceDesc.transform, &transform, sizeof(transform));
            tlasInstances.push_back(instanceDesc);
        }

        // Build TLAS
        nvrhi::rt::AccelStructDesc tlasDesc;
        tlasDesc.isTopLevel = true;
        tlasDesc.topLevelMaxInstances = static_cast<uint32_t>(tlasInstances.size());

        m_TopLevelAS = GetDevice()->createAccelStruct(tlasDesc);
        m_CommandList->buildTopLevelAccelStruct(m_TopLevelAS, tlasInstances.data(), tlasInstances.size());
    }

    void SetupCameraFromScene()
    {
        // Extract camera position and orientation from transform matrix
        float4x4 viewInverse = m_SceneParser.camera.transform;
        
        m_CameraPosition = float3(viewInverse.m03, viewInverse.m13, viewInverse.m23);
        
        // Initialize camera constants
        m_CameraConstants.samplesPerPixel = 1;
        m_CameraConstants.maxBounces = 8;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    {
        m_RenderTarget = nullptr;
        m_AccumulationTarget = nullptr;
        m_BindingCache->Clear();
        m_FrameIndex = 0;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        // Create render targets if needed
        if (!m_RenderTarget)
        {
            nvrhi::TextureDesc textureDesc;
            textureDesc.width = fbinfo.width;
            textureDesc.height = fbinfo.height;
            textureDesc.isUAV = true;
            textureDesc.initialState = nvrhi::ResourceStates::UnorderedAccess;
            textureDesc.keepInitialState = true;
            textureDesc.format = nvrhi::Format::RGBA32_FLOAT;
            textureDesc.debugName = "RenderTarget";
            m_RenderTarget = GetDevice()->createTexture(textureDesc);

            textureDesc.debugName = "AccumulationTarget";
            m_AccumulationTarget = GetDevice()->createTexture(textureDesc);

            // Create binding set
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_TopLevelAS),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_VertexBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_IndexBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_MaterialBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_InstanceBuffer),
                nvrhi::BindingSetItem::Texture_UAV(0, m_RenderTarget),
                nvrhi::BindingSetItem::Texture_UAV(1, m_AccumulationTarget),
                nvrhi::BindingSetItem::ConstantBuffer(0, m_CameraBuffer)
            };
            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
        }

        // Update camera constants
        float aspect = float(fbinfo.width) / float(fbinfo.height);
        float fovRadians = m_SceneParser.camera.fov * (3.14159265f / 180.0f);

        float4x4 proj = perspProjD3DStyleReverse(fovRadians, aspect, 0.1f);
        float4x4 view = m_SceneParser.camera.transform;

        m_CameraConstants.viewInverse = view;
        m_CameraConstants.projInverse = inverse(proj);
        m_CameraConstants.cameraPosition = float3(view.m03, view.m13, view.m23);
        m_CameraConstants.frameIndex = m_FrameIndex;

        m_CommandList->open();

        // Update camera buffer
        m_CommandList->writeBuffer(m_CameraBuffer, &m_CameraConstants, sizeof(CameraConstants));

        // Dispatch rays
        nvrhi::rt::State state;
        state.shaderTable = m_ShaderTable;
        state.bindings = { m_BindingSet };
        m_CommandList->setRayTracingState(state);

        nvrhi::rt::DispatchRaysArguments args;
        args.width = fbinfo.width;
        args.height = fbinfo.height;
        m_CommandList->dispatchRays(args);

        // Blit to framebuffer
        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, m_RenderTarget, m_BindingCache.get());

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        m_FrameIndex++;
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
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
    deviceParams.enableRayTracingExtensions = true;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }

    if (!deviceManager->GetDevice()->queryFeatureSupport(nvrhi::Feature::RayTracingPipeline))
    {
        log::fatal("The graphics device does not support Ray Tracing Pipelines");
        return 1;
    }

    // Get scene path from command line or use default
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
        log::info("Usage: rt_scene <scene.xml>");
        log::info("No scene file specified. Please provide a Mitsuba scene XML file.");
        
        // Try to use a default path for testing
        scenePath = "E:/SW/CG/mitsuba3/scenes/bathroom2/bathroom2/scene.xml";
        log::info("Trying default path: %s", scenePath.string().c_str());
    }

    if (!std::filesystem::exists(scenePath))
    {
        log::fatal("Scene file does not exist: %s", scenePath.string().c_str());
        return 1;
    }

    {
        RayTracedScene example(deviceManager, scenePath);
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
