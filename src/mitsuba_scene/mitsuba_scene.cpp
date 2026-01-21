#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/Camera.h>
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

static const char* g_WindowTitle = "Mitsuba Scene Rasterizer";

// ============================================================================
// GPU Structures (must match HLSL)
// ============================================================================
struct GPUVertex
{
    float3 position;
    float3 normal;
    float2 texcoord;
};

struct PerObjectConstants
{
    float4x4 worldViewProj;
    float4x4 world;
    float3 baseColor;
    float roughness;
    float3 emission;
    uint32_t isEmitter;
};

struct LightConstants
{
    float3 lightDir;
    float pad0;
    float3 lightColor;
    float pad1;
    float3 ambientColor;
    float pad2;
};

// ============================================================================
// Mitsuba Scene Parser (same as rt_scene)
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
        float3 baseColor = float3(0.5f);
        float roughness = 0.5f;
    };

    struct Shape
    {
        std::string type;
        std::string filename;
        std::string materialRef;
        float4x4 transform = float4x4::identity();
        bool isEmitter = false;
        float3 emission = float3(0.0f);
        
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
    float4x4 ParseMatrix(const std::string& matrixStr)
    {
        std::istringstream iss(matrixStr);
        float values[16];
        for (int i = 0; i < 16; i++)
        {
            iss >> values[i];
        }
        
        return float4x4(
            values[0], values[4], values[8], values[12],
            values[1], values[5], values[9], values[13],
            values[2], values[6], values[10], values[14],
            values[3], values[7], values[11], values[15]
        );
    }

    float3 ParseRGB(const std::string& rgbStr)
    {
        float3 color(0.0f);
        std::string cleaned = rgbStr;
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
                float3 color = ParseRGB(child.attribute("value").value());
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
    float3 baseColor;
    float roughness;
    float3 emission;
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

    // Depth buffer
    nvrhi::TextureHandle m_DepthTexture;
    nvrhi::FramebufferHandle m_Framebuffer;

    std::vector<RenderMesh> m_Meshes;
    
    MitsubaSceneParser m_SceneParser;
    std::filesystem::path m_ScenePath;
    
    // Camera control
    app::FirstPersonCamera m_Camera;
    float3 m_CameraPosition;
    bool m_CameraInitialized = false;

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

                float4 pos = float4(
                    attrib.vertices[3 * idx.v_idx + 0],
                    attrib.vertices[3 * idx.v_idx + 1],
                    attrib.vertices[3 * idx.v_idx + 2],
                    1.0f
                );
                pos = pos * shape.transform;
                vertex.position = float3(pos.x, pos.y, pos.z);

                if (idx.vn_idx >= 0 && attrib.normals)
                {
                    float4 normal = float4(
                        attrib.normals[3 * idx.vn_idx + 0],
                        attrib.normals[3 * idx.vn_idx + 1],
                        attrib.normals[3 * idx.vn_idx + 2],
                        0.0f
                    );
                    normal = normal * shape.transform;
                    vertex.normal = normalize(float3(normal.x, normal.y, normal.z));
                }
                else
                {
                    vertex.normal = float3(0.0f, 1.0f, 0.0f);
                }

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
            mesh.baseColor = float3(0.5f);
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

        std::vector<GPUVertex> vertices(4);
        for (int i = 0; i < 4; i++)
        {
            float4 pos = float4(positions[i], 1.0f);
            pos = pos * shape.transform;
            vertices[i].position = float3(pos.x, pos.y, pos.z);

            float4 n = float4(normal, 0.0f);
            n = n * shape.transform;
            vertices[i].normal = normalize(float3(n.x, n.y, n.z));

            vertices[i].texcoord = texcoords[i];
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

        if (shape.hasInlineMaterial)
        {
            mesh.baseColor = shape.inlineMaterial.baseColor;
            mesh.roughness = shape.inlineMaterial.roughness;
        }
        else
        {
            mesh.baseColor = float3(0.5f);
            mesh.roughness = 0.5f;
        }

        mesh.isEmitter = shape.isEmitter;
        mesh.emission = shape.emission;

        m_Meshes.push_back(mesh);
    }

    void InitializeCamera()
    {
        // Extract camera position from scene transform matrix
        float4x4& camTransform = m_SceneParser.camera.transform;
        m_CameraPosition = float3(camTransform.m03, camTransform.m13, camTransform.m23);
        
        // Extract forward direction
        float3 forward = -float3(camTransform.m02, camTransform.m12, camTransform.m22);
        
        m_Camera.SetMoveSpeed(10.0f);
        m_Camera.LookAt(m_CameraPosition, m_CameraPosition + forward, float3(0.0f, 1.0f, 0.0f));
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        m_Camera.KeyboardUpdate(key, scancode, action, mods);
        return true;
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        m_Camera.MousePosUpdate(xpos, ypos);
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        m_Camera.MouseButtonUpdate(button, action, mods);
        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        m_Camera.Animate(fElapsedTimeSeconds);
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    { 
        m_Pipeline = nullptr;
        m_DepthTexture = nullptr;
        m_Framebuffer = nullptr;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        const auto& fbinfo = framebuffer->getFramebufferInfo();

        // Create depth texture and custom framebuffer if needed
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

            // Create framebuffer with color from swapchain and our depth buffer
            nvrhi::FramebufferDesc fbDesc;
            fbDesc.addColorAttachment(framebuffer->getDesc().colorAttachments[0].texture);
            fbDesc.setDepthAttachment(m_DepthTexture);
            m_Framebuffer = GetDevice()->createFramebuffer(fbDesc);
        }

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

            m_Pipeline = GetDevice()->createGraphicsPipeline(psoDesc, m_Framebuffer->getFramebufferInfo());
        }

        // Calculate view/projection matrices
        float aspect = float(fbinfo.width) / float(fbinfo.height);
        float fovRadians = m_SceneParser.camera.fov * (3.14159265f / 180.0f);
        
        float4x4 view = affineToHomogeneous(m_Camera.GetWorldToViewMatrix());
        float4x4 proj = perspProjD3DStyle(fovRadians, aspect, 0.1f, 1000.0f);
        float4x4 viewProj = view * proj;

        m_CommandList->open();

        // Clear framebuffer
        nvrhi::utils::ClearColorAttachment(m_CommandList, m_Framebuffer, 0, nvrhi::Color(0.1f, 0.1f, 0.15f, 1.0f));
        
        // Clear depth
        m_CommandList->clearDepthStencilTexture(m_DepthTexture, 
            nvrhi::AllSubresources, true, 1.0f, false, 0);

        // Update light constants
        LightConstants lightConstants;
        lightConstants.lightDir = normalize(float3(0.5f, 1.0f, 0.3f));
        lightConstants.lightColor = float3(1.0f, 0.98f, 0.95f);
        lightConstants.ambientColor = float3(0.15f, 0.15f, 0.2f);
        m_CommandList->writeBuffer(m_LightBuffer, &lightConstants, sizeof(LightConstants));

        // Create binding set
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc.bindings = {
            nvrhi::BindingSetItem::ConstantBuffer(0, m_PerObjectBuffer),
            nvrhi::BindingSetItem::ConstantBuffer(1, m_LightBuffer)
        };
        nvrhi::BindingSetHandle bindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);

        // Render each mesh
        for (auto& mesh : m_Meshes)
        {
            // Update per-object constants
            PerObjectConstants perObject;
            perObject.worldViewProj = viewProj;  // Already transformed vertices
            perObject.world = float4x4::identity();
            perObject.baseColor = mesh.baseColor;
            perObject.roughness = mesh.roughness;
            perObject.emission = mesh.emission;
            perObject.isEmitter = mesh.isEmitter ? 1 : 0;
            m_CommandList->writeBuffer(m_PerObjectBuffer, &perObject, sizeof(PerObjectConstants));

            nvrhi::GraphicsState state;
            state.pipeline = m_Pipeline;
            state.framebuffer = m_Framebuffer;
            state.bindings = { bindingSet };
            state.vertexBuffers = { { mesh.vertexBuffer, 0, 0 } };
            state.indexBuffer = { mesh.indexBuffer, nvrhi::Format::R32_UINT, 0 };
            state.viewport.addViewportAndScissorRect(m_Framebuffer->getFramebufferInfo().getViewport());

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
