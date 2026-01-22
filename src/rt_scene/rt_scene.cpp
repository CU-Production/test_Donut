
#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/app/imgui_renderer.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>
#include <imgui.h>

// DLSS support
#if DONUT_WITH_DLSS
#include <donut/render/DLSS.h>
#include <donut/engine/View.h>
#include <donut/core/math/math.h>
namespace dm = donut::math;
#endif

#include <pugixml.hpp>

#define HANDMADE_MATH_USE_RADIANS
#include "HandmadeMath.h"

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include <tinyobj_loader_c.h>

// Texture loading utilities
#include "../common/texture_utils.h"

#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <cmath>

using namespace donut;

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
    RoughPlastic = 6,
    ThinDielectric = 7,
    Principled = 8,
    Blend = 9,
    Mask = 10,
    Null = 11
};

// ============================================================================
// GPU Structures (must match HLSL) - using plain floats for GPU compatibility
// ============================================================================
struct GPUMaterial
{
    float baseColor[3];
    float roughness;
    
    float eta[3];          // For conductors: complex IOR real part
    float metallic;
    
    float k[3];            // For conductors: complex IOR imaginary part
    uint32_t type;
    
    float intIOR;          // Interior index of refraction
    float extIOR;          // Exterior index of refraction
    int32_t baseColorTexIdx;   // -1 if no texture
    int32_t roughnessTexIdx;   // -1 if no texture
    
    int32_t normalTexIdx;      // -1 if no texture
    
    // Principled BSDF parameters
    float specular;        // Specular intensity
    float specTint;        // Tint specular towards base color
    float sheen;           // Sheen intensity
    
    float sheenTint;       // Tint sheen towards base color
    float clearcoat;       // Clearcoat intensity
    float clearcoatGloss;  // Clearcoat glossiness
    float specTrans;       // Specular transmission
    
    // Mask/Blend parameters
    float opacity;         // Opacity for mask material
    float blendWeight;     // Blend weight for blendbsdf
    float nonlinear;       // Nonlinear mode for plastic (0 or 1)
    float padding;
};

struct GPUVertex
{
    float position[3];
    float pad0;
    float normal[3];
    float pad1;
    float texcoord[2];
    float pad2[2];
};

struct GPUInstance
{
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t materialIndex;
    uint32_t isEmitter;
    float emission[3];
    float pad;
};

struct CameraConstants
{
    float viewInverse[16];   // column-major 4x4 matrix
    float projInverse[16];   // column-major 4x4 matrix
    float cameraPosition[3];
    uint32_t frameIndex;
    uint32_t samplesPerPixel;
    uint32_t maxBounces;
    float envMapIntensity;
    uint32_t hasEnvMap;
    float exposure;          // Exposure control for tone mapping
    float padding[3];        // Padding to align to 16 bytes
};

// ============================================================================
// Mitsuba Scene Parser - using HandmadeMath
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

    // Texture reference structure
    struct TextureRef
    {
        std::string filename;
        bool isValid = false;
        int textureIndex = -1;  // Index into loaded textures array
    };

    struct Material
    {
        std::string id;
        MaterialType type = MaterialType::Diffuse;
        HMM_Vec3 baseColor = HMM_V3(0.5f, 0.5f, 0.5f);  // Mitsuba default: 0.5
        float roughness = 0.1f;   // Mitsuba default alpha: 0.1 (NOT 0.5!)
        HMM_Vec3 eta = HMM_V3(1.0f, 1.0f, 1.0f);
        HMM_Vec3 k = HMM_V3(0.0f, 0.0f, 0.0f);
        float intIOR = 1.5046f;   // Mitsuba default: bk7 (1.5046)
        float extIOR = 1.000277f; // Mitsuba default: air (1.000277)
        float metallic = 0.0f;
        
        // Principled BSDF parameters
        float specular = 0.5f;
        float specTint = 0.0f;
        float sheen = 0.0f;
        float sheenTint = 0.0f;
        float clearcoat = 0.0f;
        float clearcoatGloss = 0.0f;
        float specTrans = 0.0f;
        
        // Mask/Blend parameters
        float opacity = 1.0f;
        float blendWeight = 0.5f;
        
        // Plastic-specific parameters
        bool nonlinear = false;  // Mitsuba default: false (preserve texture colors)
        
        // Texture references
        TextureRef baseColorTexture;
        TextureRef roughnessTexture;
        TextureRef normalTexture;
    };
    
    // Environment map structure
    struct EnvironmentMapInfo
    {
        std::string filename;
        float intensity = 1.0f;
        bool isValid = false;
    };

    struct Shape
    {
        std::string type;           // "obj" or "rectangle"
        std::string filename;       // OBJ filename
        std::string materialRef;    // Reference to material ID
        HMM_Mat4 transform = HMM_M4D(1.0f);  // Identity matrix
        bool isEmitter = false;
        HMM_Vec3 emission = HMM_V3(0.0f, 0.0f, 0.0f);
        
        // For inline materials
        Material inlineMaterial;
        bool hasInlineMaterial = false;
    };

    Camera camera;
    std::unordered_map<std::string, Material> materials;
    std::vector<Shape> shapes;
    std::filesystem::path sceneDirectory;
    
    // Environment map
    EnvironmentMapInfo environmentMap;
    
    // Loaded textures (indexed by material texture references)
    std::unordered_map<std::string, int> textureIndexMap;
    std::vector<texture_utils::TextureData> loadedTextures;

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
            else if (nodeName == "emitter")
            {
                ParseEmitter(node);
            }
            else if (nodeName == "texture")
            {
                ParseTextureDefinition(node);
            }
        }

        // Load all referenced textures
        LoadReferencedTextures();

        log::info("Parsed %zu materials, %zu shapes, %zu textures", 
            materials.size(), shapes.size(), loadedTextures.size());
        
        // Debug: print material types
        for (auto& [id, mat] : materials)
        {
            const char* typeNames[] = {
                "Diffuse", "Conductor", "RoughConductor", "Dielectric", "RoughDielectric", 
                "Plastic", "RoughPlastic", "ThinDielectric", "Principled", "Blend", "Mask", "Null"
            };
            uint32_t typeIdx = static_cast<uint32_t>(mat.type);
            const char* typeName = (typeIdx < 12) ? typeNames[typeIdx] : "Unknown";
            log::info("  Material '%s': type=%s, roughness=%.3f, baseColor=(%.2f,%.2f,%.2f), intIOR=%.2f, extIOR=%.2f, texIdx=%d, nonlinear=%s",
                id.c_str(), typeName, mat.roughness, 
                mat.baseColor.X, mat.baseColor.Y, mat.baseColor.Z,
                mat.intIOR, mat.extIOR,
                mat.baseColorTexture.textureIndex,
                mat.nonlinear ? "true" : "false");
        }
        if (environmentMap.isValid)
        {
            log::info("Environment map: %s (intensity: %.2f)", 
                environmentMap.filename.c_str(), environmentMap.intensity);
        }
        return true;
    }

private:
    // Parse a 4x4 matrix from Mitsuba format to HMM column-major storage
    // Mitsuba XML text order: m00 m01 m02 m03  m10 m11 m12 m13  m20 m21 m22 m23  m30 m31 m32 m33
    // HMM stores column-major: Columns[j] contains (m0j, m1j, m2j, m3j)
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

    // Parse RGB color from "r, g, b" format
    HMM_Vec3 ParseRGB(const std::string& rgbStr)
    {
        HMM_Vec3 color = HMM_V3(0.0f, 0.0f, 0.0f);
        std::string cleaned = rgbStr;
        // Remove commas
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
            mat.roughness = 1.0f;  // Lambertian diffuse - roughness doesn't apply
            mat.baseColor = HMM_V3(0.5f, 0.5f, 0.5f);  // Mitsuba default reflectance
        }
        else if (type == "conductor")
        {
            mat.type = MaterialType::Conductor;
            mat.roughness = 0.0f;
            mat.baseColor = HMM_V3(1.0f, 1.0f, 1.0f);
        }
        else if (type == "roughconductor")
        {
            mat.type = MaterialType::RoughConductor;
            mat.roughness = 0.1f;  // Mitsuba default alpha
            mat.baseColor = HMM_V3(1.0f, 1.0f, 1.0f);  // Default specular reflectance
        }
        else if (type == "dielectric")
        {
            mat.type = MaterialType::Dielectric;
            mat.roughness = 0.0f;
        }
        else if (type == "roughdielectric")
        {
            mat.type = MaterialType::RoughDielectric;
            mat.roughness = 0.1f;  // Mitsuba default alpha
        }
        else if (type == "plastic")
        {
            mat.type = MaterialType::Plastic;
            mat.roughness = 0.0f;  // Smooth plastic
            mat.intIOR = 1.49f;    // Mitsuba default: polypropylene
        }
        else if (type == "roughplastic")
        {
            mat.type = MaterialType::RoughPlastic;
            mat.roughness = 0.1f;  // Mitsuba default alpha
            mat.intIOR = 1.49f;    // Mitsuba default: polypropylene
        }
        else if (type == "thindielectric")
        {
            mat.type = MaterialType::ThinDielectric;
            mat.roughness = 0.0f;
        }
        else if (type == "principled")
        {
            mat.type = MaterialType::Principled;
            mat.specular = 0.5f;  // Default specular
        }
        else if (type == "blendbsdf")
        {
            mat.type = MaterialType::Blend;
            mat.blendWeight = 0.5f;
        }
        else if (type == "mask")
        {
            mat.type = MaterialType::Mask;
            mat.opacity = 0.5f;
        }
        else if (type == "null")
        {
            mat.type = MaterialType::Null;
        }

        // Parse material properties
        for (pugi::xml_node child : bsdfNode.children())
        {
            std::string childName = child.name();
            std::string propName = child.attribute("name").value();

            if (childName == "rgb" || childName == "spectrum")
            {
                HMM_Vec3 color = ParseRGB(child.attribute("value").value());
                if (propName == "reflectance" || propName == "diffuse_reflectance" || 
                    propName == "specular_reflectance" || propName == "base_color")
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
                    // Mitsuba's alpha is the GGX roughness directly
                    // Our shader squares roughness to get alpha, so we take sqrt here
                    // to get the correct final alpha value
                    mat.roughness = sqrtf(value);
                }
                else if (propName == "roughness")
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
                else if (propName == "eta")
                {
                    // Scalar eta for dielectrics
                    mat.intIOR = value;
                }
                // Principled BSDF parameters
                else if (propName == "metallic")
                {
                    mat.metallic = value;
                }
                else if (propName == "specular")
                {
                    mat.specular = value;
                }
                else if (propName == "spec_tint")
                {
                    mat.specTint = value;
                }
                else if (propName == "sheen")
                {
                    mat.sheen = value;
                }
                else if (propName == "sheen_tint")
                {
                    mat.sheenTint = value;
                }
                else if (propName == "clearcoat")
                {
                    mat.clearcoat = value;
                }
                else if (propName == "clearcoat_gloss")
                {
                    mat.clearcoatGloss = value;
                }
                else if (propName == "spec_trans")
                {
                    mat.specTrans = value;
                }
                // Mask/Blend parameters
                else if (propName == "opacity")
                {
                    mat.opacity = value;
                }
                else if (propName == "weight")
                {
                    mat.blendWeight = value;
                }
            }
            else if (childName == "string")
            {
                std::string value = child.attribute("value").value();
                if (propName == "material")
                {
                    // Mitsuba conductor material presets
                    // Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
                    if (value == "none")
                    {
                        // Perfect mirror - 100% reflective
                        mat.eta = HMM_V3(0.0f, 0.0f, 0.0f);
                        mat.k = HMM_V3(0.0f, 0.0f, 0.0f);
                    }
                    else if (value == "Ag" || value == "silver")
                    {
                        mat.eta = HMM_V3(0.155f, 0.117f, 0.138f);
                        mat.k = HMM_V3(4.827f, 3.122f, 2.147f);
                    }
                    else if (value == "Au" || value == "gold")
                    {
                        mat.eta = HMM_V3(0.143f, 0.374f, 1.442f);
                        mat.k = HMM_V3(3.983f, 2.387f, 1.603f);
                    }
                    else if (value == "Cu" || value == "copper")
                    {
                        mat.eta = HMM_V3(0.200f, 0.924f, 1.102f);
                        mat.k = HMM_V3(3.912f, 2.452f, 2.142f);
                    }
                    else if (value == "Al" || value == "aluminium" || value == "aluminum")
                    {
                        mat.eta = HMM_V3(1.657f, 0.880f, 0.521f);
                        mat.k = HMM_V3(9.224f, 6.269f, 4.837f);
                    }
                    else if (value == "Cr" || value == "chromium")
                    {
                        mat.eta = HMM_V3(3.180f, 3.180f, 2.010f);
                        mat.k = HMM_V3(3.300f, 3.330f, 3.040f);
                    }
                    else if (value == "Ni" || value == "nickel")
                    {
                        mat.eta = HMM_V3(1.970f, 1.860f, 1.670f);
                        mat.k = HMM_V3(3.740f, 3.060f, 2.580f);
                    }
                    else if (value == "Ti" || value == "titanium")
                    {
                        mat.eta = HMM_V3(2.160f, 1.970f, 1.810f);
                        mat.k = HMM_V3(2.930f, 2.620f, 2.350f);
                    }
                    else if (value == "W" || value == "tungsten")
                    {
                        mat.eta = HMM_V3(4.350f, 3.400f, 2.850f);
                        mat.k = HMM_V3(3.400f, 2.700f, 2.150f);
                    }
                    else if (value == "Fe" || value == "iron")
                    {
                        mat.eta = HMM_V3(2.950f, 2.930f, 2.650f);
                        mat.k = HMM_V3(3.000f, 2.950f, 2.800f);
                    }
                    // Add more presets as needed
                }
                // Mitsuba dielectric IOR presets (for int_ior / ext_ior)
                else if (propName == "int_ior" || propName == "ext_ior")
                {
                    float ior = 1.0f;
                    // Mitsuba IOR preset table
                    if (value == "vacuum")              ior = 1.0f;
                    else if (value == "helium")         ior = 1.00004f;
                    else if (value == "hydrogen")       ior = 1.00013f;
                    else if (value == "air")            ior = 1.000277f;
                    else if (value == "carbon dioxide") ior = 1.00045f;
                    else if (value == "water")          ior = 1.333f;
                    else if (value == "acetone")        ior = 1.36f;
                    else if (value == "ethanol")        ior = 1.361f;
                    else if (value == "carbon tetrachloride") ior = 1.461f;
                    else if (value == "glycerol")       ior = 1.4729f;
                    else if (value == "benzene")        ior = 1.501f;
                    else if (value == "silicone oil")   ior = 1.52045f;
                    else if (value == "bromine")        ior = 1.661f;
                    else if (value == "water ice")      ior = 1.31f;
                    else if (value == "fused quartz")   ior = 1.458f;
                    else if (value == "pyrex")          ior = 1.470f;
                    else if (value == "acrylic glass")  ior = 1.49f;
                    else if (value == "polypropylene")  ior = 1.49f;
                    else if (value == "bk7")            ior = 1.5046f;
                    else if (value == "sodium chloride") ior = 1.544f;
                    else if (value == "amber")          ior = 1.55f;
                    else if (value == "pet")            ior = 1.575f;
                    else if (value == "diamond")        ior = 2.419f;
                    
                    if (propName == "int_ior") mat.intIOR = ior;
                    else mat.extIOR = ior;
                }
            }
            else if (childName == "texture")
            {
                // Parse texture reference
                TextureRef texRef = ParseTextureRef(child);
                if (texRef.isValid)
                {
                    if (propName == "reflectance" || propName == "diffuse_reflectance")
                    {
                        mat.baseColorTexture = texRef;
                    }
                    else if (propName == "alpha" || propName == "roughness")
                    {
                        mat.roughnessTexture = texRef;
                    }
                }
            }
            else if (childName == "boolean")
            {
                std::string value = child.attribute("value").value();
                bool boolValue = (value == "true" || value == "1");
                if (propName == "nonlinear")
                {
                    mat.nonlinear = boolValue;
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

    // Parse global emitter (environment map)
    void ParseEmitter(pugi::xml_node emitterNode)
    {
        std::string type = emitterNode.attribute("type").value();
        
        // Environment map emitter
        if (type == "envmap")
        {
            for (pugi::xml_node child : emitterNode.children("string"))
            {
                std::string name = child.attribute("name").value();
                if (name == "filename")
                {
                    environmentMap.filename = child.attribute("value").value();
                    environmentMap.isValid = true;
                }
            }
            
            for (pugi::xml_node child : emitterNode.children("float"))
            {
                std::string name = child.attribute("name").value();
                if (name == "scale")
                {
                    environmentMap.intensity = child.attribute("value").as_float(1.0f);
                }
            }
            
            // Also check for intensity in rgb format
            for (pugi::xml_node child : emitterNode.children("rgb"))
            {
                std::string name = child.attribute("name").value();
                if (name == "scale")
                {
                    HMM_Vec3 scale = ParseRGB(child.attribute("value").value());
                    environmentMap.intensity = (scale.X + scale.Y + scale.Z) / 3.0f;
                }
            }
            
            log::info("Found environment map: %s", environmentMap.filename.c_str());
        }
        // Constant environment
        else if (type == "constant")
        {
            // Could be extended to support constant environment color
        }
    }
    
    // Parse standalone texture definition
    void ParseTextureDefinition(pugi::xml_node textureNode)
    {
        std::string id = textureNode.attribute("id").value();
        std::string type = textureNode.attribute("type").value();
        
        if (type == "bitmap")
        {
            for (pugi::xml_node child : textureNode.children("string"))
            {
                std::string name = child.attribute("name").value();
                if (name == "filename")
                {
                    std::string filename = child.attribute("value").value();
                    // Store for later loading
                    textureIndexMap[id] = -1;  // Will be updated when loaded
                    log::info("Found texture definition: %s -> %s", id.c_str(), filename.c_str());
                }
            }
        }
    }
    
    // Parse texture reference in BSDF
    TextureRef ParseTextureRef(pugi::xml_node textureNode)
    {
        TextureRef ref;
        std::string type = textureNode.attribute("type").value();
        
        if (type == "bitmap")
        {
            for (pugi::xml_node child : textureNode.children("string"))
            {
                std::string name = child.attribute("name").value();
                if (name == "filename")
                {
                    ref.filename = child.attribute("value").value();
                    ref.isValid = true;
                }
            }
        }
        else if (type == "ref")
        {
            // Reference to a standalone texture definition
            std::string refId = textureNode.attribute("id").value();
            if (textureIndexMap.find(refId) != textureIndexMap.end())
            {
                ref.isValid = true;
                // Filename will be resolved later
            }
        }
        
        return ref;
    }
    
    // Load all referenced textures
    void LoadReferencedTextures()
    {
        std::unordered_set<std::string> textureFiles;
        
        // Collect texture filenames from materials
        for (auto& [id, mat] : materials)
        {
            if (mat.baseColorTexture.isValid && !mat.baseColorTexture.filename.empty())
            {
                textureFiles.insert(mat.baseColorTexture.filename);
            }
            if (mat.roughnessTexture.isValid && !mat.roughnessTexture.filename.empty())
            {
                textureFiles.insert(mat.roughnessTexture.filename);
            }
            if (mat.normalTexture.isValid && !mat.normalTexture.filename.empty())
            {
                textureFiles.insert(mat.normalTexture.filename);
            }
        }
        
        // Load each unique texture
        for (const auto& filename : textureFiles)
        {
            std::filesystem::path texturePath = sceneDirectory / filename;
            texture_utils::TextureData texData = texture_utils::LoadTexture(texturePath);
            
            if (texData.IsValid())
            {
                int index = static_cast<int>(loadedTextures.size());
                textureIndexMap[filename] = index;
                loadedTextures.push_back(std::move(texData));
                log::info("Loaded texture [%d]: %s (%dx%d)", index, filename.c_str(), texData.width, texData.height);
            }
            else
            {
                log::error("Failed to load texture: %s", texturePath.string().c_str());
            }
        }
        
        // Update material texture indices
        for (auto& [id, mat] : materials)
        {
            if (mat.baseColorTexture.isValid && !mat.baseColorTexture.filename.empty())
            {
                auto it = textureIndexMap.find(mat.baseColorTexture.filename);
                if (it != textureIndexMap.end())
                {
                    mat.baseColorTexture.textureIndex = it->second;
                }
            }
            if (mat.roughnessTexture.isValid && !mat.roughnessTexture.filename.empty())
            {
                auto it = textureIndexMap.find(mat.roughnessTexture.filename);
                if (it != textureIndexMap.end())
                {
                    mat.roughnessTexture.textureIndex = it->second;
                }
            }
            if (mat.normalTexture.isValid && !mat.normalTexture.filename.empty())
            {
                auto it = textureIndexMap.find(mat.normalTexture.filename);
                if (it != textureIndexMap.end())
                {
                    mat.normalTexture.textureIndex = it->second;
                }
            }
        }
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
    
    // G-buffer textures for DLSS Ray Reconstruction
    nvrhi::TextureHandle m_DepthBuffer;
    nvrhi::TextureHandle m_MotionVectors;
    nvrhi::TextureHandle m_DiffuseAlbedo;
    nvrhi::TextureHandle m_SpecularAlbedo;
    nvrhi::TextureHandle m_NormalRoughness;
    nvrhi::TextureHandle m_DLSSOutput;
    
#if DONUT_WITH_DLSS
    // DLSS
    std::unique_ptr<render::DLSS> m_DLSS;
    bool m_DLSSEnabled = false;
    bool m_DLSSAvailable = false;
#endif
    
    // Textures
    nvrhi::TextureHandle m_EnvironmentMap;
    nvrhi::TextureHandle m_DefaultMaterialTexture;  // 1x1 white texture for empty slots
    std::vector<nvrhi::TextureHandle> m_MaterialTextures;
    nvrhi::SamplerHandle m_LinearSampler;
    
    static constexpr int MAX_MATERIAL_TEXTURES = 64;

    // Render passes
    std::shared_ptr<engine::CommonRenderPasses> m_CommonPasses;
    std::unique_ptr<engine::BindingCache> m_BindingCache;

    // Scene data
    MitsubaSceneParser m_SceneParser;
    std::vector<GPUVertex> m_Vertices;
    std::vector<uint32_t> m_Indices;
    std::vector<GPUMaterial> m_Materials;
    std::vector<GPUInstance> m_Instances;

    // Camera (using HMM)
    CameraConstants m_CameraConstants;
    HMM_Vec3 m_CameraPosition;
    HMM_Vec3 m_CameraTarget;
    HMM_Vec3 m_CameraUp;
    float m_CameraYaw = 0.0f;
    float m_CameraPitch = 0.0f;
    float m_CameraSpeed = 10.0f;
    uint32_t m_FrameIndex = 0;
    
    // Mouse state
    bool m_MouseDown = false;
    float m_LastMouseX = 0.0f;
    float m_LastMouseY = 0.0f;
    
    // Keyboard state
    bool m_KeyW = false, m_KeyS = false, m_KeyA = false, m_KeyD = false;
    bool m_KeyQ = false, m_KeyE = false;
    
    // Rendering parameters (adjustable via ImGui)
    float m_Exposure = 0.015f;
    
public:
    // Accessors for UI
    float& GetExposure() { return m_Exposure; }
    uint32_t& GetMaxBounces() { return m_CameraConstants.maxBounces; }
    uint32_t GetFrameIndex() const { return m_FrameIndex; }
    void ResetAccumulation() { m_FrameIndex = 0; }
    
#if DONUT_WITH_DLSS
    bool& GetDLSSEnabled() { return m_DLSSEnabled; }
    bool IsDLSSAvailable() const { return m_DLSSAvailable; }
#endif
    
private:

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

        // Create sampler for textures
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.setAllFilters(true);
        samplerDesc.setAllAddressModes(nvrhi::SamplerAddressMode::Wrap);
        m_LinearSampler = GetDevice()->createSampler(samplerDesc);

        // Create binding layout
        nvrhi::BindingLayoutDesc bindingLayoutDesc;
        bindingLayoutDesc.visibility = nvrhi::ShaderType::All;
        bindingLayoutDesc.bindings = {
            nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),     // t0: TLAS
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(1),      // t1: Vertices
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(2),      // t2: Indices
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(3),      // t3: Materials
            nvrhi::BindingLayoutItem::StructuredBuffer_SRV(4),      // t4: Instances
            nvrhi::BindingLayoutItem::Texture_SRV(5),               // t5: Environment map
            nvrhi::BindingLayoutItem::Texture_SRV(6).setSize(MAX_MATERIAL_TEXTURES),  // t6-t69: Material textures array
            nvrhi::BindingLayoutItem::Texture_UAV(0),               // u0: Output
            nvrhi::BindingLayoutItem::Texture_UAV(1),               // u1: Accumulation
            nvrhi::BindingLayoutItem::Texture_UAV(2),               // u2: Depth buffer (for DLSS)
            nvrhi::BindingLayoutItem::Texture_UAV(3),               // u3: Motion vectors (for DLSS)
            nvrhi::BindingLayoutItem::Texture_UAV(4),               // u4: Diffuse albedo (for DLSS RR)
            nvrhi::BindingLayoutItem::Texture_UAV(5),               // u5: Specular albedo (for DLSS RR)
            nvrhi::BindingLayoutItem::Texture_UAV(6),               // u6: Normal + Roughness (for DLSS RR)
            nvrhi::BindingLayoutItem::ConstantBuffer(0),            // b0: Camera
            nvrhi::BindingLayoutItem::Sampler(0)                    // s0: Linear sampler
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

        // HitInfo struct size: float3 color (12) + float hitT (4) + float3 emission (12) + 
        // uint instanceID (4) + float2 texcoord (8) + float2 padding (8) = 48 bytes
        pipelineDesc.maxPayloadSize = 48;
        pipelineDesc.maxRecursionDepth = 2;  // We only do iterative tracing, not recursive

        m_Pipeline = GetDevice()->createRayTracingPipeline(pipelineDesc);

        m_ShaderTable = m_Pipeline->createShaderTable();
        m_ShaderTable->setRayGenerationShader("RayGen");
        m_ShaderTable->addHitGroup("HitGroup");
        m_ShaderTable->addHitGroup("ShadowHitGroup");
        m_ShaderTable->addMissShader("Miss");
        m_ShaderTable->addMissShader("ShadowMiss");

        m_CommandList = GetDevice()->createCommandList();

        // Create textures (needs command list)
        CreateEnvironmentMapTexture();
        CreateMaterialTextures();

        // Create GPU buffers and acceleration structures
        CreateGPUResources();

        // Setup camera from scene
        SetupCameraFromScene();

#if DONUT_WITH_DLSS
        // Initialize DLSS - note: this only creates the DLSS object, Init() is called later with resolution
        // This matches FeatureDemo.cpp: DLSS doesn't need to be re-created when we reload shaders
        std::string executableDir = app::GetDirectoryWithExecutable().generic_string();
        log::info("Initializing DLSS from directory: %s", executableDir.c_str());
        
        m_DLSS = render::DLSS::Create(GetDevice(), *shaderFactory, 
            executableDir, render::DLSS::DefaultApplicationID);
        
        if (m_DLSS)
        {
            // Check DLSS availability - IsDlssSupported() checks if NGX was initialized successfully
            // and if DLSS feature is available on this system
            if (m_DLSS->IsDlssSupported())
            {
                m_DLSSAvailable = true;
                if (m_DLSS->IsRayReconstructionSupported())
                {
                    log::info("DLSS Ray Reconstruction is available");
                }
                else
                {
                    log::info("DLSS is available (Ray Reconstruction not supported)");
                }
            }
            else
            {
                log::warning("DLSS is not available on this system");
                log::warning("This may be due to:");
                log::warning("  1. NGX DLL not found or failed to load (check RuntimeLibrary mismatch)");
                log::warning("  2. No NVIDIA RTX GPU detected");
                log::warning("  3. DLSS driver version too old");
                log::warning("  4. NGX initialization failed");
                m_DLSSAvailable = false;
            }
        }
        else
        {
            log::warning("Failed to create DLSS instance - DLSS::Create() returned nullptr");
            log::warning("This usually means the graphics API is not supported or NGX failed to initialize");
            m_DLSSAvailable = false;
        }
#endif

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
            GPUMaterial gpuMat = {};  // Zero-initialize
            gpuMat.baseColor[0] = mat.baseColor.X;
            gpuMat.baseColor[1] = mat.baseColor.Y;
            gpuMat.baseColor[2] = mat.baseColor.Z;
            gpuMat.roughness = mat.roughness;
            gpuMat.eta[0] = mat.eta.X;
            gpuMat.eta[1] = mat.eta.Y;
            gpuMat.eta[2] = mat.eta.Z;
            gpuMat.k[0] = mat.k.X;
            gpuMat.k[1] = mat.k.Y;
            gpuMat.k[2] = mat.k.Z;
            gpuMat.type = static_cast<uint32_t>(mat.type);
            gpuMat.intIOR = mat.intIOR;
            gpuMat.extIOR = mat.extIOR;
            gpuMat.metallic = mat.metallic;
            if (mat.type == MaterialType::Conductor || mat.type == MaterialType::RoughConductor)
            {
                gpuMat.metallic = 1.0f;
            }
            gpuMat.baseColorTexIdx = mat.baseColorTexture.textureIndex;
            gpuMat.roughnessTexIdx = mat.roughnessTexture.textureIndex;
            gpuMat.normalTexIdx = mat.normalTexture.textureIndex;
            
            // Principled BSDF parameters
            gpuMat.specular = mat.specular;
            gpuMat.specTint = mat.specTint;
            gpuMat.sheen = mat.sheen;
            gpuMat.sheenTint = mat.sheenTint;
            gpuMat.clearcoat = mat.clearcoat;
            gpuMat.clearcoatGloss = mat.clearcoatGloss;
            gpuMat.specTrans = mat.specTrans;
            
            // Mask/Blend parameters
            gpuMat.opacity = mat.opacity;
            gpuMat.blendWeight = mat.blendWeight;
            gpuMat.nonlinear = mat.nonlinear ? 1.0f : 0.0f;
            gpuMat.padding = 0.0f;
            
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

                // Position - transform using HMM (column-vector: M * v)
                HMM_Vec4 pos = HMM_V4(
                    attrib.vertices[3 * idx.v_idx + 0],
                    attrib.vertices[3 * idx.v_idx + 1],
                    attrib.vertices[3 * idx.v_idx + 2],
                    1.0f
                );
                HMM_Vec4 worldPos = HMM_MulM4V4(shape.transform, pos);
                vertex.position[0] = worldPos.X;
                vertex.position[1] = worldPos.Y;
                vertex.position[2] = worldPos.Z;

                // Normal - transform using upper 3x3 of world matrix
                if (idx.vn_idx >= 0 && attrib.normals)
                {
                    HMM_Vec4 normal = HMM_V4(
                        attrib.normals[3 * idx.vn_idx + 0],
                        attrib.normals[3 * idx.vn_idx + 1],
                        attrib.normals[3 * idx.vn_idx + 2],
                        0.0f
                    );
                    HMM_Vec4 worldNormal = HMM_MulM4V4(shape.transform, normal);
                    HMM_Vec3 n = HMM_NormV3(HMM_V3(worldNormal.X, worldNormal.Y, worldNormal.Z));
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

                // Texcoord
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
            GPUMaterial gpuMat = {};
            const auto& mat = shape.inlineMaterial;
            gpuMat.baseColor[0] = mat.baseColor.X;
            gpuMat.baseColor[1] = mat.baseColor.Y;
            gpuMat.baseColor[2] = mat.baseColor.Z;
            gpuMat.roughness = mat.roughness;
            gpuMat.eta[0] = mat.eta.X;
            gpuMat.eta[1] = mat.eta.Y;
            gpuMat.eta[2] = mat.eta.Z;
            gpuMat.k[0] = mat.k.X;
            gpuMat.k[1] = mat.k.Y;
            gpuMat.k[2] = mat.k.Z;
            gpuMat.type = static_cast<uint32_t>(mat.type);
            gpuMat.intIOR = mat.intIOR;
            gpuMat.extIOR = mat.extIOR;
            gpuMat.metallic = mat.metallic;
            if (mat.type == MaterialType::Conductor || mat.type == MaterialType::RoughConductor)
                gpuMat.metallic = 1.0f;
            gpuMat.baseColorTexIdx = mat.baseColorTexture.textureIndex;
            gpuMat.roughnessTexIdx = mat.roughnessTexture.textureIndex;
            gpuMat.normalTexIdx = mat.normalTexture.textureIndex;
            gpuMat.specular = mat.specular;
            gpuMat.specTint = mat.specTint;
            gpuMat.sheen = mat.sheen;
            gpuMat.sheenTint = mat.sheenTint;
            gpuMat.clearcoat = mat.clearcoat;
            gpuMat.clearcoatGloss = mat.clearcoatGloss;
            gpuMat.specTrans = mat.specTrans;
            gpuMat.opacity = mat.opacity;
            gpuMat.blendWeight = mat.blendWeight;
            gpuMat.nonlinear = mat.nonlinear ? 1.0f : 0.0f;
            gpuMat.padding = 0.0f;
            matIndex = static_cast<uint32_t>(m_Materials.size());
            m_Materials.push_back(gpuMat);
        }

        // Create instance
        GPUInstance instance;
        instance.vertexOffset = startVertexIndex;
        instance.indexOffset = startIndexOffset;
        instance.materialIndex = matIndex;
        instance.isEmitter = shape.isEmitter ? 1 : 0;
        instance.emission[0] = shape.emission.X;
        instance.emission[1] = shape.emission.Y;
        instance.emission[2] = shape.emission.Z;
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
        HMM_Vec4 positions[4] = {
            HMM_V4(-1.0f, -1.0f, 0.0f, 1.0f),
            HMM_V4( 1.0f, -1.0f, 0.0f, 1.0f),
            HMM_V4( 1.0f,  1.0f, 0.0f, 1.0f),
            HMM_V4(-1.0f,  1.0f, 0.0f, 1.0f)
        };

        HMM_Vec4 normal = HMM_V4(0.0f, 0.0f, 1.0f, 0.0f);
        float texcoords[4][2] = {
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f},
            {0.0f, 1.0f}
        };

        // Transform and add vertices
        for (int i = 0; i < 4; i++)
        {
            GPUVertex vertex;
            HMM_Vec4 worldPos = HMM_MulM4V4(shape.transform, positions[i]);
            vertex.position[0] = worldPos.X;
            vertex.position[1] = worldPos.Y;
            vertex.position[2] = worldPos.Z;

            HMM_Vec4 worldNormal = HMM_MulM4V4(shape.transform, normal);
            HMM_Vec3 n = HMM_NormV3(HMM_V3(worldNormal.X, worldNormal.Y, worldNormal.Z));
            vertex.normal[0] = n.X;
            vertex.normal[1] = n.Y;
            vertex.normal[2] = n.Z;

            vertex.texcoord[0] = texcoords[i][0];
            vertex.texcoord[1] = texcoords[i][1];
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
            GPUMaterial gpuMat = {};
            const auto& mat = shape.inlineMaterial;
            gpuMat.baseColor[0] = mat.baseColor.X;
            gpuMat.baseColor[1] = mat.baseColor.Y;
            gpuMat.baseColor[2] = mat.baseColor.Z;
            gpuMat.roughness = mat.roughness;
            gpuMat.eta[0] = mat.eta.X;
            gpuMat.eta[1] = mat.eta.Y;
            gpuMat.eta[2] = mat.eta.Z;
            gpuMat.k[0] = mat.k.X;
            gpuMat.k[1] = mat.k.Y;
            gpuMat.k[2] = mat.k.Z;
            gpuMat.type = static_cast<uint32_t>(mat.type);
            gpuMat.intIOR = mat.intIOR;
            gpuMat.extIOR = mat.extIOR;
            gpuMat.metallic = mat.metallic;
            if (mat.type == MaterialType::Conductor || mat.type == MaterialType::RoughConductor)
                gpuMat.metallic = 1.0f;
            gpuMat.baseColorTexIdx = mat.baseColorTexture.textureIndex;
            gpuMat.roughnessTexIdx = mat.roughnessTexture.textureIndex;
            gpuMat.normalTexIdx = mat.normalTexture.textureIndex;
            gpuMat.specular = mat.specular;
            gpuMat.specTint = mat.specTint;
            gpuMat.sheen = mat.sheen;
            gpuMat.sheenTint = mat.sheenTint;
            gpuMat.clearcoat = mat.clearcoat;
            gpuMat.clearcoatGloss = mat.clearcoatGloss;
            gpuMat.specTrans = mat.specTrans;
            gpuMat.opacity = mat.opacity;
            gpuMat.blendWeight = mat.blendWeight;
            gpuMat.nonlinear = mat.nonlinear ? 1.0f : 0.0f;
            gpuMat.padding = 0.0f;
            matIndex = static_cast<uint32_t>(m_Materials.size());
            m_Materials.push_back(gpuMat);
        }

        // Create instance
        GPUInstance instance;
        instance.vertexOffset = startVertexIndex;
        instance.indexOffset = startIndexOffset;
        instance.materialIndex = matIndex;
        instance.isEmitter = shape.isEmitter ? 1 : 0;
        instance.emission[0] = shape.emission.X;
        instance.emission[1] = shape.emission.Y;
        instance.emission[2] = shape.emission.Z;
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
    
    void CreateEnvironmentMapTexture()
    {
        // Load environment map if specified
        if (m_SceneParser.environmentMap.isValid)
        {
            std::filesystem::path envMapPath = m_SceneParser.sceneDirectory / m_SceneParser.environmentMap.filename;
            texture_utils::TextureData envMapData = texture_utils::LoadTexture(envMapPath);
            
            if (envMapData.IsValid())
            {
                nvrhi::TextureDesc textureDesc;
                textureDesc.width = envMapData.width;
                textureDesc.height = envMapData.height;
                textureDesc.format = nvrhi::Format::RGBA32_FLOAT;
                textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
                textureDesc.keepInitialState = true;
                textureDesc.debugName = "EnvironmentMap";
                
                m_EnvironmentMap = GetDevice()->createTexture(textureDesc);
                
                m_CommandList->open();
                m_CommandList->writeTexture(m_EnvironmentMap, 0, 0, 
                    envMapData.data.data(), envMapData.width * 4 * sizeof(float));
                m_CommandList->close();
                GetDevice()->executeCommandList(m_CommandList);
                
                log::info("Created environment map texture: %dx%d", envMapData.width, envMapData.height);
            }
        }
        
        // Create a default 1x1 black texture if no environment map
        if (!m_EnvironmentMap)
        {
            nvrhi::TextureDesc textureDesc;
            textureDesc.width = 1;
            textureDesc.height = 1;
            textureDesc.format = nvrhi::Format::RGBA32_FLOAT;
            textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            textureDesc.keepInitialState = true;
            textureDesc.debugName = "DefaultEnvironmentMap";
            
            m_EnvironmentMap = GetDevice()->createTexture(textureDesc);
            
            float black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            m_CommandList->open();
            m_CommandList->writeTexture(m_EnvironmentMap, 0, 0, black, sizeof(black));
            m_CommandList->close();
            GetDevice()->executeCommandList(m_CommandList);
        }
    }
    
    void CreateMaterialTextures()
    {
        // Create default 1x1 white texture for empty slots
        {
            nvrhi::TextureDesc textureDesc;
            textureDesc.width = 1;
            textureDesc.height = 1;
            textureDesc.format = nvrhi::Format::RGBA32_FLOAT;
            textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            textureDesc.keepInitialState = true;
            textureDesc.debugName = "DefaultMaterialTexture";
            
            m_DefaultMaterialTexture = GetDevice()->createTexture(textureDesc);
            
            float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
            m_CommandList->open();
            m_CommandList->writeTexture(m_DefaultMaterialTexture, 0, 0, white, sizeof(white));
            m_CommandList->close();
            GetDevice()->executeCommandList(m_CommandList);
        }
        
        // Create textures from loaded texture data
        for (const auto& texData : m_SceneParser.loadedTextures)
        {
            nvrhi::TextureDesc textureDesc;
            textureDesc.width = texData.width;
            textureDesc.height = texData.height;
            textureDesc.format = nvrhi::Format::RGBA32_FLOAT;
            textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            textureDesc.keepInitialState = true;
            textureDesc.debugName = texData.path.c_str();
            
            nvrhi::TextureHandle texture = GetDevice()->createTexture(textureDesc);
            
            m_CommandList->open();
            m_CommandList->writeTexture(texture, 0, 0, 
                texData.data.data(), texData.width * 4 * sizeof(float));
            m_CommandList->close();
            GetDevice()->executeCommandList(m_CommandList);
            
            m_MaterialTextures.push_back(texture);
        }
        
        if (!m_MaterialTextures.empty())
        {
            log::info("Created %zu material textures", m_MaterialTextures.size());
        }
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
            // Note: Indices are stored as global indices, so we use the entire vertex buffer
            // with no vertex offset. The indices directly reference the correct vertices.
            nvrhi::rt::AccelStructDesc blasDesc;
            blasDesc.isTopLevel = false;

            nvrhi::rt::GeometryDesc geometryDesc;
            auto& triangles = geometryDesc.geometryData.triangles;
            triangles.indexBuffer = m_IndexBuffer;
            triangles.indexOffset = instance.indexOffset * sizeof(uint32_t);
            triangles.indexFormat = nvrhi::Format::R32_UINT;
            triangles.indexCount = indexCount;
            triangles.vertexBuffer = m_VertexBuffer;
            triangles.vertexOffset = 0;  // Use global indices - no vertex offset
            triangles.vertexFormat = nvrhi::Format::RGB32_FLOAT;
            triangles.vertexStride = sizeof(GPUVertex);
            triangles.vertexCount = static_cast<uint32_t>(m_Vertices.size());  // Total vertex count
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
            // Identity transform for TLAS instance (row-major 3x4 matrix)
            float transform[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,  // row 0
                0.0f, 1.0f, 0.0f, 0.0f,  // row 1
                0.0f, 0.0f, 1.0f, 0.0f   // row 2
            };
            memcpy(instanceDesc.transform, transform, sizeof(transform));
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
        // Extract camera position from transform matrix
        // HMM column-major: Columns[3] = translation = (m03, m13, m23, m33)
        HMM_Mat4& camTransform = m_SceneParser.camera.transform;
        
        m_CameraPosition = HMM_V3(
            camTransform.Columns[3].X,
            camTransform.Columns[3].Y,
            camTransform.Columns[3].Z
        );
        
        // Forward direction: Column[2] is local Z axis in world space
        // For Mitsuba cameras, this is the view direction
        HMM_Vec3 forward = HMM_V3(
            camTransform.Columns[2].X,
            camTransform.Columns[2].Y,
            camTransform.Columns[2].Z
        );
        
        m_CameraTarget = HMM_AddV3(m_CameraPosition, forward);
        
        // Up vector: Column[1]
        m_CameraUp = HMM_V3(
            camTransform.Columns[1].X,
            camTransform.Columns[1].Y,
            camTransform.Columns[1].Z
        );
        
        // Calculate initial yaw and pitch from forward direction
        m_CameraPitch = asinf(-forward.Y);
        m_CameraYaw = atan2f(forward.X, forward.Z);
        
        // Initialize camera constants
        m_CameraConstants.samplesPerPixel = 1;
        m_CameraConstants.maxBounces = 16;  // Higher for multiple mirror reflections
        
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
            
            // Reset accumulation when camera moves
            m_FrameIndex = 0;
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
        bool moved = false;
        float speed = m_CameraSpeed * fElapsedTimeSeconds;
        if (m_KeyW) { m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(forward, speed)); moved = true; }
        if (m_KeyS) { m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(forward, -speed)); moved = true; }
        if (m_KeyA) { m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(right, -speed)); moved = true; }
        if (m_KeyD) { m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(right, speed)); moved = true; }
        if (m_KeyE) { m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(up, speed)); moved = true; }
        if (m_KeyQ) { m_CameraPosition = HMM_AddV3(m_CameraPosition, HMM_MulV3F(up, -speed)); moved = true; }
        
        // Reset accumulation when camera moves
        if (moved)
        {
            m_FrameIndex = 0;
        }
        
        m_CameraTarget = HMM_AddV3(m_CameraPosition, forward);
        
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void BackBufferResizing() override
    {
        m_RenderTarget = nullptr;
        m_AccumulationTarget = nullptr;
        m_DepthBuffer = nullptr;
        m_MotionVectors = nullptr;
        m_DiffuseAlbedo = nullptr;
        m_SpecularAlbedo = nullptr;
        m_NormalRoughness = nullptr;
        m_DLSSOutput = nullptr;
        m_BindingCache->Clear();
        m_FrameIndex = 0;
        
#if DONUT_WITH_DLSS
        // Re-initialize DLSS on resize
        if (m_DLSS && m_DLSS->IsDlssInitialized())
        {
            // DLSS will be re-initialized on next frame
        }
#endif
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
            
            // Create G-buffer textures for DLSS Ray Reconstruction
            textureDesc.format = nvrhi::Format::R32_FLOAT;
            textureDesc.debugName = "DepthBuffer";
            m_DepthBuffer = GetDevice()->createTexture(textureDesc);
            
            textureDesc.format = nvrhi::Format::RG16_FLOAT;
            textureDesc.debugName = "MotionVectors";
            m_MotionVectors = GetDevice()->createTexture(textureDesc);
            
            textureDesc.format = nvrhi::Format::RGBA16_FLOAT;
            textureDesc.debugName = "DiffuseAlbedo";
            m_DiffuseAlbedo = GetDevice()->createTexture(textureDesc);
            
            textureDesc.debugName = "SpecularAlbedo";
            m_SpecularAlbedo = GetDevice()->createTexture(textureDesc);
            
            textureDesc.debugName = "NormalRoughness";
            m_NormalRoughness = GetDevice()->createTexture(textureDesc);
            
            // DLSS output texture (same format as render target)
            textureDesc.format = nvrhi::Format::RGBA16_FLOAT;
            textureDesc.isUAV = false;
            textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
            textureDesc.debugName = "DLSSOutput";
            m_DLSSOutput = GetDevice()->createTexture(textureDesc);

#if DONUT_WITH_DLSS
            // Initialize DLSS with current resolution
            if (m_DLSS && m_DLSSAvailable)
            {
                render::DLSS::InitParameters dlssParams;
                dlssParams.inputWidth = fbinfo.width;
                dlssParams.inputHeight = fbinfo.height;
                dlssParams.outputWidth = fbinfo.width;
                dlssParams.outputHeight = fbinfo.height;
                dlssParams.useAutoExposure = true;
                // useLinearDepth = false for hardware depth (D3D12 depth format is non-linear)
                // This matches Donut's default behavior for standard depth buffers
                dlssParams.useLinearDepth = false;
                
                // Try with Ray Reconstruction first if supported, fallback to regular DLSS if it fails
                bool useRayReconstruction = m_DLSS->IsRayReconstructionSupported();
                dlssParams.useRayReconstruction = useRayReconstruction;
                
                log::info("Attempting DLSS Init at resolution %dx%d (RayReconstruction=%s)", 
                    fbinfo.width, fbinfo.height, useRayReconstruction ? "true" : "false");
                
                m_DLSS->Init(dlssParams);
                
                // Update availability based on actual initialization result
                // Note: When using Ray Reconstruction, check IsRayReconstructionInitialized() instead of IsDlssInitialized()
                bool initialized = useRayReconstruction 
                    ? m_DLSS->IsRayReconstructionInitialized() 
                    : m_DLSS->IsDlssInitialized();
                
                if (initialized)
                {
                    m_DLSSAvailable = true;
                    if (useRayReconstruction)
                    {
                        log::info("DLSS Ray Reconstruction initialized successfully at resolution %dx%d", 
                            fbinfo.width, fbinfo.height);
                    }
                    else
                    {
                        log::info("DLSS initialized successfully at resolution %dx%d", 
                            fbinfo.width, fbinfo.height);
                    }
                }
                else
                {
                    // If Ray Reconstruction failed, try regular DLSS
                    if (useRayReconstruction)
                    {
                        log::warning("DLSS Ray Reconstruction Init() failed, trying regular DLSS...");
                        dlssParams.useRayReconstruction = false;
                        m_DLSS->Init(dlssParams);
                        
                        if (m_DLSS->IsDlssInitialized())
                        {
                            m_DLSSAvailable = true;
                            log::info("DLSS (without Ray Reconstruction) initialized successfully at resolution %dx%d", 
                                fbinfo.width, fbinfo.height);
                        }
                        else
                        {
                            m_DLSSAvailable = false;
                            log::warning("DLSS Init() failed even without Ray Reconstruction");
                            log::warning("Resolution: %dx%d, useLinearDepth: %s, useAutoExposure: %s", 
                                fbinfo.width, fbinfo.height, 
                                dlssParams.useLinearDepth ? "true" : "false",
                                dlssParams.useAutoExposure ? "true" : "false");
                            log::warning("Possible causes:");
                            log::warning("  1. Resolution too small (DLSS requires minimum resolution)");
                            log::warning("  2. Missing required textures (depth, motion vectors)");
                            log::warning("  3. NGX CreateFeature failed - check DLSS library logs");
                        }
                    }
                    else
                    {
                        m_DLSSAvailable = false;
                        log::warning("DLSS Init() failed - DLSS may not be available or initialization parameters are incorrect");
                        log::warning("Resolution: %dx%d, useLinearDepth: %s, useAutoExposure: %s", 
                            fbinfo.width, fbinfo.height, 
                            dlssParams.useLinearDepth ? "true" : "false",
                            dlssParams.useAutoExposure ? "true" : "false");
                        log::warning("Possible causes:");
                        log::warning("  1. Resolution too small (DLSS requires minimum resolution)");
                        log::warning("  2. Missing required textures (depth, motion vectors)");
                        log::warning("  3. NGX CreateFeature failed - check DLSS library logs");
                    }
                }
            }
#endif

            // Create binding set
            nvrhi::BindingSetDesc bindingSetDesc;
            bindingSetDesc.bindings = {
                nvrhi::BindingSetItem::RayTracingAccelStruct(0, m_TopLevelAS),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_VertexBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(2, m_IndexBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(3, m_MaterialBuffer),
                nvrhi::BindingSetItem::StructuredBuffer_SRV(4, m_InstanceBuffer),
                nvrhi::BindingSetItem::Texture_SRV(5, m_EnvironmentMap),
                nvrhi::BindingSetItem::Texture_UAV(0, m_RenderTarget),
                nvrhi::BindingSetItem::Texture_UAV(1, m_AccumulationTarget),
                nvrhi::BindingSetItem::Texture_UAV(2, m_DepthBuffer),
                nvrhi::BindingSetItem::Texture_UAV(3, m_MotionVectors),
                nvrhi::BindingSetItem::Texture_UAV(4, m_DiffuseAlbedo),
                nvrhi::BindingSetItem::Texture_UAV(5, m_SpecularAlbedo),
                nvrhi::BindingSetItem::Texture_UAV(6, m_NormalRoughness),
                nvrhi::BindingSetItem::ConstantBuffer(0, m_CameraBuffer),
                nvrhi::BindingSetItem::Sampler(0, m_LinearSampler)
            };
            
            // Add material textures array (64 slots starting at t6)
            for (int i = 0; i < MAX_MATERIAL_TEXTURES; i++)
            {
                nvrhi::TextureHandle tex = m_DefaultMaterialTexture;
                if (i < static_cast<int>(m_MaterialTextures.size()))
                {
                    tex = m_MaterialTextures[i];
                }
                bindingSetDesc.bindings.push_back(
                    nvrhi::BindingSetItem::Texture_SRV(6, tex).setArrayElement(i)
                );
            }
            
            m_BindingSet = GetDevice()->createBindingSet(bindingSetDesc, m_BindingLayout);
        }

        // Update camera constants using HMM
        float aspect = float(fbinfo.width) / float(fbinfo.height);
        
        // Mitsuba uses horizontal FOV by default, convert to vertical FOV
        float horizontalFovRadians = m_SceneParser.camera.fov * (HMM_PI32 / 180.0f);
        float verticalFovRadians = 2.0f * atanf(tanf(horizontalFovRadians * 0.5f) / aspect);

        // Use RH perspective with [0,1] depth (matches Mitsuba's RH convention)
        HMM_Mat4 proj = HMM_Perspective_RH_ZO(verticalFovRadians, aspect, 0.1f, 10000.0f);
        HMM_Mat4 projInverse = HMM_InvPerspective_RH(proj);
        
        // Compute view matrix from current camera position/target, then invert
        // This updates dynamically when camera moves (unlike using scene's initial transform)
        HMM_Mat4 view = HMM_LookAt_RH(m_CameraPosition, m_CameraTarget, HMM_V3(0.0f, 1.0f, 0.0f));
        HMM_Mat4 viewInverse = HMM_InvGeneralM4(view);

        // Copy matrices to GPU format
        memcpy(m_CameraConstants.viewInverse, &viewInverse, sizeof(float) * 16);
        memcpy(m_CameraConstants.projInverse, &projInverse, sizeof(float) * 16);
        m_CameraConstants.cameraPosition[0] = m_CameraPosition.X;
        m_CameraConstants.cameraPosition[1] = m_CameraPosition.Y;
        m_CameraConstants.cameraPosition[2] = m_CameraPosition.Z;
        m_CameraConstants.frameIndex = m_FrameIndex;
        m_CameraConstants.envMapIntensity = m_SceneParser.environmentMap.intensity;
        m_CameraConstants.hasEnvMap = m_SceneParser.environmentMap.isValid ? 1 : 0;
        m_CameraConstants.exposure = m_Exposure;

        // Debug: print first frame info
        static bool firstFrame = true;
        if (firstFrame)
        {
            log::info("=== RT DEBUG (HMM) ===");
            log::info("Camera pos: (%.2f, %.2f, %.2f)", m_CameraPosition.X, m_CameraPosition.Y, m_CameraPosition.Z);
            log::info("Camera target: (%.2f, %.2f, %.2f)", m_CameraTarget.X, m_CameraTarget.Y, m_CameraTarget.Z);
            log::info("viewInverse col0: %.3f %.3f %.3f %.3f", 
                viewInverse.Columns[0].X, viewInverse.Columns[0].Y, viewInverse.Columns[0].Z, viewInverse.Columns[0].W);
            log::info("viewInverse col2: %.3f %.3f %.3f %.3f", 
                viewInverse.Columns[2].X, viewInverse.Columns[2].Y, viewInverse.Columns[2].Z, viewInverse.Columns[2].W);
            log::info("viewInverse col3 (pos): %.3f %.3f %.3f %.3f", 
                viewInverse.Columns[3].X, viewInverse.Columns[3].Y, viewInverse.Columns[3].Z, viewInverse.Columns[3].W);
            firstFrame = false;
        }

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

        // Choose output texture based on DLSS state
        nvrhi::TextureHandle outputTexture = m_RenderTarget;

#if DONUT_WITH_DLSS
        // Apply DLSS if enabled and available
        if (m_DLSSEnabled && m_DLSS && m_DLSS->IsDlssInitialized())
        {
            // Create a simple PlanarView for DLSS
            engine::PlanarView planarView;
            planarView.SetViewport(nvrhi::Viewport(float(fbinfo.width), float(fbinfo.height)));
            
            // Convert HMM view matrix to donut affine3
            // HMM matrices are column-major, affine3 needs 3x3 linear + translation
            dm::float3 col0(view.Columns[0].X, view.Columns[0].Y, view.Columns[0].Z);
            dm::float3 col1(view.Columns[1].X, view.Columns[1].Y, view.Columns[1].Z);
            dm::float3 col2(view.Columns[2].X, view.Columns[2].Y, view.Columns[2].Z);
            dm::float3 translation(view.Columns[3].X, view.Columns[3].Y, view.Columns[3].Z);
            dm::affine3 viewAffine = dm::affine3::from_cols(col0, col1, col2, translation);
            
            // Convert projection matrix
            dm::float4x4 projMatrix;
            memcpy(&projMatrix, &proj, sizeof(float) * 16);
            planarView.SetMatrices(viewAffine, projMatrix);
            
            render::DLSS::EvaluateParameters dlssParams;
            dlssParams.depthTexture = m_DepthBuffer;
            dlssParams.motionVectorsTexture = m_MotionVectors;
            dlssParams.inputColorTexture = m_RenderTarget;
            dlssParams.outputColorTexture = m_DLSSOutput;
            dlssParams.resetHistory = (m_FrameIndex == 0);
            
            // DLSS RR specific textures
            if (m_DLSS->IsRayReconstructionSupported())
            {
                dlssParams.diffuseAlbedo = m_DiffuseAlbedo;
                dlssParams.specularAlbedo = m_SpecularAlbedo;
                dlssParams.normalRoughness = m_NormalRoughness;
            }
            
            m_CommandList->close();
            GetDevice()->executeCommandList(m_CommandList);
            
            // DLSS needs its own command list execution
            nvrhi::CommandListHandle dlssCommandList = GetDevice()->createCommandList();
            dlssCommandList->open();
            m_DLSS->Evaluate(dlssCommandList, dlssParams, planarView);
            dlssCommandList->close();
            GetDevice()->executeCommandList(dlssCommandList);
            
            // Re-open for blit
            m_CommandList->open();
            outputTexture = m_DLSSOutput;
        }
#endif

        // Blit to framebuffer
        m_CommonPasses->BlitTexture(m_CommandList, framebuffer, outputTexture, m_BindingCache.get());

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        m_FrameIndex++;
    }
};

// ============================================================================
// ImGui UI Renderer
// ============================================================================
class UIRenderer : public app::ImGui_Renderer
{
private:
    RayTracedScene* m_pScene = nullptr;
    
public:
    UIRenderer(app::DeviceManager* deviceManager, RayTracedScene* scene)
        : ImGui_Renderer(deviceManager)
        , m_pScene(scene)
    {
    }
    
protected:
    void buildUI() override
    {
        if (!m_pScene)
            return;
            
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 280), ImGuiCond_FirstUseEver);
        
        if (ImGui::Begin("Render Settings"))
        {
            ImGui::Text("Frame: %u", m_pScene->GetFrameIndex());
            ImGui::Separator();
            
#if DONUT_WITH_DLSS
            // DLSS controls
            if (m_pScene->IsDLSSAvailable())
            {
                bool dlssEnabled = m_pScene->GetDLSSEnabled();
                if (ImGui::Checkbox("DLSS Ray Reconstruction", &dlssEnabled))
                {
                    m_pScene->GetDLSSEnabled() = dlssEnabled;
                    m_pScene->ResetAccumulation();
                }
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Enable NVIDIA DLSS Ray Reconstruction for denoising");
                }
            }
            else
            {
                ImGui::TextDisabled("DLSS not available");
            }
            ImGui::Separator();
#endif
            
            // Exposure control
            float exposure = m_pScene->GetExposure();
            if (ImGui::SliderFloat("Exposure", &exposure, 0.001f, 1.0f, "%.4f", ImGuiSliderFlags_Logarithmic))
            {
                m_pScene->GetExposure() = exposure;
                m_pScene->ResetAccumulation();
            }
            
            // Max bounces control
            int maxBounces = static_cast<int>(m_pScene->GetMaxBounces());
            if (ImGui::SliderInt("Max Bounces", &maxBounces, 1, 256))
            {
                m_pScene->GetMaxBounces() = static_cast<uint32_t>(maxBounces);
                m_pScene->ResetAccumulation();
            }
            
            ImGui::Separator();
            ImGui::Text("Controls:");
            ImGui::BulletText("WASD - Move camera");
            ImGui::BulletText("QE - Move up/down");
            ImGui::BulletText("Right Mouse - Look around");
        }
        ImGui::End();
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
    deviceParams.enableRayTracingExtensions = true;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

#if DONUT_WITH_DLSS
    // DLSS requires specific Vulkan extensions - must be added BEFORE device creation
    if (api == nvrhi::GraphicsAPI::VULKAN)
    {
        render::DLSS::GetRequiredVulkanExtensions(
            deviceParams.optionalVulkanInstanceExtensions,
            deviceParams.optionalVulkanDeviceExtensions);
    }
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
        // Create shader factory for ImGui
        std::shared_ptr<vfs::RootFileSystem> rootFs = std::make_shared<vfs::RootFileSystem>();
        rootFs->mount("/", std::filesystem::current_path().parent_path() / "shaders");

        std::filesystem::path frameworkShaderPath = app::GetDirectoryWithExecutable() / "shaders/framework" / app::GetShaderTypeName(deviceManager->GetDevice()->getGraphicsAPI());
        rootFs->mount("/donut", frameworkShaderPath);

        std::shared_ptr<engine::ShaderFactory> shaderFactory = std::make_shared<engine::ShaderFactory>(
            deviceManager->GetDevice(), rootFs, "/");
        
        RayTracedScene example(deviceManager, scenePath);
        if (example.Init())
        {
            UIRenderer ui(deviceManager, &example);
            ui.Init(shaderFactory);
            
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->AddRenderPassToBack(&ui);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&ui);
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;

    return 0;
}
