// ============================================================================
// material.hlsli - Unified Material Interface
// 
// This file provides a unified interface for all material types.
// Include this file to access all material evaluation and sampling functions.
// ============================================================================
#ifndef MATERIALS_MATERIAL_HLSLI
#define MATERIALS_MATERIAL_HLSLI

#include "common.hlsli"
#include "diffuse.hlsli"
#include "conductor.hlsli"
#include "dielectric.hlsli"
#include "plastic.hlsli"
#include "thindielectric.hlsli"
#include "principled.hlsli"
#include "modifiers.hlsli"
#include "normalmap.hlsli"

// ============================================================================
// Material Type Constants
// ============================================================================
static const uint MATERIAL_DIFFUSE = 0;
static const uint MATERIAL_CONDUCTOR = 1;
static const uint MATERIAL_ROUGH_CONDUCTOR = 2;
static const uint MATERIAL_DIELECTRIC = 3;
static const uint MATERIAL_ROUGH_DIELECTRIC = 4;
static const uint MATERIAL_PLASTIC = 5;
static const uint MATERIAL_ROUGH_PLASTIC = 6;
static const uint MATERIAL_THIN_DIELECTRIC = 7;
static const uint MATERIAL_PRINCIPLED = 8;
static const uint MATERIAL_BLEND = 9;
static const uint MATERIAL_MASK = 10;
static const uint MATERIAL_NULL = 11;

// ============================================================================
// Material Structure
// ============================================================================
struct MaterialParams
{
    float3 baseColor;      // Diffuse albedo or specular color (F0)
    float roughness;       // Surface roughness [0, 1]
    
    float3 eta;            // Complex IOR real part (for conductors)
    float metallic;        // Metallic factor [0, 1]
    
    float3 k;              // Complex IOR imaginary part (for conductors)
    uint type;             // Material type (see constants above)
    
    float intIOR;          // Interior IOR (for dielectrics/plastics)
    float extIOR;          // Exterior IOR (for dielectrics/plastics)
    
    // Principled BSDF parameters
    float specular;        // Specular intensity (F0 = 0.08 * specular for dielectrics)
    float specTint;        // Tint specular towards base color
    float sheen;           // Sheen intensity
    float sheenTint;       // Tint sheen towards base color
    float clearcoat;       // Clearcoat intensity
    float clearcoatGloss;  // Clearcoat glossiness
    float specTrans;       // Specular transmission (glass-like)
    
    // Mask/Blend parameters
    float opacity;         // Opacity for mask material [0, 1]
    float blendWeight;     // Blend weight for blendbsdf [0, 1]
};

// Convert MaterialParams to PrincipledParams
PrincipledParams MaterialToPrincipled(MaterialParams mat)
{
    PrincipledParams p;
    p.baseColor = mat.baseColor;
    p.roughness = mat.roughness;
    p.metallic = mat.metallic;
    p.specular = mat.specular;
    p.specTint = mat.specTint;
    p.anisotropic = 0.0f;  // Not yet supported in GPU struct
    p.sheen = mat.sheen;
    p.sheenTint = mat.sheenTint;
    p.clearcoat = mat.clearcoat;
    p.clearcoatGloss = mat.clearcoatGloss;
    p.specTrans = mat.specTrans;
    p.eta = mat.intIOR / mat.extIOR;
    return p;
}

// ============================================================================
// Unified Evaluation Function
// 
// Evaluates the BSDF for given directions.
// Returns the BSDF value f(wo, wi).
// ============================================================================
float3 Material_Evaluate(MaterialParams mat, float3 wo, float3 wi, float3 normal)
{
    float NdotL = dot(normal, wi);
    float NdotV = dot(normal, wo);
    
    switch (mat.type)
    {
        case MATERIAL_DIFFUSE:
            return Diffuse_Evaluate(mat.baseColor, wo, wi, normal);
            
        case MATERIAL_CONDUCTOR:
        case MATERIAL_ROUGH_CONDUCTOR:
        {
            // Check if this is a perfect mirror (material="none")
            bool isPerfectMirror = IsPerfectMirror(mat.eta, mat.k);
            
            if (isPerfectMirror)
            {
                // Perfect mirror - use Schlick with baseColor
                return ConductorSchlick_Evaluate(mat.baseColor, mat.roughness, wo, wi, normal);
            }
            else
            {
                // Real conductor with complex IOR
                return Conductor_Evaluate(mat.eta, mat.k, mat.roughness, wo, wi, normal);
            }
        }
            
        case MATERIAL_DIELECTRIC:
        case MATERIAL_ROUGH_DIELECTRIC:
            return Dielectric_Evaluate(mat.intIOR, mat.extIOR, mat.roughness, wo, wi, normal);
            
        case MATERIAL_PLASTIC:
        case MATERIAL_ROUGH_PLASTIC:
            return Plastic_Evaluate(mat.baseColor, mat.roughness, mat.intIOR, mat.extIOR, wo, wi, normal);
        
        case MATERIAL_THIN_DIELECTRIC:
            return ThinDielectric_Evaluate(mat.intIOR, mat.extIOR, wo, wi, normal);
        
        case MATERIAL_PRINCIPLED:
        {
            PrincipledParams p = MaterialToPrincipled(mat);
            return Principled_Evaluate(p, wo, wi, normal);
        }
        
        case MATERIAL_NULL:
            return Null_Evaluate(wo, wi, normal);
        
        case MATERIAL_MASK:
            // Mask evaluation depends on the base material
            // For now, treat as diffuse
            return mat.opacity * Diffuse_Evaluate(mat.baseColor, wo, wi, normal);
            
        default:
            return Diffuse_Evaluate(mat.baseColor, wo, wi, normal);
    }
}

// ============================================================================
// Unified Sampling Function
// 
// Samples a new direction based on the material's BSDF.
// Returns the sampled direction wi.
// Outputs:
//   - throughputWeight: BSDF * cos(theta) / pdf (precomputed for efficiency)
//   - pdf: Probability density of the sampled direction
//   - isRefracted: True if the ray was refracted (for dielectrics)
// ============================================================================
float3 Material_Sample(MaterialParams mat, float3 wo, float3 normal, float2 u, 
                       out float3 throughputWeight, out float pdf, out bool isRefracted)
{
    isRefracted = false;
    
    switch (mat.type)
    {
        case MATERIAL_DIFFUSE:
        {
            float3 wi = Diffuse_Sample(normal, u, pdf);
            float NdotL = max(0.0f, dot(normal, wi));
            throughputWeight = mat.baseColor;  // baseColor * INV_PI * NdotL / (NdotL * INV_PI) = baseColor
            return wi;
        }
        
        case MATERIAL_CONDUCTOR:
        case MATERIAL_ROUGH_CONDUCTOR:
        {
            // Check if this is a perfect mirror (material="none" in Mitsuba)
            // Detect by checking if eta and k are both zero or default values
            bool isPerfectMirror = IsPerfectMirror(mat.eta, mat.k);
            bool isSmooth = (mat.roughness < 0.01f);
            
            if (isPerfectMirror)
            {
                // Perfect mirror - 100% reflective
                if (isSmooth)
                {
                    // Smooth perfect mirror
                    float3 wi = reflect(-wo, normal);
                    pdf = 1.0f;
                    throughputWeight = mat.baseColor;  // Use baseColor as specular_reflectance
                    return wi;
                }
                else
                {
                    // Rough perfect mirror - use Schlick with baseColor as F0
                    float alpha = RoughnessToAlpha(mat.roughness);
                    float3 h = LocalToWorld(SampleGGX(u, alpha), normal);
                    float3 wi = reflect(-wo, h);
                    
                    float NdotV = dot(normal, wo);
                    float NdotL = dot(normal, wi);
                    float NdotH = max(0.0f, dot(normal, h));
                    float VdotH = max(0.0f, dot(wo, h));
                    
                    if (NdotL <= 0.0f)
                    {
                        pdf = 0.0f;
                        throughputWeight = float3(0.0f, 0.0f, 0.0f);
                        return wi;
                    }
                    
                    float D = D_GGX(NdotH, alpha);
                    float G = G_SmithGGX(NdotV, NdotL, alpha);
                    // For perfect mirror without complex IOR, use 100% reflectance
                    float3 F = mat.baseColor;
                    
                    pdf = D * NdotH / (4.0f * VdotH + EPSILON);
                    throughputWeight = F * G * VdotH / (NdotV * NdotH + EPSILON);
                    return wi;
                }
            }
            else
            {
                // Real conductor with complex IOR - use proper Fresnel
                return Conductor_Sample(mat.eta, mat.k, mat.roughness, wo, normal, u, throughputWeight, pdf);
            }
        }
        
        case MATERIAL_DIELECTRIC:
        case MATERIAL_ROUGH_DIELECTRIC:
            return Dielectric_Sample(mat.intIOR, mat.extIOR, mat.roughness, wo, normal, u, throughputWeight, pdf, isRefracted);
        
        case MATERIAL_PLASTIC:
        case MATERIAL_ROUGH_PLASTIC:
            return Plastic_Sample(mat.baseColor, mat.roughness, mat.intIOR, mat.extIOR, wo, normal, u, throughputWeight, pdf);
        
        case MATERIAL_THIN_DIELECTRIC:
        {
            bool isTransmitted;
            float3 wi = ThinDielectric_Sample(mat.intIOR, mat.extIOR, wo, normal, u, throughputWeight, pdf, isTransmitted);
            isRefracted = isTransmitted;
            return wi;
        }
        
        case MATERIAL_PRINCIPLED:
        {
            PrincipledParams p = MaterialToPrincipled(mat);
            bool isTransmitted;
            float3 wi = Principled_Sample(p, wo, normal, u, throughputWeight, pdf, isTransmitted);
            isRefracted = isTransmitted;
            return wi;
        }
        
        case MATERIAL_NULL:
        {
            float3 wi = Null_Sample(wo, throughputWeight, pdf);
            isRefracted = true;  // Ray passes through
            return wi;
        }
        
        case MATERIAL_MASK:
        {
            // Decide if ray passes through or interacts with base material
            if (Mask_ShouldPassThrough(mat.opacity, u.x))
            {
                // Pass through (transparent)
                float3 wi = Mask_PassThrough(wo);
                pdf = 1.0f - mat.opacity;
                throughputWeight = float3(1.0f, 1.0f, 1.0f);
                isRefracted = true;
                return wi;
            }
            else
            {
                // Use diffuse as base material
                float2 u2 = float2(u.x / mat.opacity, u.y);
                float3 wi = Diffuse_Sample(normal, u2, pdf);
                pdf *= mat.opacity;
                throughputWeight = mat.baseColor;
                return wi;
            }
        }
        
        default:
        {
            float3 wi = Diffuse_Sample(normal, u, pdf);
            throughputWeight = mat.baseColor;
            return wi;
        }
    }
}

// ============================================================================
// Simplified Sampling (without refraction tracking)
// ============================================================================
float3 Material_SampleSimple(MaterialParams mat, float3 wo, float3 normal, float2 u, out float pdf)
{
    float3 throughputWeight;
    bool isRefracted;
    return Material_Sample(mat, wo, normal, u, throughputWeight, pdf, isRefracted);
}

// ============================================================================
// Check if material is emissive
// ============================================================================
bool Material_IsEmissive(MaterialParams mat)
{
    // This should be determined by the instance, not material
    return false;
}

// ============================================================================
// Check if material has delta distribution (perfect specular)
// ============================================================================
bool Material_IsDelta(MaterialParams mat)
{
    if (mat.roughness < 0.01f)
    {
        return (mat.type == MATERIAL_CONDUCTOR || 
                mat.type == MATERIAL_DIELECTRIC ||
                mat.type == MATERIAL_PLASTIC);
    }
    return false;
}

// ============================================================================
// Get the probability of sampling reflection vs refraction (for MIS)
// ============================================================================
float Material_GetReflectionProbability(MaterialParams mat, float3 wo, float3 normal)
{
    if (mat.type == MATERIAL_DIELECTRIC || mat.type == MATERIAL_ROUGH_DIELECTRIC)
    {
        float eta = mat.extIOR / mat.intIOR;
        float cosTheta = abs(dot(wo, normal));
        return FresnelDielectric(cosTheta, eta);
    }
    return 1.0f;  // Other materials always reflect
}

#endif // MATERIALS_MATERIAL_HLSLI
