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
};

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
            // Use complex IOR if available, otherwise use baseColor as F0
            if (any(mat.eta > 0.0f) || any(mat.k > 0.0f))
                return Conductor_Evaluate(mat.eta, mat.k, mat.roughness, wo, wi, normal);
            else
                return ConductorSchlick_Evaluate(mat.baseColor, mat.roughness, wo, wi, normal);
            
        case MATERIAL_DIELECTRIC:
        case MATERIAL_ROUGH_DIELECTRIC:
            return Dielectric_Evaluate(mat.intIOR, mat.extIOR, mat.roughness, wo, wi, normal);
            
        case MATERIAL_PLASTIC:
        case MATERIAL_ROUGH_PLASTIC:
            return Plastic_Evaluate(mat.baseColor, mat.roughness, mat.intIOR, mat.extIOR, wo, wi, normal);
            
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
            float3 eta = mat.eta;
            float3 k = mat.k;
            
            // If no complex IOR provided, derive from baseColor
            if (!any(eta > 0.0f) && !any(k > 0.0f))
            {
                eta = mat.baseColor;  // Use as F0 approximation
                k = float3(0.0f, 0.0f, 0.0f);
            }
            
            return Conductor_Sample(eta, k, mat.roughness, wo, normal, u, throughputWeight, pdf);
        }
        
        case MATERIAL_DIELECTRIC:
        case MATERIAL_ROUGH_DIELECTRIC:
            return Dielectric_Sample(mat.intIOR, mat.extIOR, mat.roughness, wo, normal, u, throughputWeight, pdf, isRefracted);
        
        case MATERIAL_PLASTIC:
        case MATERIAL_ROUGH_PLASTIC:
            return Plastic_Sample(mat.baseColor, mat.roughness, mat.intIOR, mat.extIOR, wo, normal, u, throughputWeight, pdf);
        
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
