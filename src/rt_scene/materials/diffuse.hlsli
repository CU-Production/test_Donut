// ============================================================================
// diffuse.hlsli - Lambertian Diffuse BRDF
// ============================================================================
#ifndef MATERIALS_DIFFUSE_HLSLI
#define MATERIALS_DIFFUSE_HLSLI

#include "common.hlsli"

// ============================================================================
// Lambertian Diffuse BRDF
// 
// The simplest diffuse model - light is scattered uniformly in all directions.
// f(wo, wi) = albedo / PI
// ============================================================================

// Evaluate Lambertian diffuse BRDF
float3 Diffuse_Evaluate(float3 albedo, float3 wo, float3 wi, float3 normal)
{
    float NdotL = dot(normal, wi);
    if (NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    return albedo * INV_PI;
}

// Sample Lambertian diffuse BRDF using cosine-weighted hemisphere sampling
float3 Diffuse_Sample(float3 normal, float2 u, out float pdf)
{
    // Cosine-weighted hemisphere sampling
    float3 localDir = CosineSampleHemisphere(u);
    float3 wi = LocalToWorld(localDir, normal);
    
    // PDF for cosine-weighted sampling: cos(theta) / PI
    float NdotL = max(0.0f, dot(normal, wi));
    pdf = NdotL * INV_PI;
    
    return wi;
}

// PDF of sampling direction wi for Lambertian diffuse
float Diffuse_PDF(float3 normal, float3 wi)
{
    float NdotL = max(0.0f, dot(normal, wi));
    return NdotL * INV_PI;
}

#endif // MATERIALS_DIFFUSE_HLSLI
