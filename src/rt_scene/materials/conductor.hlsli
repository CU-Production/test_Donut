// ============================================================================
// conductor.hlsli - Conductor (Metal) BRDF (Mitsuba3 compatible)
// 
// Implements both smooth (perfect mirror) and rough conductor materials
// using the GGX microfacet model with conductor Fresnel.
//
// Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_CONDUCTOR_HLSLI
#define MATERIALS_CONDUCTOR_HLSLI

#include "common.hlsli"
#include "conductor_ior.hlsli"

// ============================================================================
// Smooth Conductor (Perfect Mirror) - Mitsuba3 compatible
// ============================================================================

// Check if this is a "none" material (perfect mirror with 100% reflectance)
bool IsPerfectMirror(float3 eta, float3 k)
{
    // In Mitsuba, material="none" means 100% reflective mirror
    // We detect this by checking if eta is close to 0 or 1 and k is non-zero
    // Our convention: eta=(0,0,0) or eta=(1,1,1) with default k means perfect mirror
    bool etaIsDefault = (all(eta < 0.01f)) || 
                        (abs(eta.x - 1.0f) < 0.01f && abs(eta.y - 1.0f) < 0.01f && abs(eta.z - 1.0f) < 0.01f);
    bool kIsDefault = all(abs(k) < 0.01f) || all(abs(k - 1.0f) < 0.01f);
    return etaIsDefault && kIsDefault;
}

// Evaluate smooth conductor - returns 0 since it's a delta distribution
float3 SmoothConductor_Evaluate(float3 eta, float3 k, float3 wo, float3 wi, float3 normal)
{
    // Delta distribution - no contribution except at perfect reflection
    return float3(0.0f, 0.0f, 0.0f);
}

// Sample smooth conductor - perfect mirror reflection (Mitsuba3 compatible)
float3 SmoothConductor_Sample(float3 eta, float3 k, float3 specularReflectance, float3 wo, float3 normal, out float3 throughputWeight, out float pdf)
{
    float3 wi = reflect(-wo, normal);
    
    // For delta distributions, pdf is conceptually infinite, but we set to 1
    pdf = 1.0f;
    
    // Check for perfect mirror (material="none" in Mitsuba)
    if (IsPerfectMirror(eta, k))
    {
        // 100% reflective mirror - use specular_reflectance as the color
        throughputWeight = specularReflectance;
    }
    else
    {
        // Real conductor with complex IOR
        float cosTheta = abs(dot(wo, normal));
        float3 F = FresnelConductor(cosTheta, eta, k);
        throughputWeight = F * specularReflectance;
    }
    
    return wi;
}

// Legacy overload for compatibility
float3 SmoothConductor_Sample(float3 eta, float3 k, float3 wo, float3 normal, out float3 throughputWeight, out float pdf)
{
    return SmoothConductor_Sample(eta, k, float3(1.0f, 1.0f, 1.0f), wo, normal, throughputWeight, pdf);
}

// ============================================================================
// Rough Conductor (GGX Microfacet)
// 
// Uses the Cook-Torrance microfacet model:
// f(wo, wi) = D(h) * G(wo, wi) * F(wo, h) / (4 * |wo.n| * |wi.n|)
// ============================================================================

// Evaluate rough conductor BRDF
float3 RoughConductor_Evaluate(float3 eta, float3 k, float roughness, float3 wo, float3 wi, float3 normal)
{
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float VdotH = max(0.0f, dot(wo, h));
    
    float alpha = RoughnessToAlpha(roughness);
    
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float3 F = FresnelConductor(VdotH, eta, k);
    
    return (D * G * F) / (4.0f * NdotV * NdotL + EPSILON);
}

// Sample rough conductor using GGX importance sampling
float3 RoughConductor_Sample(float3 eta, float3 k, float roughness, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    float alpha = RoughnessToAlpha(roughness);
    
    // Sample microfacet normal using GGX
    float3 h = LocalToWorld(SampleGGX(u, alpha), normal);
    
    // Reflect to get light direction
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
    
    // PDF of sampling the half-vector, converted to solid angle measure
    float D = D_GGX(NdotH, alpha);
    pdf = D * NdotH / (4.0f * VdotH + EPSILON);
    
    // Evaluate BRDF
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float3 F = FresnelConductor(VdotH, eta, k);
    
    // Throughput weight = BRDF * cos(theta) / pdf
    // This simplifies nicely for GGX sampling
    throughputWeight = F * G * VdotH / (NdotV * NdotH + EPSILON);
    
    return wi;
}

// PDF for rough conductor
float RoughConductor_PDF(float roughness, float3 wo, float3 wi, float3 normal)
{
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float VdotH = max(0.0f, dot(wo, h));
    
    float alpha = RoughnessToAlpha(roughness);
    float D = D_GGX(NdotH, alpha);
    
    return D * NdotH / (4.0f * VdotH + EPSILON);
}

// ============================================================================
// Combined Conductor Interface
// ============================================================================

// Evaluate conductor (automatically selects smooth or rough)
float3 Conductor_Evaluate(float3 eta, float3 k, float roughness, float3 wo, float3 wi, float3 normal)
{
    if (roughness < 0.01f)
        return SmoothConductor_Evaluate(eta, k, wo, wi, normal);
    else
        return RoughConductor_Evaluate(eta, k, roughness, wo, wi, normal);
}

// Sample conductor (automatically selects smooth or rough)
float3 Conductor_Sample(float3 eta, float3 k, float roughness, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    if (roughness < 0.01f)
        return SmoothConductor_Sample(eta, k, wo, normal, throughputWeight, pdf);
    else
        return RoughConductor_Sample(eta, k, roughness, wo, normal, u, throughputWeight, pdf);
}

// ============================================================================
// Simplified Conductor using Schlick Fresnel (for colored metals)
// ============================================================================

// Evaluate using Schlick approximation (baseColor as F0)
float3 ConductorSchlick_Evaluate(float3 baseColor, float roughness, float3 wo, float3 wi, float3 normal)
{
    if (roughness < 0.01f)
        return float3(0.0f, 0.0f, 0.0f);  // Delta distribution
    
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float VdotH = max(0.0f, dot(wo, h));
    
    float alpha = RoughnessToAlpha(roughness);
    
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float3 F = FresnelSchlick(VdotH, baseColor);
    
    return (D * G * F) / (4.0f * NdotV * NdotL + EPSILON);
}

#endif // MATERIALS_CONDUCTOR_HLSLI
