// ============================================================================
// dielectric.hlsli - Dielectric (Glass/Water) BSDF
// 
// Implements both smooth and rough dielectric materials with refraction.
// Uses GGX microfacet model for rough surfaces.
// ============================================================================
#ifndef MATERIALS_DIELECTRIC_HLSLI
#define MATERIALS_DIELECTRIC_HLSLI

#include "common.hlsli"

// ============================================================================
// Smooth Dielectric (Perfect Glass)
// ============================================================================

// Sample smooth dielectric - choose between reflection and refraction
float3 SmoothDielectric_Sample(float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf, out bool isRefracted)
{
    float eta = extIOR / intIOR;
    float cosI = dot(wo, normal);
    
    // Determine if we're entering or exiting the medium
    float3 n = normal;
    if (cosI < 0.0f)
    {
        // Exiting the medium
        n = -normal;
        cosI = -cosI;
        eta = intIOR / extIOR;
    }
    
    // Compute Fresnel reflectance
    float F = FresnelDielectric(cosI, eta);
    
    // Russian roulette to choose between reflection and refraction
    if (u.x < F)
    {
        // Reflection
        float3 wi = reflect(-wo, n);
        pdf = 1.0f;
        throughputWeight = float3(1.0f, 1.0f, 1.0f);  // Energy conserving
        isRefracted = false;
        return wi;
    }
    else
    {
        // Refraction
        float3 wi;
        if (Refract(wo, n, eta, wi))
        {
            pdf = 1.0f;
            // Account for non-symmetry of refraction (radiance compression)
            float etaRatio = (cosI > 0.0f) ? (intIOR / extIOR) : (extIOR / intIOR);
            throughputWeight = float3(1.0f, 1.0f, 1.0f) * etaRatio * etaRatio;
            isRefracted = true;
            return wi;
        }
        else
        {
            // Total internal reflection
            float3 wi = reflect(-wo, n);
            pdf = 1.0f;
            throughputWeight = float3(1.0f, 1.0f, 1.0f);
            isRefracted = false;
            return wi;
        }
    }
}

// Evaluate smooth dielectric - returns 0 since it's a delta distribution
float3 SmoothDielectric_Evaluate(float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    return float3(0.0f, 0.0f, 0.0f);
}

// ============================================================================
// Rough Dielectric (GGX Microfacet)
// ============================================================================

// Sample rough dielectric
float3 RoughDielectric_Sample(float intIOR, float extIOR, float roughness, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf, out bool isRefracted)
{
    float alpha = RoughnessToAlpha(roughness);
    
    // Sample microfacet normal
    float3 h = LocalToWorld(SampleGGX(float2(u.y, frac(u.x * 2.0f)), alpha), normal);
    
    float cosI = dot(wo, h);
    float eta = extIOR / intIOR;
    
    // Determine if we're entering or exiting
    float3 n = h;
    if (cosI < 0.0f)
    {
        n = -h;
        cosI = -cosI;
        eta = intIOR / extIOR;
    }
    
    // Compute Fresnel
    float F = FresnelDielectric(cosI, eta);
    
    float NdotV = abs(dot(normal, wo));
    float NdotH = abs(dot(normal, h));
    float VdotH = abs(dot(wo, h));
    
    float D = D_GGX(NdotH, alpha);
    
    // Choose reflection or refraction
    if (u.x < F)
    {
        // Reflection
        float3 wi = reflect(-wo, h);
        float NdotL = abs(dot(normal, wi));
        
        if (NdotL <= 0.0f || dot(normal, wo) * dot(normal, wi) <= 0.0f)
        {
            pdf = 0.0f;
            throughputWeight = float3(0.0f, 0.0f, 0.0f);
            isRefracted = false;
            return wi;
        }
        
        float G = G_SmithGGX(NdotV, NdotL, alpha);
        
        // PDF for reflection
        pdf = F * D * NdotH / (4.0f * VdotH + EPSILON);
        
        // Throughput weight
        throughputWeight = float3(1.0f, 1.0f, 1.0f) * G * VdotH / (NdotV * NdotH + EPSILON);
        isRefracted = false;
        return wi;
    }
    else
    {
        // Refraction
        float3 wi;
        if (!Refract(wo, n, eta, wi))
        {
            // Total internal reflection
            wi = reflect(-wo, h);
            float NdotL = abs(dot(normal, wi));
            
            if (NdotL <= 0.0f)
            {
                pdf = 0.0f;
                throughputWeight = float3(0.0f, 0.0f, 0.0f);
                isRefracted = false;
                return wi;
            }
            
            float G = G_SmithGGX(NdotV, NdotL, alpha);
            pdf = D * NdotH / (4.0f * VdotH + EPSILON);
            throughputWeight = float3(1.0f, 1.0f, 1.0f) * G * VdotH / (NdotV * NdotH + EPSILON);
            isRefracted = false;
            return wi;
        }
        
        float NdotL = abs(dot(normal, wi));
        float LdotH = abs(dot(wi, h));
        
        if (NdotL <= 0.0f)
        {
            pdf = 0.0f;
            throughputWeight = float3(0.0f, 0.0f, 0.0f);
            isRefracted = true;
            return wi;
        }
        
        float G = G_SmithGGX(NdotV, NdotL, alpha);
        
        // PDF for refraction (more complex due to change of variables)
        float sqrtDenom = VdotH + eta * LdotH;
        float dwh_dwi = abs(LdotH) / (sqrtDenom * sqrtDenom + EPSILON);
        pdf = (1.0f - F) * D * NdotH * dwh_dwi;
        
        // Throughput weight with radiance compression
        float etaRatio = (dot(wo, normal) > 0.0f) ? (intIOR / extIOR) : (extIOR / intIOR);
        throughputWeight = float3(1.0f, 1.0f, 1.0f) * etaRatio * etaRatio * G * VdotH / (NdotV * NdotH + EPSILON);
        isRefracted = true;
        return wi;
    }
}

// Evaluate rough dielectric BSDF
float3 RoughDielectric_Evaluate(float intIOR, float extIOR, float roughness, float3 wo, float3 wi, float3 normal)
{
    float eta = extIOR / intIOR;
    float alpha = RoughnessToAlpha(roughness);
    
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    // Check if reflection or refraction
    bool isReflection = (NdotV * NdotL > 0.0f);
    
    if (isReflection)
    {
        // Reflection
        if (NdotV <= 0.0f) NdotV = -NdotV;
        if (NdotL <= 0.0f) NdotL = -NdotL;
        
        float3 h = normalize(wo + wi);
        float NdotH = abs(dot(normal, h));
        float VdotH = abs(dot(wo, h));
        
        float D = D_GGX(NdotH, alpha);
        float G = G_SmithGGX(NdotV, NdotL, alpha);
        float F = FresnelDielectric(VdotH, eta);
        
        return float3(1.0f, 1.0f, 1.0f) * F * D * G / (4.0f * NdotV * NdotL + EPSILON);
    }
    else
    {
        // Refraction
        bool entering = NdotV > 0.0f;
        if (!entering)
        {
            NdotV = -NdotV;
            NdotL = -NdotL;
            eta = 1.0f / eta;
        }
        
        // Compute half-vector for refraction
        float3 h = normalize(wo + eta * wi);
        if (dot(h, normal) < 0.0f) h = -h;
        
        float VdotH = abs(dot(wo, h));
        float LdotH = abs(dot(wi, h));
        float NdotH = abs(dot(normal, h));
        
        float D = D_GGX(NdotH, alpha);
        float G = G_SmithGGX(NdotV, NdotL, alpha);
        float F = FresnelDielectric(VdotH, eta);
        
        float sqrtDenom = VdotH + eta * LdotH;
        
        return float3(1.0f, 1.0f, 1.0f) * (1.0f - F) * D * G * VdotH * LdotH / 
               (NdotV * NdotL * sqrtDenom * sqrtDenom + EPSILON);
    }
}

// ============================================================================
// Combined Dielectric Interface
// ============================================================================

float3 Dielectric_Sample(float intIOR, float extIOR, float roughness, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf, out bool isRefracted)
{
    if (roughness < 0.01f)
        return SmoothDielectric_Sample(intIOR, extIOR, wo, normal, u, throughputWeight, pdf, isRefracted);
    else
        return RoughDielectric_Sample(intIOR, extIOR, roughness, wo, normal, u, throughputWeight, pdf, isRefracted);
}

float3 Dielectric_Evaluate(float intIOR, float extIOR, float roughness, float3 wo, float3 wi, float3 normal)
{
    if (roughness < 0.01f)
        return SmoothDielectric_Evaluate(intIOR, extIOR, wo, wi, normal);
    else
        return RoughDielectric_Evaluate(intIOR, extIOR, roughness, wo, wi, normal);
}

#endif // MATERIALS_DIELECTRIC_HLSLI
