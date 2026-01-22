// ============================================================================
// plastic.hlsli - Plastic BRDF (Diffuse + Specular Coating)
// 
// Implements Mitsuba3-compatible plastic materials.
// Plastic materials combine a diffuse substrate with a dielectric coating.
// The coating reflects some light specularly, while the rest is transmitted
// through the coating and scattered diffusely by the substrate.
//
// Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_PLASTIC_HLSLI
#define MATERIALS_PLASTIC_HLSLI

#include "common.hlsli"
#include "diffuse.hlsli"

// ============================================================================
// Internal Scattering Helpers (Mitsuba3 compatible)
// ============================================================================

// Compute average Fresnel reflectance for diffuse incidence
// This is the hemispherical integral of Fresnel over cosine-weighted directions
float AverageFresnel(float eta)
{
    // Approximation from "Revisiting Physically Based Shading at Imageworks"
    // For external reflection (eta < 1)
    if (eta >= 1.0f)
    {
        // Internal reflection average
        return saturate(-1.4399f * eta * eta + 0.7099f * eta + 0.6681f + 0.0636f / eta);
    }
    else
    {
        // External reflection average
        float eta2 = eta * eta;
        float eta3 = eta2 * eta;
        return saturate(0.919317f - 3.4793f * eta + 6.75335f * eta2 - 7.80989f * eta3 +
                       4.98554f * eta2 * eta2 - 1.36881f * eta2 * eta3);
    }
}

// Compute the internal diffuse reflectance accounting for multiple bounces
// Based on Mitsuba's plastic model
float3 InternalDiffuseReflectance(float3 diffuseColor, float fdrInt)
{
    // Account for multiple internal reflections
    // This is the "nonlinear" effect in Mitsuba
    float3 denom = 1.0f - diffuseColor * fdrInt;
    return diffuseColor / max(denom, float3(EPSILON, EPSILON, EPSILON));
}

// ============================================================================
// Smooth Plastic (Mitsuba3 compatible)
// 
// Simple model: Fresnel-weighted blend between specular and diffuse
// ============================================================================

// Evaluate smooth plastic BRDF
float3 SmoothPlastic_Evaluate(float3 diffuseColor, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    // Fresnel at the interface
    float eta = extIOR / intIOR;
    float Fv = FresnelDielectric(NdotV, eta);
    float Fl = FresnelDielectric(NdotL, eta);
    
    // Diffuse component (light that enters and exits the coating)
    // Accounts for Fresnel transmission both ways
    // For smooth plastic, we use the simple linear model (nonlinear=false in Mitsuba)
    float3 diffuse = diffuseColor * INV_PI * (1.0f - Fv) * (1.0f - Fl);
    
    // Specular component is delta distribution (perfect reflection)
    // Only contributes if wo and wi are perfect reflection of each other
    // For evaluation, we return 0 for specular (handled in sampling)
    
    return diffuse;
}

// Sample smooth plastic
float3 SmoothPlastic_Sample(float3 diffuseColor, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    float NdotV = dot(normal, wo);
    if (NdotV <= 0.0f)
    {
        pdf = 0.0f;
        throughputWeight = float3(0.0f, 0.0f, 0.0f);
        return float3(0.0f, 0.0f, 0.0f);
    }
    
    // Fresnel reflectance
    float eta = extIOR / intIOR;
    float F = FresnelDielectric(NdotV, eta);
    
    // Choose between specular reflection and diffuse
    if (u.x < F)
    {
        // Specular reflection
        float3 wi = reflect(-wo, normal);
        pdf = F;
        throughputWeight = float3(1.0f, 1.0f, 1.0f);
        return wi;
    }
    else
    {
        // Diffuse
        float3 wi = Diffuse_Sample(normal, float2((u.x - F) / (1.0f - F), u.y), pdf);
        pdf *= (1.0f - F);
        
        float NdotL = max(0.0f, dot(normal, wi));
        float Fl = FresnelDielectric(NdotL, eta);
        
        // Weight includes Fresnel transmission
        throughputWeight = diffuseColor * (1.0f - Fl);
        return wi;
    }
}

// ============================================================================
// Rough Plastic (GGX Microfacet Coating) - Mitsuba3 compatible
// 
// More physically accurate model with rough dielectric coating over diffuse.
// Based on Mitsuba3's roughplastic implementation.
// ============================================================================

// Evaluate rough plastic BRDF (Mitsuba3 compatible)
float3 RoughPlastic_Evaluate(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    float eta = extIOR / intIOR;
    float alpha = RoughnessToAlpha(roughness);
    
    // Half-vector for specular
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float VdotH = max(0.0f, dot(wo, h));
    
    // Fresnel at half-vector angle (for specular)
    float F_spec = FresnelDielectric(VdotH, eta);
    
    // Specular term (GGX microfacet) - white specular for dielectric
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float3 specular = float3(F_spec, F_spec, F_spec) * D * G / (4.0f * NdotV * NdotL + EPSILON);
    
    // Diffuse term (Mitsuba3 model)
    // Use average Fresnel for energy conservation
    float fdrInt = AverageFresnel(1.0f / eta);  // Internal average Fresnel
    float fdrExt = AverageFresnel(eta);          // External average Fresnel
    
    // Fresnel transmission factors
    float Fv = FresnelDielectric(NdotV, eta);
    float Fl = FresnelDielectric(NdotL, eta);
    
    // Simple linear model (nonlinear=false in Mitsuba)
    // This preserves the original texture colors better
    float3 diffuse = diffuseColor * INV_PI * (1.0f - Fv) * (1.0f - Fl);
    
    return specular + diffuse;
}

// Sample rough plastic (Mitsuba3 compatible)
// Uses MIS between specular and diffuse lobes
float3 RoughPlastic_Sample(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    float NdotV = dot(normal, wo);
    if (NdotV <= 0.0f)
    {
        pdf = 0.0f;
        throughputWeight = float3(0.0f, 0.0f, 0.0f);
        return float3(0.0f, 0.0f, 0.0f);
    }
    
    float eta = extIOR / intIOR;
    float alpha = RoughnessToAlpha(roughness);
    
    // Compute specular sampling weight based on Fresnel and material properties
    // This follows Mitsuba's approach of using Fresnel at the viewing angle
    float F = FresnelDielectric(NdotV, eta);
    
    // For dielectric plastic, the specular weight should be based on:
    // 1. Fresnel reflectance at viewing angle
    // 2. Relative importance of specular vs diffuse (diffuse dominates for most plastics)
    // Mitsuba uses a fixed ratio based on material properties
    float diffuseWeight = Luminance(diffuseColor) * (1.0f - F);
    float specularWeight = F;
    float totalWeight = diffuseWeight + specularWeight;
    
    if (totalWeight < EPSILON)
    {
        pdf = 0.0f;
        throughputWeight = float3(0.0f, 0.0f, 0.0f);
        return float3(0.0f, 0.0f, 1.0f);
    }
    
    float probSpec = specularWeight / totalWeight;
    
    float3 wi;
    
    if (u.x < probSpec)
    {
        // Sample specular lobe (GGX)
        float2 u2 = float2(u.x / probSpec, u.y);
        
        float3 h = LocalToWorld(SampleGGX(u2, alpha), normal);
        wi = reflect(-wo, h);
        
        float NdotL = dot(normal, wi);
        if (NdotL <= 0.0f)
        {
            pdf = 0.0f;
            throughputWeight = float3(0.0f, 0.0f, 0.0f);
            return wi;
        }
        
        float NdotH = max(0.0f, dot(normal, h));
        float VdotH = max(0.0f, dot(wo, h));
        
        // Compute MIS PDF
        float specPdf = D_GGX(NdotH, alpha) * NdotH / (4.0f * VdotH + EPSILON);
        float diffPdf = NdotL * INV_PI;
        pdf = probSpec * specPdf + (1.0f - probSpec) * diffPdf;
        
        // Evaluate full BRDF and compute weight
        float3 brdf = RoughPlastic_Evaluate(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal);
        throughputWeight = brdf * NdotL / (pdf + EPSILON);
    }
    else
    {
        // Sample diffuse lobe (cosine-weighted)
        float2 u2 = float2((u.x - probSpec) / (1.0f - probSpec), u.y);
        float diffPdfOnly;
        wi = Diffuse_Sample(normal, u2, diffPdfOnly);
        
        float NdotL = max(0.0f, dot(normal, wi));
        if (NdotL <= 0.0f)
        {
            pdf = 0.0f;
            throughputWeight = float3(0.0f, 0.0f, 0.0f);
            return wi;
        }
        
        // Compute MIS PDF (need specular PDF for MIS)
        float3 h = normalize(wo + wi);
        float NdotH = max(0.0f, dot(normal, h));
        float VdotH = max(0.0f, dot(wo, h));
        float specPdf = D_GGX(NdotH, alpha) * NdotH / (4.0f * VdotH + EPSILON);
        float diffPdf = NdotL * INV_PI;
        pdf = probSpec * specPdf + (1.0f - probSpec) * diffPdf;
        
        // Evaluate full BRDF and compute weight
        float3 brdf = RoughPlastic_Evaluate(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal);
        throughputWeight = brdf * NdotL / (pdf + EPSILON);
    }
    
    return wi;
}

// ============================================================================
// Combined Plastic Interface
// ============================================================================

float3 Plastic_Evaluate(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    if (roughness < 0.01f)
        return SmoothPlastic_Evaluate(diffuseColor, intIOR, extIOR, wo, wi, normal);
    else
        return RoughPlastic_Evaluate(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal);
}

float3 Plastic_Sample(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    if (roughness < 0.01f)
        return SmoothPlastic_Sample(diffuseColor, intIOR, extIOR, wo, normal, u, throughputWeight, pdf);
    else
        return RoughPlastic_Sample(diffuseColor, roughness, intIOR, extIOR, wo, normal, u, throughputWeight, pdf);
}

#endif // MATERIALS_PLASTIC_HLSLI
