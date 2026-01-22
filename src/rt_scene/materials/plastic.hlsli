// ============================================================================
// plastic.hlsli - Plastic BRDF (Diffuse + Specular Coating)
// 
// Plastic materials combine a diffuse substrate with a dielectric coating.
// The coating reflects some light specularly, while the rest is transmitted
// through the coating and scattered diffusely by the substrate.
// ============================================================================
#ifndef MATERIALS_PLASTIC_HLSLI
#define MATERIALS_PLASTIC_HLSLI

#include "common.hlsli"
#include "diffuse.hlsli"

// ============================================================================
// Smooth Plastic
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
// Rough Plastic (GGX Microfacet Coating)
// 
// More physically accurate model with rough dielectric coating over diffuse
// ============================================================================

// Evaluate rough plastic BRDF
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
    
    // Fresnel terms
    float Fv = FresnelDielectric(VdotH, eta);
    float Fl = FresnelDielectric(NdotL, eta);
    
    // Specular term (GGX microfacet)
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float3 specular = float3(1.0f, 1.0f, 1.0f) * Fv * D * G / (4.0f * NdotV * NdotL + EPSILON);
    
    // Diffuse term (with energy conservation)
    // Account for multiple bounces inside the coating (approximate)
    float avgFresnel = FresnelDielectric(0.5f, eta);  // Average Fresnel at 60 degrees
    float3 diffuse = diffuseColor * INV_PI * (1.0f - Fv) * (1.0f - Fl) / (1.0f - avgFresnel * Luminance(diffuseColor) + EPSILON);
    
    return specular + diffuse;
}

// Sample rough plastic
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
    
    // Estimate specular weight based on Fresnel and roughness
    float F = FresnelDielectric(NdotV, eta);
    float specularWeight = F + (1.0f - F) * 0.5f * (1.0f - roughness);  // Rough surfaces have more diffuse
    
    float3 wi;
    
    if (u.x < specularWeight)
    {
        // Sample specular lobe (GGX)
        float2 u2 = float2(u.x / specularWeight, u.y);
        
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
        
        float D = D_GGX(NdotH, alpha);
        float G = G_SmithGGX(NdotV, NdotL, alpha);
        float Fh = FresnelDielectric(VdotH, eta);
        
        // PDF: specularWeight * D * NdotH / (4 * VdotH)
        float specPdf = D * NdotH / (4.0f * VdotH + EPSILON);
        float diffPdf = NdotL * INV_PI;
        pdf = specularWeight * specPdf + (1.0f - specularWeight) * diffPdf;
        
        // Evaluate full BRDF
        float3 brdf = RoughPlastic_Evaluate(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal);
        throughputWeight = brdf * NdotL / (pdf + EPSILON);
    }
    else
    {
        // Sample diffuse lobe
        float2 u2 = float2((u.x - specularWeight) / (1.0f - specularWeight), u.y);
        wi = Diffuse_Sample(normal, u2, pdf);
        
        float NdotL = max(0.0f, dot(normal, wi));
        if (NdotL <= 0.0f)
        {
            pdf = 0.0f;
            throughputWeight = float3(0.0f, 0.0f, 0.0f);
            return wi;
        }
        
        // Compute specular PDF for MIS
        float3 h = normalize(wo + wi);
        float NdotH = max(0.0f, dot(normal, h));
        float VdotH = max(0.0f, dot(wo, h));
        float D = D_GGX(NdotH, alpha);
        float specPdf = D * NdotH / (4.0f * VdotH + EPSILON);
        
        float diffPdf = NdotL * INV_PI;
        pdf = specularWeight * specPdf + (1.0f - specularWeight) * diffPdf;
        
        // Evaluate full BRDF
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
