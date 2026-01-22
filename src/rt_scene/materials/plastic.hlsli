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

// Evaluate smooth plastic BRDF (with nonlinear support)
// Uses a small minimum roughness instead of perfect mirror
float3 SmoothPlastic_EvaluateNonlinear(float3 diffuseColor, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal, bool nonlinear)
{
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    float eta = extIOR / intIOR;
    
    // Use minimum roughness for "smooth" plastic
    static const float MIN_PLASTIC_ROUGHNESS = 0.05f;
    float alpha = MIN_PLASTIC_ROUGHNESS * MIN_PLASTIC_ROUGHNESS;
    
    // Half-vector for specular
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float VdotH = max(0.0f, dot(wo, h));
    
    // Specular term (GGX microfacet with Fresnel)
    // Scale down specular to better match Mitsuba's look
    // Mitsuba's plastic has more matte appearance due to internal scattering
    float Fh = FresnelDielectric(VdotH, eta);
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float specularScale = 0.3f;  // Reduce specular intensity
    float3 specular = float3(Fh, Fh, Fh) * D * G * specularScale / (4.0f * NdotV * NdotL + EPSILON);
    
    // Diffuse component - boost to compensate for reduced specular
    float Fv = FresnelDielectric(NdotV, eta);
    float Fl = FresnelDielectric(NdotL, eta);
    
    // Compute effective diffuse color
    float3 effectiveDiffuse = diffuseColor;
    if (nonlinear)
    {
        float fdrInt = AverageFresnel(1.0f / eta);
        effectiveDiffuse = InternalDiffuseReflectance(diffuseColor, fdrInt);
    }
    
    // Reduce Fresnel transmission penalty at grazing angles
    float transmissionBoost = 1.0f + 0.5f * (1.0f - NdotV);  // Boost diffuse at grazing angles
    float3 diffuse = effectiveDiffuse * INV_PI * (1.0f - Fv * 0.5f) * (1.0f - Fl * 0.5f) * transmissionBoost;
    
    return specular + diffuse;
}

// Legacy version without nonlinear parameter
float3 SmoothPlastic_Evaluate(float3 diffuseColor, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    return SmoothPlastic_EvaluateNonlinear(diffuseColor, intIOR, extIOR, wo, wi, normal, false);
}

// Sample smooth plastic (with nonlinear support)
// Note: We use a small minimum roughness to avoid perfectly mirror-like reflections
float3 SmoothPlastic_SampleNonlinear(float3 diffuseColor, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, bool nonlinear, out float3 throughputWeight, out float pdf)
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
    
    // Use a small roughness for "smooth" plastic to avoid harsh mirror reflections
    static const float MIN_PLASTIC_ROUGHNESS = 0.05f;
    float alpha = MIN_PLASTIC_ROUGHNESS * MIN_PLASTIC_ROUGHNESS;
    
    // Compute effective diffuse color for nonlinear mode
    float3 effectiveDiffuse = diffuseColor;
    if (nonlinear)
    {
        float fdrInt = AverageFresnel(1.0f / eta);
        effectiveDiffuse = InternalDiffuseReflectance(diffuseColor, fdrInt);
    }
    
    // Choose between specular reflection and diffuse based on estimated contribution
    // Heavily bias towards diffuse to match the reduced specular in evaluation
    float diffuseLum = Luminance(effectiveDiffuse);
    float specContrib = F * 0.3f;  // Match reduced specular scale
    float diffContrib = diffuseLum * (1.0f - F * 0.5f);  // Match boosted diffuse
    float totalWeight = specContrib + diffContrib;
    // Clamp specular probability to max 15% for more matte appearance
    float probSpec = (totalWeight > EPSILON) ? min(specContrib / totalWeight, 0.15f) : 0.05f;
    
    if (u.x < probSpec)
    {
        // Sample specular lobe with GGX
        float2 u2 = float2(u.x / probSpec, u.y);
        float3 localH = SampleGGX(u2, alpha);
        float3 h = LocalToWorld(localH, normal);
        float3 wi = reflect(-wo, h);
        
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
        float specPdf = D * NdotH / (4.0f * VdotH + EPSILON);
        float diffPdf = NdotL * INV_PI;
        pdf = probSpec * specPdf + (1.0f - probSpec) * diffPdf;
        
        // Specular weight (white for dielectric)
        float Fh = FresnelDielectric(VdotH, eta);
        float3 specular = float3(Fh, Fh, Fh) * D * G / (4.0f * NdotV * NdotL + EPSILON);
        float3 diffuse = effectiveDiffuse * INV_PI * (1.0f - F) * (1.0f - FresnelDielectric(NdotL, eta));
        throughputWeight = (specular + diffuse) * NdotL / (pdf + EPSILON);
        return wi;
    }
    else
    {
        // Sample diffuse lobe
        float2 u2 = float2((u.x - probSpec) / (1.0f - probSpec), u.y);
        float diffPdfOnly;
        float3 wi = Diffuse_Sample(normal, u2, diffPdfOnly);
        
        float NdotL = max(0.0f, dot(normal, wi));
        if (NdotL <= 0.0f)
        {
            pdf = 0.0f;
            throughputWeight = float3(0.0f, 0.0f, 0.0f);
            return wi;
        }
        
        // Compute MIS PDF
        float3 h = normalize(wo + wi);
        float NdotH = max(0.0f, dot(normal, h));
        float VdotH = max(0.0f, dot(wo, h));
        float specPdf = D_GGX(NdotH, alpha) * NdotH / (4.0f * VdotH + EPSILON);
        float diffPdf = NdotL * INV_PI;
        pdf = probSpec * specPdf + (1.0f - probSpec) * diffPdf;
        
        // Full BRDF evaluation
        float D = D_GGX(NdotH, alpha);
        float G = G_SmithGGX(NdotV, NdotL, alpha);
        float Fh = FresnelDielectric(VdotH, eta);
        float3 specular = float3(Fh, Fh, Fh) * D * G / (4.0f * NdotV * NdotL + EPSILON);
        float Fl = FresnelDielectric(NdotL, eta);
        float3 diffuse = effectiveDiffuse * INV_PI * (1.0f - F) * (1.0f - Fl);
        throughputWeight = (specular + diffuse) * NdotL / (pdf + EPSILON);
        return wi;
    }
}

// Legacy version without nonlinear parameter
float3 SmoothPlastic_Sample(float3 diffuseColor, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    return SmoothPlastic_SampleNonlinear(diffuseColor, intIOR, extIOR, wo, normal, u, false, throughputWeight, pdf);
}

// ============================================================================
// Rough Plastic (GGX Microfacet Coating) - Mitsuba3 compatible
// 
// More physically accurate model with rough dielectric coating over diffuse.
// Based on Mitsuba3's roughplastic implementation.
// ============================================================================

// Evaluate rough plastic BRDF (Mitsuba3 compatible)
// nonlinear parameter enables internal scattering compensation
float3 RoughPlastic_EvaluateNonlinear(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal, bool nonlinear)
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
    // Scale down specular to better match Mitsuba's matte wood appearance
    float D = D_GGX(NdotH, alpha);
    float G = G_SmithGGX(NdotV, NdotL, alpha);
    float specularScale = 0.3f;  // Reduce specular intensity for more matte look
    float3 specular = float3(F_spec, F_spec, F_spec) * D * G * specularScale / (4.0f * NdotV * NdotL + EPSILON);
    
    // Fresnel transmission factors
    float Fv = FresnelDielectric(NdotV, eta);
    float Fl = FresnelDielectric(NdotL, eta);
    
    // Compute effective diffuse color
    float3 effectiveDiffuse = diffuseColor;
    if (nonlinear)
    {
        // Nonlinear mode: account for multiple internal reflections
        // This makes the diffuse appear more saturated and brighter
        float fdrInt = AverageFresnel(1.0f / eta);
        effectiveDiffuse = InternalDiffuseReflectance(diffuseColor, fdrInt);
    }
    
    // Diffuse term with Fresnel transmission - reduce Fresnel penalty
    float transmissionBoost = 1.0f + 0.5f * (1.0f - NdotV);
    float3 diffuse = effectiveDiffuse * INV_PI * (1.0f - Fv * 0.5f) * (1.0f - Fl * 0.5f) * transmissionBoost;
    
    return specular + diffuse;
}

// Legacy version without nonlinear parameter
float3 RoughPlastic_Evaluate(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    return RoughPlastic_EvaluateNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal, false);
}

// Sample rough plastic (Mitsuba3 compatible)
// Uses MIS between specular and diffuse lobes
float3 RoughPlastic_SampleNonlinear(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, bool nonlinear, out float3 throughputWeight, out float pdf)
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
    
    // For nonlinear mode, compute effective diffuse color
    float3 effectiveDiffuse = diffuseColor;
    if (nonlinear)
    {
        float fdrInt = AverageFresnel(1.0f / eta);
        effectiveDiffuse = InternalDiffuseReflectance(diffuseColor, fdrInt);
    }
    
    // Compute specular sampling weight based on Fresnel and material properties
    float F = FresnelDielectric(NdotV, eta);
    
    // For dielectric plastic, heavily favor diffuse sampling
    // Match the reduced specular contribution in the evaluation function
    float diffuseLum = Luminance(effectiveDiffuse);
    float specContrib = F * 0.3f;  // Match reduced specular scale
    float diffContrib = diffuseLum * (1.0f - F * 0.5f);  // Match boosted diffuse
    float totalWeight = specContrib + diffContrib;
    
    if (totalWeight < EPSILON)
    {
        pdf = 0.0f;
        throughputWeight = float3(0.0f, 0.0f, 0.0f);
        return float3(0.0f, 0.0f, 1.0f);
    }
    
    // Clamp specular probability to max 15% for more matte appearance
    float probSpec = min(specContrib / totalWeight, 0.15f);
    
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
        float3 brdf = RoughPlastic_EvaluateNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal, nonlinear);
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
        float3 brdf = RoughPlastic_EvaluateNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal, nonlinear);
        throughputWeight = brdf * NdotL / (pdf + EPSILON);
    }
    
    return wi;
}

// Legacy version without nonlinear parameter
float3 RoughPlastic_Sample(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    return RoughPlastic_SampleNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, normal, u, false, throughputWeight, pdf);
}

// ============================================================================
// Combined Plastic Interface (with nonlinear support)
// ============================================================================

float3 Plastic_EvaluateNonlinear(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal, bool nonlinear)
{
    if (roughness < 0.01f)
        return SmoothPlastic_EvaluateNonlinear(diffuseColor, intIOR, extIOR, wo, wi, normal, nonlinear);
    else
        return RoughPlastic_EvaluateNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal, nonlinear);
}

float3 Plastic_SampleNonlinear(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, bool nonlinear, out float3 throughputWeight, out float pdf)
{
    if (roughness < 0.01f)
        return SmoothPlastic_SampleNonlinear(diffuseColor, intIOR, extIOR, wo, normal, u, nonlinear, throughputWeight, pdf);
    else
        return RoughPlastic_SampleNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, normal, u, nonlinear, throughputWeight, pdf);
}

// Legacy versions without nonlinear parameter
float3 Plastic_Evaluate(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    return Plastic_EvaluateNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, wi, normal, false);
}

float3 Plastic_Sample(float3 diffuseColor, float roughness, float intIOR, float extIOR, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf)
{
    return Plastic_SampleNonlinear(diffuseColor, roughness, intIOR, extIOR, wo, normal, u, false, throughputWeight, pdf);
}

#endif // MATERIALS_PLASTIC_HLSLI
