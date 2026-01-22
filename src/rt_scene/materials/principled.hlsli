// ============================================================================
// principled.hlsli - Disney Principled BSDF (Mitsuba3 compatible)
// 
// A versatile material model based on Disney's Principled BRDF.
// Can represent metals, plastics, glass, and many other materials.
//
// Reference: 
// - Burley 2012: "Physically Based Shading at Disney"
// - Burley 2015: "Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering"
// - https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_PRINCIPLED_HLSLI
#define MATERIALS_PRINCIPLED_HLSLI

#include "common.hlsli"

// ============================================================================
// Principled BSDF Parameters
// ============================================================================
struct PrincipledParams
{
    float3 baseColor;      // Base color of the material
    float roughness;       // Surface roughness [0, 1]
    float metallic;        // Metallic factor [0, 1]
    float specular;        // Specular intensity (controls F0 for dielectrics)
    float specTint;        // Tint specular towards base color
    float anisotropic;     // Anisotropy factor [0, 1]
    float sheen;           // Sheen intensity
    float sheenTint;       // Tint sheen towards base color
    float clearcoat;       // Clearcoat intensity
    float clearcoatGloss;  // Clearcoat glossiness
    float specTrans;       // Specular transmission (glass-like)
    float eta;             // Index of refraction
};

// ============================================================================
// Helper Functions
// ============================================================================

// Schlick Fresnel approximation
float3 SchlickFresnel(float cosTheta, float3 F0)
{
    float x = 1.0f - cosTheta;
    float x2 = x * x;
    float x5 = x2 * x2 * x;
    return F0 + (1.0f - F0) * x5;
}

// GTR1 (Generalized Trowbridge-Reitz) for clearcoat
float GTR1(float NdotH, float a)
{
    if (a >= 1.0f) return INV_PI;
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return (a2 - 1.0f) / (PI * log(a2) * t + EPSILON);
}

// GTR2 (GGX) distribution
float GTR2(float NdotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
    return a2 / (PI * t * t + EPSILON);
}

// Smith G1 for GGX
float SmithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1.0f / (NdotV + sqrt(a + b - a * b) + EPSILON);
}

// Compute F0 for dielectric from specular parameter
float3 ComputeF0(float specular, float3 baseColor, float metallic, float specTint)
{
    // F0 for dielectric (specular = 0.5 corresponds to IOR 1.5, F0 = 0.04)
    float dielectricF0 = 0.08f * specular;
    
    // Tinted specular
    float3 tintedF0 = dielectricF0 * lerp(float3(1.0f, 1.0f, 1.0f), 
                                          baseColor / (Luminance(baseColor) + EPSILON), 
                                          specTint);
    
    // Metallic uses base color as F0
    return lerp(tintedF0, baseColor, metallic);
}

// ============================================================================
// Principled BSDF Evaluation
// ============================================================================
float3 Principled_Evaluate(PrincipledParams params, float3 wo, float3 wi, float3 normal)
{
    float NdotV = dot(normal, wo);
    float NdotL = dot(normal, wi);
    
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return float3(0.0f, 0.0f, 0.0f);
    
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float LdotH = max(0.0f, dot(wi, h));
    
    float alpha = max(0.001f, params.roughness * params.roughness);
    
    // Compute F0
    float3 F0 = ComputeF0(params.specular, params.baseColor, params.metallic, params.specTint);
    
    // ========== Diffuse ==========
    // Disney diffuse with roughness-dependent Fresnel
    float FL = pow(1.0f - NdotL, 5.0f);
    float FV = pow(1.0f - NdotV, 5.0f);
    float Fd90 = 0.5f + 2.0f * LdotH * LdotH * params.roughness;
    float Fd = (1.0f + (Fd90 - 1.0f) * FL) * (1.0f + (Fd90 - 1.0f) * FV);
    
    float3 diffuse = params.baseColor * INV_PI * Fd * (1.0f - params.metallic) * (1.0f - params.specTrans);
    
    // ========== Specular ==========
    float D = GTR2(NdotH, alpha);
    float3 F = SchlickFresnel(LdotH, F0);
    float G = SmithG_GGX(NdotV, alpha) * SmithG_GGX(NdotL, alpha);
    
    float3 specular = D * F * G;
    
    // ========== Sheen ==========
    float3 sheenColor = lerp(float3(1.0f, 1.0f, 1.0f), 
                              params.baseColor / (Luminance(params.baseColor) + EPSILON), 
                              params.sheenTint);
    float FH = pow(1.0f - LdotH, 5.0f);
    float3 sheen = FH * params.sheen * sheenColor * (1.0f - params.metallic);
    
    // ========== Clearcoat ==========
    float Dc = GTR1(NdotH, lerp(0.1f, 0.001f, params.clearcoatGloss));
    float Fc = lerp(0.04f, 1.0f, FH);
    float Gc = SmithG_GGX(NdotV, 0.25f) * SmithG_GGX(NdotL, 0.25f);
    float3 clearcoat = float3(0.25f * params.clearcoat * Dc * Fc * Gc, 
                               0.25f * params.clearcoat * Dc * Fc * Gc, 
                               0.25f * params.clearcoat * Dc * Fc * Gc);
    
    return diffuse + specular + sheen + clearcoat;
}

// ============================================================================
// Principled BSDF Sampling
// ============================================================================
float3 Principled_Sample(PrincipledParams params, float3 wo, float3 normal, float2 u,
                          out float3 throughputWeight, out float pdf, out bool isTransmitted)
{
    isTransmitted = false;
    
    float NdotV = dot(normal, wo);
    if (NdotV <= 0.0f)
    {
        pdf = 0.0f;
        throughputWeight = float3(0.0f, 0.0f, 0.0f);
        return float3(0.0f, 0.0f, 0.0f);
    }
    
    float alpha = max(0.001f, params.roughness * params.roughness);
    
    // Compute sampling weights for different lobes
    float diffuseWeight = (1.0f - params.metallic) * (1.0f - params.specTrans);
    float specularWeight = 1.0f;
    float clearcoatWeight = params.clearcoat * 0.25f;
    float transmissionWeight = params.specTrans * (1.0f - params.metallic);
    
    float totalWeight = diffuseWeight + specularWeight + clearcoatWeight + transmissionWeight;
    diffuseWeight /= totalWeight;
    specularWeight /= totalWeight;
    clearcoatWeight /= totalWeight;
    transmissionWeight /= totalWeight;
    
    float3 wi;
    float3 F0 = ComputeF0(params.specular, params.baseColor, params.metallic, params.specTint);
    
    float r = u.x;
    
    if (r < diffuseWeight)
    {
        // Sample diffuse
        float2 u2 = float2(r / diffuseWeight, u.y);
        wi = LocalToWorld(CosineSampleHemisphere(u2), normal);
    }
    else if (r < diffuseWeight + specularWeight)
    {
        // Sample specular (GGX)
        float2 u2 = float2((r - diffuseWeight) / specularWeight, u.y);
        float3 h = LocalToWorld(SampleGGX(u2, alpha), normal);
        wi = reflect(-wo, h);
    }
    else if (r < diffuseWeight + specularWeight + clearcoatWeight)
    {
        // Sample clearcoat (GTR1)
        float2 u2 = float2((r - diffuseWeight - specularWeight) / clearcoatWeight, u.y);
        float ccAlpha = lerp(0.1f, 0.001f, params.clearcoatGloss);
        float3 h = LocalToWorld(SampleGGX(u2, ccAlpha), normal);
        wi = reflect(-wo, h);
    }
    else
    {
        // Sample transmission
        float2 u2 = float2((r - diffuseWeight - specularWeight - clearcoatWeight) / transmissionWeight, u.y);
        float3 h = LocalToWorld(SampleGGX(u2, alpha), normal);
        
        float eta = params.eta;
        float cosI = dot(wo, h);
        float sin2T = eta * eta * (1.0f - cosI * cosI);
        
        if (sin2T >= 1.0f)
        {
            // Total internal reflection
            wi = reflect(-wo, h);
        }
        else
        {
            float cosT = sqrt(1.0f - sin2T);
            wi = eta * (-wo) + (eta * cosI - cosT) * h;
            wi = normalize(wi);
            isTransmitted = true;
        }
    }
    
    float NdotL = dot(normal, wi);
    if (!isTransmitted && NdotL <= 0.0f)
    {
        pdf = 0.0f;
        throughputWeight = float3(0.0f, 0.0f, 0.0f);
        return wi;
    }
    
    // Evaluate BSDF
    float3 brdf = Principled_Evaluate(params, wo, wi, normal);
    
    // Compute PDF (MIS over all lobes)
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float VdotH = max(0.0f, dot(wo, h));
    
    float diffusePdf = max(0.0f, NdotL) * INV_PI;
    float specularPdf = GTR2(NdotH, alpha) * NdotH / (4.0f * VdotH + EPSILON);
    float ccAlpha = lerp(0.1f, 0.001f, params.clearcoatGloss);
    float clearcoatPdf = GTR1(NdotH, ccAlpha) * NdotH / (4.0f * VdotH + EPSILON);
    float transPdf = GTR2(NdotH, alpha) * NdotH / (4.0f * VdotH + EPSILON);
    
    pdf = diffuseWeight * diffusePdf + specularWeight * specularPdf + 
          clearcoatWeight * clearcoatPdf + transmissionWeight * transPdf;
    
    throughputWeight = brdf * abs(NdotL) / (pdf + EPSILON);
    
    return wi;
}

// ============================================================================
// Default Parameters
// ============================================================================
PrincipledParams Principled_DefaultParams()
{
    PrincipledParams params;
    params.baseColor = float3(0.5f, 0.5f, 0.5f);
    params.roughness = 0.5f;
    params.metallic = 0.0f;
    params.specular = 0.5f;
    params.specTint = 0.0f;
    params.anisotropic = 0.0f;
    params.sheen = 0.0f;
    params.sheenTint = 0.0f;
    params.clearcoat = 0.0f;
    params.clearcoatGloss = 0.0f;
    params.specTrans = 0.0f;
    params.eta = 1.5f;
    return params;
}

#endif // MATERIALS_PRINCIPLED_HLSLI
