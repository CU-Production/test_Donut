// ============================================================================
// thindielectric.hlsli - Thin Dielectric BSDF (Mitsuba3 compatible)
// 
// Models a thin dielectric sheet embedded in another medium.
// Light passes through without angular deflection (no refraction offset).
// Useful for windows, soap bubbles, etc.
//
// Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_THINDIELECTRIC_HLSLI
#define MATERIALS_THINDIELECTRIC_HLSLI

#include "common.hlsli"

// ============================================================================
// Thin Dielectric
// 
// Models a thin layer where multiple internal reflections are accounted for.
// Unlike regular dielectric, transmitted rays don't change direction.
// ============================================================================

// Evaluate thin dielectric - returns 0 since both reflection and transmission
// are delta distributions
float3 ThinDielectric_Evaluate(float intIOR, float extIOR, float3 wo, float3 wi, float3 normal)
{
    // Delta distribution - no contribution except at perfect reflection/transmission
    return float3(0.0f, 0.0f, 0.0f);
}

// Sample thin dielectric
// Accounts for multiple internal reflections: R, TRT, TR^3T, ... for reflection
// and TT, TR^2T, TR^4T, ... for transmission
float3 ThinDielectric_Sample(float intIOR, float extIOR, float3 specularReflectance, float3 specularTransmittance,
                              float3 wo, float3 normal, float2 u, 
                              out float3 throughputWeight, out float pdf, out bool isTransmitted)
{
    float cosTheta = dot(wo, normal);
    
    // Compute Fresnel reflectance for a single interface
    float eta = extIOR / intIOR;
    float R = FresnelDielectric(abs(cosTheta), eta);
    
    // For thin dielectric, we account for infinite internal reflections:
    // Total reflectance = R + T*R*T + T*R^3*T + ... = R + T^2*R/(1-R^2)
    //                   = R * (1 + T^2/(1-R^2)) when R < 1
    // Simplified: R_total = 2*R / (1 + R)
    // And T_total = 1 - R_total = (1 - R) / (1 + R) * 2 = 2*(1-R)/(1+R)
    
    float R2 = R * R;
    float T = 1.0f - R;
    float T2 = T * T;
    
    // Total reflectance accounting for multiple bounces
    float R_total = R + T2 * R / (1.0f - R2 + EPSILON);
    float T_total = T2 / (1.0f - R2 + EPSILON);
    
    // Normalize
    float sum = R_total + T_total;
    R_total /= sum;
    T_total /= sum;
    
    if (u.x < R_total)
    {
        // Reflection
        float3 wi = reflect(-wo, normal);
        pdf = R_total;
        throughputWeight = specularReflectance;
        isTransmitted = false;
        return wi;
    }
    else
    {
        // Transmission - direction unchanged (thin approximation)
        float3 wi = -wo;  // Pass through without deflection
        pdf = T_total;
        throughputWeight = specularTransmittance;
        isTransmitted = true;
        return wi;
    }
}

// Simplified version without specular modulation
float3 ThinDielectric_Sample(float intIOR, float extIOR, float3 wo, float3 normal, float2 u,
                              out float3 throughputWeight, out float pdf, out bool isTransmitted)
{
    return ThinDielectric_Sample(intIOR, extIOR, 
                                  float3(1.0f, 1.0f, 1.0f), float3(1.0f, 1.0f, 1.0f),
                                  wo, normal, u, throughputWeight, pdf, isTransmitted);
}

#endif // MATERIALS_THINDIELECTRIC_HLSLI
