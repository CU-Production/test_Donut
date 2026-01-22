// ============================================================================
// normalmap.hlsli - Normal/Bump Map Support (Mitsuba3 compatible)
// 
// Provides functions for perturbing surface normals based on:
// - Normal maps: RGB texture encoding XYZ normals in tangent space
// - Bump maps: Height field texture
//
// Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_NORMALMAP_HLSLI
#define MATERIALS_NORMALMAP_HLSLI

#include "common.hlsli"

// ============================================================================
// Normal Map
// 
// Perturbs the shading normal based on a normal map texture.
// Normal maps store normals in tangent space as RGB values [0,1]
// mapped from XYZ [-1,1] using: color = (normal + 1) / 2
// ============================================================================

// Decode normal from normal map RGB value
// Input: RGB value from normal map [0, 1]
// Output: Normal vector in tangent space [-1, 1]
float3 DecodeNormalMap(float3 normalMapValue)
{
    // Convert from [0,1] to [-1,1]
    return normalize(normalMapValue * 2.0f - 1.0f);
}

// Apply normal map to shading normal
// geometricNormal: The original surface normal
// tangent: Surface tangent vector
// bitangent: Surface bitangent vector (can be computed from normal x tangent)
// normalMapValue: RGB value sampled from normal map
float3 ApplyNormalMap(float3 geometricNormal, float3 tangent, float3 bitangent, float3 normalMapValue)
{
    // Decode tangent-space normal
    float3 tsNormal = DecodeNormalMap(normalMapValue);
    
    // Build TBN matrix (columns are tangent, bitangent, normal)
    // Transform from tangent space to world space
    float3 worldNormal = normalize(
        tsNormal.x * tangent +
        tsNormal.y * bitangent +
        tsNormal.z * geometricNormal
    );
    
    return worldNormal;
}

// Simplified version when only normal and up direction are available
// Constructs tangent frame from normal
float3 ApplyNormalMapSimple(float3 geometricNormal, float3 normalMapValue)
{
    // Build orthonormal basis around the normal
    float3 tangent, bitangent;
    BuildOrthonormalBasis(geometricNormal, tangent, bitangent);
    
    return ApplyNormalMap(geometricNormal, tangent, bitangent, normalMapValue);
}

// ============================================================================
// Bump Map
// 
// Perturbs the shading normal based on a height field gradient.
// The bump map stores height values, and the gradient is computed
// from finite differences.
// ============================================================================

// Compute perturbed normal from bump map gradients
// geometricNormal: The original surface normal  
// tangent: Surface tangent vector
// bitangent: Surface bitangent vector
// dhdx: Height derivative in tangent direction
// dhdy: Height derivative in bitangent direction
// scale: Bump intensity multiplier
float3 ApplyBumpMap(float3 geometricNormal, float3 tangent, float3 bitangent, 
                    float dhdx, float dhdy, float scale)
{
    // Perturb the normal based on height gradients
    float3 perturbedNormal = geometricNormal - scale * (dhdx * tangent + dhdy * bitangent);
    return normalize(perturbedNormal);
}

// Compute bump map gradients from height samples
// h00: Height at current texcoord
// h10: Height at texcoord + (du, 0)
// h01: Height at texcoord + (0, dv)
// du, dv: Texcoord step size
void ComputeBumpGradients(float h00, float h10, float h01, float du, float dv,
                          out float dhdx, out float dhdy)
{
    dhdx = (h10 - h00) / (du + EPSILON);
    dhdy = (h01 - h00) / (dv + EPSILON);
}

// ============================================================================
// Shading Normal Validation
// 
// Ensures perturbed normals are consistent with geometric normal
// to prevent light leaking artifacts.
// ============================================================================

// Flip shading normal if it's on the wrong side of the surface
// (as described in "Microfacet-based Normal Mapping for Robust MC Path Tracing")
float3 ValidateShadingNormal(float3 geometricNormal, float3 shadingNormal, float3 wo)
{
    // If wo is above geometric normal but below shading normal (or vice versa),
    // flip the shading normal
    float geoSide = dot(geometricNormal, wo);
    float shadeSide = dot(shadingNormal, wo);
    
    if (geoSide * shadeSide < 0.0f)
    {
        // Flip shading normal
        return reflect(shadingNormal, geometricNormal);
    }
    
    return shadingNormal;
}

// Softer validation that blends towards geometric normal
float3 ValidateShadingNormalSmooth(float3 geometricNormal, float3 shadingNormal, float3 wo)
{
    float geoSide = dot(geometricNormal, wo);
    float shadeSide = dot(shadingNormal, wo);
    
    if (geoSide * shadeSide < 0.0f)
    {
        // Blend towards geometric normal
        float blend = saturate(-shadeSide / (abs(geoSide) + abs(shadeSide) + EPSILON));
        return normalize(lerp(shadingNormal, geometricNormal, blend));
    }
    
    return shadingNormal;
}

#endif // MATERIALS_NORMALMAP_HLSLI
