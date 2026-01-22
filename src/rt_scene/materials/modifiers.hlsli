// ============================================================================
// modifiers.hlsli - BSDF Modifiers (Mitsuba3 compatible)
// 
// Contains material modifiers and adapters:
// - blendbsdf: Linear blend between two materials
// - mask: Opacity mask for transparency
// - null: Completely transparent material
//
// Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_MODIFIERS_HLSLI
#define MATERIALS_MODIFIERS_HLSLI

#include "common.hlsli"

// ============================================================================
// Blend BSDF
// 
// Linearly interpolates between two materials based on a weight.
// weight = 0: first material only
// weight = 1: second material only
// ============================================================================

// Note: Actual blending is done at a higher level by sampling one of the
// two materials and weighting appropriately. This function helps compute
// which material to sample.

// Returns true if we should sample the second material
bool Blend_ChooseMaterial(float weight, float u)
{
    return u >= (1.0f - weight);
}

// Get the PDF correction factor for blending
float Blend_PDFWeight(float weight, bool isSecondMaterial)
{
    return isSecondMaterial ? weight : (1.0f - weight);
}

// ============================================================================
// Mask (Opacity)
// 
// Implements transparency by blending between the base material and
// a null (pass-through) material.
// opacity = 1: fully opaque (base material)
// opacity = 0: fully transparent (null material)
// ============================================================================

// Sample mask material
// Returns true if the ray should pass through (transparent)
bool Mask_ShouldPassThrough(float opacity, float u)
{
    return u >= opacity;
}

// Get the pass-through direction (unchanged)
float3 Mask_PassThrough(float3 wo)
{
    return -wo;  // Continue in the same direction
}

// ============================================================================
// Null Material
// 
// Completely invisible - rays pass through unchanged.
// Used for boundaries of participating media.
// ============================================================================

float3 Null_Sample(float3 wo, out float3 throughputWeight, out float pdf)
{
    // Ray continues unchanged
    pdf = 1.0f;
    throughputWeight = float3(1.0f, 1.0f, 1.0f);
    return -wo;
}

float3 Null_Evaluate(float3 wo, float3 wi, float3 normal)
{
    // Delta distribution - no contribution in evaluation
    return float3(0.0f, 0.0f, 0.0f);
}

#endif // MATERIALS_MODIFIERS_HLSLI
