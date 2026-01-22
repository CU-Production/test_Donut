// ============================================================================
// Material System Include
// ============================================================================
#include "materials/material.hlsli"

// ============================================================================
// Structures
// ============================================================================
struct GPUVertex
{
    float3 position;
    float pad0;
    float3 normal;
    float pad1;
    float2 texcoord;
    float2 pad2;
};

struct GPUMaterial
{
    float3 baseColor;
    float roughness;
    
    float3 eta;
    float metallic;
    
    float3 k;
    uint type;
    
    float intIOR;
    float extIOR;
    int baseColorTexIdx;   // -1 if no texture
    int roughnessTexIdx;   // -1 if no texture
    
    int normalTexIdx;      // -1 if no texture
    
    // Principled BSDF parameters
    float specular;        // Specular intensity
    float specTint;        // Tint specular towards base color
    float sheen;           // Sheen intensity
    
    float sheenTint;       // Tint sheen towards base color
    float clearcoat;       // Clearcoat intensity
    float clearcoatGloss;  // Clearcoat glossiness
    float specTrans;       // Specular transmission
    
    // Mask/Blend parameters
    float opacity;         // Opacity for mask material
    float blendWeight;     // Blend weight for blendbsdf
    float2 padding;
};

struct GPUInstance
{
    uint vertexOffset;
    uint indexOffset;
    uint materialIndex;
    uint isEmitter;
    float3 emission;
    float pad;
};

struct CameraConstants
{
    float4x4 viewInverse;
    float4x4 projInverse;
    float3 cameraPosition;
    uint frameIndex;
    uint samplesPerPixel;
    uint maxBounces;
    float envMapIntensity;
    uint hasEnvMap;
};

struct HitInfo
{
    float3 color;       // Used to pass normal
    float hitT;
    float3 emission;
    uint instanceID;
    float2 texcoord;    // Interpolated texture coordinates
    float2 padding;
};

struct ShadowHitInfo
{
    bool isHit;
    float3 padding;
};

struct Attributes
{
    float2 barycentrics;
};

// ============================================================================
// Resources
// ============================================================================
RaytracingAccelerationStructure SceneBVH       : register(t0);
StructuredBuffer<GPUVertex> Vertices           : register(t1);
StructuredBuffer<uint> Indices                 : register(t2);
StructuredBuffer<GPUMaterial> Materials        : register(t3);
StructuredBuffer<GPUInstance> Instances        : register(t4);
Texture2D<float4> EnvironmentMap               : register(t5);

// Material textures array - uses unbounded descriptor array
// Maximum 64 material textures supported
Texture2D<float4> MaterialTextures[64]         : register(t6);

RWTexture2D<float4> OutputTexture              : register(u0);
RWTexture2D<float4> AccumulationTexture        : register(u1);

cbuffer CameraBuffer : register(b0)
{
    CameraConstants Camera;
};

SamplerState LinearSampler                     : register(s0);

// ============================================================================
// Environment Map Sampling
// ============================================================================
float3 SampleEnvironmentMap(float3 direction)
{
    // Convert direction to spherical coordinates (equirectangular mapping)
    // Theta: azimuth angle (around Y axis)
    // Phi: elevation angle (from Y axis)
    float theta = atan2(direction.x, direction.z);
    float phi = asin(clamp(direction.y, -1.0f, 1.0f));
    
    // Convert to UV coordinates [0, 1]
    float u = (theta + PI) / (2.0f * PI);
    float v = (phi + PI * 0.5f) / PI;
    
    // Sample the environment map
    float3 envColor = EnvironmentMap.SampleLevel(LinearSampler, float2(u, v), 0).rgb;
    
    return envColor * Camera.envMapIntensity;
}

// ============================================================================
// Random Number Generator (PCG)
// ============================================================================
struct RNGState
{
    uint state;
};

uint PCGHash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float RandomFloat(inout RNGState rng)
{
    rng.state = PCGHash(rng.state);
    return rng.state / 4294967295.0f;
}

float2 RandomFloat2(inout RNGState rng)
{
    return float2(RandomFloat(rng), RandomFloat(rng));
}

// ============================================================================
// Material System Helper Functions
// ============================================================================

// Convert GPUMaterial to MaterialParams for the material system
MaterialParams GPUMaterialToParams(GPUMaterial gpuMat)
{
    MaterialParams params;
    params.baseColor = gpuMat.baseColor;
    params.roughness = gpuMat.roughness;
    params.eta = gpuMat.eta;
    params.metallic = gpuMat.metallic;
    params.k = gpuMat.k;
    params.type = gpuMat.type;
    params.intIOR = gpuMat.intIOR;
    params.extIOR = gpuMat.extIOR;
    
    // Principled BSDF parameters
    params.specular = gpuMat.specular;
    params.specTint = gpuMat.specTint;
    params.sheen = gpuMat.sheen;
    params.sheenTint = gpuMat.sheenTint;
    params.clearcoat = gpuMat.clearcoat;
    params.clearcoatGloss = gpuMat.clearcoatGloss;
    params.specTrans = gpuMat.specTrans;
    
    // Mask/Blend parameters
    params.opacity = gpuMat.opacity;
    params.blendWeight = gpuMat.blendWeight;
    
    return params;
}

// Legacy wrapper functions for compatibility
float3 SampleBRDF(GPUMaterial mat, float3 wo, float3 normal, float2 u, out float pdf)
{
    MaterialParams params = GPUMaterialToParams(mat);
    float3 throughputWeight;
    bool isRefracted;
    float3 wi = Material_Sample(params, wo, normal, u, throughputWeight, pdf, isRefracted);
    return wi;
}

float3 EvaluateBRDF(GPUMaterial mat, float3 wo, float3 wi, float3 normal)
{
    MaterialParams params = GPUMaterialToParams(mat);
    return Material_Evaluate(params, wo, wi, normal);
}

// Sample BRDF and return throughput weight directly (more efficient)
float3 SampleBRDFWithWeight(GPUMaterial mat, float3 wo, float3 normal, float2 u, out float3 throughputWeight, out float pdf, out bool isRefracted)
{
    MaterialParams params = GPUMaterialToParams(mat);
    return Material_Sample(params, wo, normal, u, throughputWeight, pdf, isRefracted);
}

// ============================================================================
// Intersection Helpers
// ============================================================================
float3 GetInterpolatedNormal(uint instanceID, uint primitiveIndex, float2 barycentrics)
{
    GPUInstance instance = Instances[instanceID];
    
    // Get triangle vertex indices (global indices into vertex buffer)
    uint i0 = Indices[instance.indexOffset + primitiveIndex * 3 + 0];
    uint i1 = Indices[instance.indexOffset + primitiveIndex * 3 + 1];
    uint i2 = Indices[instance.indexOffset + primitiveIndex * 3 + 2];
    
    // Fetch vertex normals
    float3 n0 = Vertices[i0].normal;
    float3 n1 = Vertices[i1].normal;
    float3 n2 = Vertices[i2].normal;
    
    // Barycentric interpolation
    float3 normal = n0 * (1.0f - barycentrics.x - barycentrics.y) +
                    n1 * barycentrics.x +
                    n2 * barycentrics.y;
    
    return normalize(normal);
}

float2 GetInterpolatedTexcoord(uint instanceID, uint primitiveIndex, float2 barycentrics)
{
    GPUInstance instance = Instances[instanceID];
    
    // Get triangle vertex indices (global indices into vertex buffer)
    uint i0 = Indices[instance.indexOffset + primitiveIndex * 3 + 0];
    uint i1 = Indices[instance.indexOffset + primitiveIndex * 3 + 1];
    uint i2 = Indices[instance.indexOffset + primitiveIndex * 3 + 2];
    
    // Fetch vertex texcoords
    float2 t0 = Vertices[i0].texcoord;
    float2 t1 = Vertices[i1].texcoord;
    float2 t2 = Vertices[i2].texcoord;
    
    // Barycentric interpolation
    float2 texcoord = t0 * (1.0f - barycentrics.x - barycentrics.y) +
                      t1 * barycentrics.x +
                      t2 * barycentrics.y;
    
    return texcoord;
}

float3 GetHitPosition(float3 rayOrigin, float3 rayDirection, float hitT)
{
    return rayOrigin + rayDirection * hitT;
}

// Sample material texture with index validation
float4 SampleMaterialTexture(int textureIndex, float2 texcoord)
{
    if (textureIndex < 0 || textureIndex >= 64)
        return float4(1.0f, 1.0f, 1.0f, 1.0f);
    
    return MaterialTextures[NonUniformResourceIndex(textureIndex)].SampleLevel(LinearSampler, texcoord, 0);
}

// Get material with texture sampling
GPUMaterial GetMaterialWithTextures(uint materialIndex, float2 texcoord)
{
    GPUMaterial mat = Materials[materialIndex];
    
    // Sample base color texture if available
    if (mat.baseColorTexIdx >= 0)
    {
        float4 texColor = SampleMaterialTexture(mat.baseColorTexIdx, texcoord);
        mat.baseColor = texColor.rgb;
    }
    
    // Sample roughness texture if available
    if (mat.roughnessTexIdx >= 0)
    {
        float4 texRoughness = SampleMaterialTexture(mat.roughnessTexIdx, texcoord);
        mat.roughness = texRoughness.r;  // Assume roughness in red channel
    }
    
    return mat;
}

// Apply normal map to geometric normal
float3 ApplyNormalMapFromTexture(float3 geometricNormal, int normalTexIdx, float2 texcoord)
{
    if (normalTexIdx < 0)
        return geometricNormal;
    
    float4 normalMapSample = SampleMaterialTexture(normalTexIdx, texcoord);
    
    // Decode tangent-space normal from [0,1] to [-1,1]
    float3 tsNormal = normalMapSample.xyz * 2.0f - 1.0f;
    tsNormal = normalize(tsNormal);
    
    // Build orthonormal basis around geometric normal
    float3 tangent, bitangent;
    BuildOrthonormalBasis(geometricNormal, tangent, bitangent);
    
    // Transform from tangent space to world space
    float3 worldNormal = normalize(
        tsNormal.x * tangent +
        tsNormal.y * bitangent +
        tsNormal.z * geometricNormal
    );
    
    return worldNormal;
}

// Get material with texture sampling, also applies normal map
float3 GetShadingNormal(float3 geometricNormal, GPUMaterial mat, float2 texcoord)
{
    return ApplyNormalMapFromTexture(geometricNormal, mat.normalTexIdx, texcoord);
}

// ============================================================================
// Ray Generation Shader
// ============================================================================
[shader("raygeneration")]
void RayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;
    
    // Initialize RNG
    RNGState rng;
    rng.state = launchIndex.x + launchIndex.y * launchDim.x + Camera.frameIndex * launchDim.x * launchDim.y;
    
    // Generate camera ray with jitter for anti-aliasing
    float2 jitter = RandomFloat2(rng) - 0.5f;
    float2 pixelCenter = float2(launchIndex) + 0.5f + jitter;
    float2 ndc = (pixelCenter / float2(launchDim)) * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    
    float4 rayClip = float4(ndc, 1.0f, 1.0f);
    // Use column-vector multiplication (M * v) - matrices are stored column-major
    float4 rayView = mul(Camera.projInverse, rayClip);
    rayView /= rayView.w;
    
    float4 rayWorld = mul(Camera.viewInverse, float4(rayView.xyz, 0.0f));
    float3 rayDir = normalize(rayWorld.xyz);
    
    float3 rayOrigin = Camera.cameraPosition;
    
    // DEBUG MODE: Set to 1 to visualize material base colors directly, 0 for normal rendering
    #define DEBUG_MATERIAL_COLOR 0
    
    #if DEBUG_MATERIAL_COLOR
    {
        // Debug: trace one ray and output material base color directly
        RayDesc debugRay;
        debugRay.Origin = rayOrigin;
        debugRay.Direction = rayDir;
        debugRay.TMin = RAY_EPSILON;
        debugRay.TMax = 10000.0f;
        
        HitInfo debugPayload;
        debugPayload.hitT = -1.0f;
        debugPayload.instanceID = 0xFFFFFFFF;
        
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 2, 0, debugRay, debugPayload);
        
        float3 debugColor = float3(0.0f, 0.0f, 0.0f);
        if (debugPayload.hitT > 0.0f)
        {
            GPUInstance instance = Instances[debugPayload.instanceID];
            GPUMaterial mat = GetMaterialWithTextures(instance.materialIndex, debugPayload.texcoord);
            
            // Output base color directly (gamma corrected for display)
            debugColor = pow(mat.baseColor, 1.0f / 2.2f);
            
            // If emitter, show emission (scaled down for visibility)
            if (instance.isEmitter != 0)
            {
                debugColor = float3(1.0f, 0.9f, 0.7f);  // Mark emitters with warm white
            }
        }
        
        OutputTexture[launchIndex] = float4(debugColor, 1.0f);
        AccumulationTexture[launchIndex] = float4(debugColor, 1.0f);
        return;
    }
    #endif
    
    // Path tracing
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin = RAY_EPSILON;
    ray.TMax = 10000.0f;
    
    for (uint bounce = 0; bounce < Camera.maxBounces; bounce++)
    {
        HitInfo payload;
        payload.color = float3(0.0f, 0.0f, 0.0f);
        payload.hitT = -1.0f;
        payload.emission = float3(0.0f, 0.0f, 0.0f);
        payload.instanceID = 0xFFFFFFFF;
        
        TraceRay(
            SceneBVH,
            RAY_FLAG_NONE,
            0xFF,
            0,          // Hit group index
            2,          // Number of hit groups
            0,          // Miss shader index
            ray,
            payload);
        
        if (payload.hitT < 0.0f)
        {
            // Miss - sample environment map or use procedural sky
            float3 envColor;
            if (Camera.hasEnvMap != 0)
            {
                envColor = SampleEnvironmentMap(ray.Direction);
            }
            else
            {
                // Procedural sky fallback
                envColor = lerp(float3(0.5f, 0.7f, 1.0f), float3(0.8f, 0.9f, 1.0f), 
                               saturate(ray.Direction.y * 0.5f + 0.5f)) * 0.3f;
            }
            radiance += throughput * envColor;
            break;
        }
        
        // Add emission
        radiance += throughput * payload.emission;
        
        // Get hit information
        float3 hitPos = GetHitPosition(ray.Origin, ray.Direction, payload.hitT);
        GPUInstance instance = Instances[payload.instanceID];
        
        // Get material with texture sampling
        GPUMaterial mat = GetMaterialWithTextures(instance.materialIndex, payload.texcoord);
        
        // Get interpolated geometric normal (stored in payload.color for simplicity)
        float3 geometricNormal = payload.color;
        
        // Apply normal map to get shading normal
        float3 normal = GetShadingNormal(geometricNormal, mat, payload.texcoord);
        
        // Make sure normal faces the ray
        if (dot(normal, ray.Direction) > 0.0f)
        {
            normal = -normal;
        }
        
        // Sample new direction using the material system
        float3 wo = -ray.Direction;
        float3 sampleWeight;
        float pdf;
        bool isRefracted;
        float3 wi = SampleBRDFWithWeight(mat, wo, normal, RandomFloat2(rng), sampleWeight, pdf, isRefracted);
        
        // For refraction, the ray may go through the surface
        float NdotL = dot(normal, wi);
        if (pdf < EPSILON || (!isRefracted && NdotL <= 0.0f))
        {
            break;
        }
        
        // Update throughput using the pre-computed sample weight
        // sampleWeight already includes BRDF * cos / pdf for importance sampled materials
        // So we should NOT multiply by cosTheta again
        throughput *= sampleWeight;
        
        // Check for invalid throughput (NaN or Inf)
        if (any(isnan(throughput)) || any(isinf(throughput)))
        {
            break;
        }
        
        // Clamp throughput to prevent fireflies
        float maxThroughput = max(throughput.x, max(throughput.y, throughput.z));
        if (maxThroughput > 100.0f)
        {
            throughput *= 100.0f / maxThroughput;
        }
        
        // Russian roulette
        if (bounce > 3)
        {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if (p < EPSILON || RandomFloat(rng) > p)
            {
                break;
            }
            throughput /= p;
        }
        
        // Setup next ray
        // For refraction, offset in the opposite direction of normal
        float3 offsetDir = isRefracted ? -normal : normal;
        ray.Origin = hitPos + offsetDir * RAY_EPSILON;
        ray.Direction = wi;
    }
    
    // Clamp radiance to avoid NaN/Inf contaminating accumulation
    // This can happen due to numerical precision issues in BRDF calculations
    if (any(isnan(radiance)) || any(isinf(radiance)))
    {
        radiance = float3(0.0f, 0.0f, 0.0f);
    }
    radiance = clamp(radiance, 0.0f, 1000.0f);  // Clamp to reasonable range (increased for bright lights)
    
    // Accumulation
    float3 prevColor = AccumulationTexture[launchIndex].xyz;
    float sampleCount = AccumulationTexture[launchIndex].w;
    
    float3 newColor;
    if (Camera.frameIndex == 0)
    {
        newColor = radiance;
        sampleCount = 1.0f;
    }
    else
    {
        // Check for corrupted accumulation buffer
        if (any(isnan(prevColor)) || any(isinf(prevColor)) || sampleCount <= 0.0f)
        {
            newColor = radiance;
            sampleCount = 1.0f;
        }
        else
        {
            newColor = prevColor + (radiance - prevColor) / (sampleCount + 1.0f);
            sampleCount += 1.0f;
        }
    }
    
    AccumulationTexture[launchIndex] = float4(newColor, sampleCount);
    
    // Exposure control - apply before tone mapping
    // Mitsuba scenes often have high radiance values, need proper exposure
    // Area light radiance ~125, wall reflectance ~0.58
    // Single bounce contribution ~72, need very low exposure
    float exposure = 0.015f;  // Very low exposure for bright Mitsuba scenes
    float3 exposed = newColor * exposure;
    
    // ACES filmic tone mapping (better for high dynamic range)
    // Based on: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    float3 finalColor = saturate((exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e));
    
    // Gamma correction
    finalColor = pow(finalColor, 1.0f / 2.2f);
    
    OutputTexture[launchIndex] = float4(finalColor, 1.0f);
}

// ============================================================================
// Closest Hit Shader
// ============================================================================
[shader("closesthit")]
void ClosestHit(inout HitInfo payload, in Attributes attrib)
{
    uint instanceID = InstanceID();
    uint primitiveIndex = PrimitiveIndex();
    
    // Get interpolated normal and texcoord
    float3 normal = GetInterpolatedNormal(instanceID, primitiveIndex, attrib.barycentrics);
    float2 texcoord = GetInterpolatedTexcoord(instanceID, primitiveIndex, attrib.barycentrics);
    
    GPUInstance instance = Instances[instanceID];
    
    // Store hit information
    payload.color = normal;
    payload.hitT = RayTCurrent();
    payload.instanceID = instanceID;
    payload.texcoord = texcoord;
    
    // Check if this is an emitter
    if (instance.isEmitter != 0)
    {
        payload.emission = instance.emission;
    }
    else
    {
        payload.emission = float3(0.0f, 0.0f, 0.0f);
    }
}

// ============================================================================
// Shadow Closest Hit Shader
// ============================================================================
[shader("closesthit")]
void ShadowClosestHit(inout ShadowHitInfo payload, in Attributes attrib)
{
    payload.isHit = true;
}

// ============================================================================
// Miss Shader
// ============================================================================
[shader("miss")]
void Miss(inout HitInfo payload)
{
    payload.hitT = -1.0f;
    payload.color = float3(0.0f, 0.0f, 0.0f);
    payload.emission = float3(0.0f, 0.0f, 0.0f);
}

// ============================================================================
// Shadow Miss Shader
// ============================================================================
[shader("miss")]
void ShadowMiss(inout ShadowHitInfo payload)
{
    payload.isHit = false;
}
