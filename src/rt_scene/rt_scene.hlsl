
// ============================================================================
// Constants
// ============================================================================
static const float PI = 3.14159265358979323846f;
static const float INV_PI = 0.31830988618379067154f;
static const float EPSILON = 1e-6f;
static const float RAY_EPSILON = 1e-4f;

// Material types (must match C++ MaterialType enum)
static const uint MATERIAL_DIFFUSE = 0;
static const uint MATERIAL_CONDUCTOR = 1;
static const uint MATERIAL_ROUGH_CONDUCTOR = 2;
static const uint MATERIAL_DIELECTRIC = 3;
static const uint MATERIAL_ROUGH_DIELECTRIC = 4;
static const uint MATERIAL_PLASTIC = 5;
static const uint MATERIAL_ROUGH_PLASTIC = 6;

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
    float2 padding;
};

struct HitInfo
{
    float3 color;
    float hitT;
    float3 emission;
    uint instanceID;
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

RWTexture2D<float4> OutputTexture              : register(u0);
RWTexture2D<float4> AccumulationTexture        : register(u1);

cbuffer CameraBuffer : register(b0)
{
    CameraConstants Camera;
};

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
// Sampling Functions
// ============================================================================
float3 CosineSampleHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float theta = 2.0f * PI * u.y;
    
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0f, 1.0f - u.x));
    
    return float3(x, y, z);
}

float3 UniformSampleHemisphere(float2 u)
{
    float z = u.x;
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// Build orthonormal basis from normal
void BuildOrthonormalBasis(float3 n, out float3 tangent, out float3 bitangent)
{
    if (abs(n.x) > abs(n.y))
    {
        tangent = normalize(float3(-n.z, 0.0f, n.x));
    }
    else
    {
        tangent = normalize(float3(0.0f, n.z, -n.y));
    }
    bitangent = cross(n, tangent);
}

// Transform direction from local to world space
float3 LocalToWorld(float3 localDir, float3 normal)
{
    float3 tangent, bitangent;
    BuildOrthonormalBasis(normal, tangent, bitangent);
    return tangent * localDir.x + bitangent * localDir.y + normal * localDir.z;
}

// ============================================================================
// BRDF Functions
// ============================================================================

// Schlick Fresnel approximation
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

// GGX Normal Distribution Function
float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    
    float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom);
}

// Smith GGX Geometry function
float G_SmithGGX(float NdotV, float NdotL, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    
    float G1V = NdotV / (NdotV * (1.0f - k) + k);
    float G1L = NdotL / (NdotL * (1.0f - k) + k);
    
    return G1V * G1L;
}

// Sample GGX distribution
float3 SampleGGX(float2 u, float roughness)
{
    float a = roughness * roughness;
    float phi = 2.0f * PI * u.x;
    float cosTheta = sqrt((1.0f - u.y) / (1.0f + (a * a - 1.0f) * u.y));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    
    return float3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
}

// Evaluate diffuse BRDF
float3 EvaluateDiffuse(GPUMaterial mat, float3 wo, float3 wi, float3 normal)
{
    float NdotL = max(0.0f, dot(normal, wi));
    return mat.baseColor * INV_PI;
}

// Evaluate conductor BRDF (perfect mirror for smooth, GGX for rough)
float3 EvaluateConductor(GPUMaterial mat, float3 wo, float3 wi, float3 normal)
{
    if (mat.roughness < 0.01f)
    {
        // Perfect mirror - delta distribution
        return float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 h = normalize(wo + wi);
    float NdotH = max(0.0f, dot(normal, h));
    float NdotV = max(0.0f, dot(normal, wo));
    float NdotL = max(0.0f, dot(normal, wi));
    float VdotH = max(0.0f, dot(wo, h));
    
    float D = D_GGX(NdotH, mat.roughness);
    float G = G_SmithGGX(NdotV, NdotL, mat.roughness);
    float3 F = FresnelSchlick(VdotH, mat.baseColor);
    
    return (D * G * F) / (4.0f * NdotV * NdotL + EPSILON);
}

// Sample BRDF direction
float3 SampleBRDF(GPUMaterial mat, float3 wo, float3 normal, float2 u, out float pdf)
{
    float3 wi;
    
    if (mat.type == MATERIAL_DIFFUSE)
    {
        // Cosine-weighted hemisphere sampling
        float3 localDir = CosineSampleHemisphere(u);
        wi = LocalToWorld(localDir, normal);
        pdf = max(0.0f, dot(normal, wi)) * INV_PI;
    }
    else if (mat.type == MATERIAL_CONDUCTOR || mat.type == MATERIAL_ROUGH_CONDUCTOR)
    {
        if (mat.roughness < 0.01f)
        {
            // Perfect mirror reflection
            wi = reflect(-wo, normal);
            pdf = 1.0f;
        }
        else
        {
            // Sample GGX microfacet normal
            float3 h = LocalToWorld(SampleGGX(u, mat.roughness), normal);
            wi = reflect(-wo, h);
            
            float NdotH = max(0.0f, dot(normal, h));
            float VdotH = max(0.0f, dot(wo, h));
            float D = D_GGX(NdotH, mat.roughness);
            pdf = D * NdotH / (4.0f * VdotH + EPSILON);
        }
    }
    else if (mat.type == MATERIAL_PLASTIC || mat.type == MATERIAL_ROUGH_PLASTIC)
    {
        // Simple plastic: mix between diffuse and specular
        if (u.x < 0.5f)
        {
            // Diffuse
            float3 localDir = CosineSampleHemisphere(float2(u.x * 2.0f, u.y));
            wi = LocalToWorld(localDir, normal);
            pdf = 0.5f * max(0.0f, dot(normal, wi)) * INV_PI;
        }
        else
        {
            // Specular
            float3 h = LocalToWorld(SampleGGX(float2((u.x - 0.5f) * 2.0f, u.y), mat.roughness), normal);
            wi = reflect(-wo, h);
            
            float NdotH = max(0.0f, dot(normal, h));
            float VdotH = max(0.0f, dot(wo, h));
            float D = D_GGX(NdotH, mat.roughness);
            pdf = 0.5f * D * NdotH / (4.0f * VdotH + EPSILON);
        }
    }
    else
    {
        // Default to diffuse
        float3 localDir = CosineSampleHemisphere(u);
        wi = LocalToWorld(localDir, normal);
        pdf = max(0.0f, dot(normal, wi)) * INV_PI;
    }
    
    return wi;
}

// Evaluate BRDF
float3 EvaluateBRDF(GPUMaterial mat, float3 wo, float3 wi, float3 normal)
{
    float NdotL = dot(normal, wi);
    if (NdotL <= 0.0f) return float3(0.0f, 0.0f, 0.0f);
    
    if (mat.type == MATERIAL_DIFFUSE)
    {
        return EvaluateDiffuse(mat, wo, wi, normal);
    }
    else if (mat.type == MATERIAL_CONDUCTOR || mat.type == MATERIAL_ROUGH_CONDUCTOR)
    {
        return EvaluateConductor(mat, wo, wi, normal);
    }
    else if (mat.type == MATERIAL_PLASTIC || mat.type == MATERIAL_ROUGH_PLASTIC)
    {
        // Plastic: diffuse + specular
        float3 diffuse = mat.baseColor * INV_PI;
        float3 specular = EvaluateConductor(mat, wo, wi, normal);
        
        float fresnel = pow(1.0f - max(0.0f, dot(normal, wo)), 5.0f);
        return lerp(diffuse, specular, fresnel * 0.5f);
    }
    
    return mat.baseColor * INV_PI;
}

// ============================================================================
// Intersection Helpers
// ============================================================================
float3 GetInterpolatedNormal(uint instanceID, uint primitiveIndex, float2 barycentrics)
{
    GPUInstance instance = Instances[instanceID];
    
    uint i0 = Indices[instance.indexOffset + primitiveIndex * 3 + 0];
    uint i1 = Indices[instance.indexOffset + primitiveIndex * 3 + 1];
    uint i2 = Indices[instance.indexOffset + primitiveIndex * 3 + 2];
    
    float3 n0 = Vertices[i0].normal;
    float3 n1 = Vertices[i1].normal;
    float3 n2 = Vertices[i2].normal;
    
    float3 normal = n0 * (1.0f - barycentrics.x - barycentrics.y) +
                    n1 * barycentrics.x +
                    n2 * barycentrics.y;
    
    return normalize(normal);
}

float3 GetHitPosition(float3 rayOrigin, float3 rayDirection, float hitT)
{
    return rayOrigin + rayDirection * hitT;
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
    float4 rayView = mul(Camera.projInverse, rayClip);
    rayView /= rayView.w;
    
    float4 rayWorld = mul(Camera.viewInverse, float4(rayView.xyz, 0.0f));
    float3 rayDir = normalize(rayWorld.xyz);
    
    float3 rayOrigin = Camera.cameraPosition;
    
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
            // Miss - add background color
            float3 skyColor = lerp(float3(0.5f, 0.7f, 1.0f), float3(0.8f, 0.9f, 1.0f), 
                                   saturate(ray.Direction.y * 0.5f + 0.5f));
            radiance += throughput * skyColor * 0.3f;
            break;
        }
        
        // Add emission
        radiance += throughput * payload.emission;
        
        // Get hit information
        float3 hitPos = GetHitPosition(ray.Origin, ray.Direction, payload.hitT);
        GPUInstance instance = Instances[payload.instanceID];
        GPUMaterial mat = Materials[instance.materialIndex];
        
        // Get interpolated normal (stored in payload.color for simplicity)
        float3 normal = payload.color;
        
        // Make sure normal faces the ray
        if (dot(normal, ray.Direction) > 0.0f)
        {
            normal = -normal;
        }
        
        // Sample new direction
        float3 wo = -ray.Direction;
        float pdf;
        float3 wi = SampleBRDF(mat, wo, normal, RandomFloat2(rng), pdf);
        
        if (pdf < EPSILON || dot(normal, wi) <= 0.0f)
        {
            break;
        }
        
        // Update throughput
        float3 brdf = EvaluateBRDF(mat, wo, wi, normal);
        float cosTheta = abs(dot(normal, wi));
        throughput *= brdf * cosTheta / pdf;
        
        // Russian roulette
        if (bounce > 3)
        {
            float p = max(throughput.x, max(throughput.y, throughput.z));
            if (RandomFloat(rng) > p)
            {
                break;
            }
            throughput /= p;
        }
        
        // Setup next ray
        ray.Origin = hitPos + normal * RAY_EPSILON;
        ray.Direction = wi;
    }
    
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
        newColor = prevColor + (radiance - prevColor) / (sampleCount + 1.0f);
        sampleCount += 1.0f;
    }
    
    AccumulationTexture[launchIndex] = float4(newColor, sampleCount);
    
    // Tone mapping and gamma correction
    float3 finalColor = newColor / (newColor + 1.0f);  // Reinhard tone mapping
    finalColor = pow(finalColor, 1.0f / 2.2f);          // Gamma correction
    
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
    
    // Get interpolated normal
    float3 normal = GetInterpolatedNormal(instanceID, primitiveIndex, attrib.barycentrics);
    
    GPUInstance instance = Instances[instanceID];
    
    // Store hit information
    payload.color = normal;
    payload.hitT = RayTCurrent();
    payload.instanceID = instanceID;
    
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
