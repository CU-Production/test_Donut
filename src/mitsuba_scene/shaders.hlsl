
// ============================================================================
// Constant Buffers
// ============================================================================
cbuffer PerObjectConstants : register(b0)
{
    float4x4 g_WorldViewProj;
    float4x4 g_World;
    float3 g_BaseColor;
    float g_Roughness;
    float3 g_Emission;
    uint g_IsEmitter;
    uint g_HasBaseColorTex;
    float3 _pad_obj;
};

cbuffer LightConstants : register(b1)
{
    float3 g_LightDir;
    float _pad0;
    float3 g_LightColor;
    float _pad1;
    float3 g_AmbientColor;
    float _pad2;
    float3 g_CameraPos;
    float _pad3;
};

// ============================================================================
// Textures and Samplers
// ============================================================================
Texture2D<float4> g_BaseColorTex : register(t0);
SamplerState g_LinearSampler : register(s0);

// ============================================================================
// Vertex Shader Input/Output
// ============================================================================
struct VSInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 texcoord : TEXCOORD;
};

struct VSOutput
{
    float4 position     : SV_Position;
    float3 worldPos     : WORLD_POS;
    float3 normal       : NORMAL;
    float2 texcoord     : TEXCOORD;
};

// ============================================================================
// Vertex Shader
// ============================================================================
VSOutput main_vs(VSInput input)
{
    VSOutput output;
    
    // Transform position to clip space using MVP matrix (column-vector convention: M * v)
    output.position = mul(g_WorldViewProj, float4(input.position, 1.0f));
    
    // Transform position to world space using model matrix (column-vector convention)
    float4 worldPos4 = mul(g_World, float4(input.position, 1.0f));
    output.worldPos = worldPos4.xyz;
    
    // Transform normal to world space (using upper 3x3 of world matrix)
    // For non-uniform scaling, should use inverse transpose, but for now assume uniform scale
    output.normal = normalize(mul((float3x3)g_World, input.normal));
    
    // Pass texcoord
    output.texcoord = input.texcoord;
    
    return output;
}

// ============================================================================
// Pixel Shader
// ============================================================================
static const float PI = 3.14159265358979323846f;

// Schlick Fresnel approximation
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0f - F0) * pow(saturate(1.0f - cosTheta), 5.0f);
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

float4 main_ps(VSOutput input) : SV_Target
{
    // Normalize interpolated normal
    float3 N = normalize(input.normal);
    float3 L = normalize(g_LightDir);
    float3 V = normalize(g_CameraPos - input.worldPos);  // View direction from surface to camera
    float3 H = normalize(L + V);
    
    // Material properties - sample texture if available
    float3 albedo = g_BaseColor;
    if (g_HasBaseColorTex != 0)
    {
        float4 texColor = g_BaseColorTex.Sample(g_LinearSampler, input.texcoord);
        albedo = texColor.rgb;
    }
    float roughness = max(g_Roughness, 0.04f);  // Minimum roughness to avoid artifacts
    float metallic = 0.0f;  // Simple diffuse assumption
    
    // Handle emitters
    if (g_IsEmitter != 0)
    {
        // Scale emission for display
        float3 emissionDisplay = g_Emission * 0.01f;  // Tone down for display
        return float4(emissionDisplay, 1.0f);
    }
    
    // Calculate dot products
    float NdotL = max(dot(N, L), 0.0f);
    float NdotV = max(dot(N, V), 0.0f);
    float NdotH = max(dot(N, H), 0.0f);
    float VdotH = max(dot(V, H), 0.0f);
    
    // Fresnel
    float3 F0 = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F = FresnelSchlick(VdotH, F0);
    
    // Specular BRDF (Cook-Torrance)
    float D = D_GGX(NdotH, roughness);
    float G = G_SmithGGX(NdotV, NdotL, roughness);
    
    float3 numerator = D * G * F;
    float denominator = 4.0f * NdotV * NdotL + 0.0001f;
    float3 specular = numerator / denominator;
    
    // Diffuse BRDF (Lambertian)
    float3 kS = F;
    float3 kD = (1.0f - kS) * (1.0f - metallic);
    float3 diffuse = kD * albedo / PI;
    
    // Direct lighting
    float3 directLight = (diffuse + specular) * g_LightColor * NdotL;
    
    // Ambient lighting (simple hemisphere)
    float3 ambient = g_AmbientColor * albedo;
    
    // Final color
    float3 color = directLight + ambient;
    
    // Simple tone mapping (Reinhard)
    color = color / (color + 1.0f);
    
    // Gamma correction
    color = pow(color, 1.0f / 2.2f);
    
    return float4(color, 1.0f);
}
