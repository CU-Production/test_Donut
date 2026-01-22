// ============================================================================
// common.hlsli - Common BRDF utilities and math functions
// ============================================================================
#ifndef MATERIALS_COMMON_HLSLI
#define MATERIALS_COMMON_HLSLI

// ============================================================================
// Constants
// ============================================================================
static const float PI = 3.14159265358979323846f;
static const float TWO_PI = 6.28318530717958647692f;
static const float INV_PI = 0.31830988618379067154f;
static const float INV_TWO_PI = 0.15915494309189533577f;
static const float EPSILON = 1e-6f;
static const float RAY_EPSILON = 1e-4f;

// ============================================================================
// Fresnel Functions
// ============================================================================

// Schlick approximation for dielectric Fresnel
float FresnelSchlickScalar(float cosTheta, float F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

// Schlick approximation for conductor Fresnel (colored)
float3 FresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

// Exact Fresnel for dielectrics
float FresnelDielectric(float cosThetaI, float eta)
{
    // Handle total internal reflection
    float sinThetaI = sqrt(max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = sinThetaI / eta;
    
    if (sinThetaT >= 1.0f)
        return 1.0f;  // Total internal reflection
    
    float cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaT * sinThetaT));
    
    float Rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float Rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// Fresnel for conductors using complex IOR (eta, k)
float3 FresnelConductor(float cosTheta, float3 eta, float3 k)
{
    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta2 = 1.0f - cosTheta2;
    
    float3 eta2 = eta * eta;
    float3 k2 = k * k;
    
    float3 t0 = eta2 - k2 - sinTheta2;
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0f * eta2 * k2);
    float3 t1 = a2plusb2 + cosTheta2;
    float3 a = sqrt(0.5f * (a2plusb2 + t0));
    float3 t2 = 2.0f * a * cosTheta;
    float3 Rs = (t1 - t2) / (t1 + t2);
    
    float3 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    float3 t4 = t2 * sinTheta2;
    float3 Rp = Rs * (t3 - t4) / (t3 + t4);
    
    return 0.5f * (Rs + Rp);
}

// ============================================================================
// Microfacet Distribution Functions (GGX/Trowbridge-Reitz)
// ============================================================================

// GGX Normal Distribution Function
float D_GGX(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float denom = NdotH2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denom * denom + EPSILON);
}

// Smith GGX Geometry function (separable form)
float G1_GGX(float NdotV, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdotV2 = NdotV * NdotV;
    return 2.0f * NdotV / (NdotV + sqrt(alpha2 + (1.0f - alpha2) * NdotV2) + EPSILON);
}

// Smith GGX Geometry function for both view and light directions
float G_SmithGGX(float NdotV, float NdotL, float alpha)
{
    return G1_GGX(NdotV, alpha) * G1_GGX(NdotL, alpha);
}

// Alternative Smith GGX (Schlick approximation, faster but less accurate)
float G_SmithGGXSchlick(float NdotV, float NdotL, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    
    float G1V = NdotV / (NdotV * (1.0f - k) + k + EPSILON);
    float G1L = NdotL / (NdotL * (1.0f - k) + k + EPSILON);
    
    return G1V * G1L;
}

// GGX VNDF (Visible Normal Distribution Function) PDF
float PDF_GGX_VNDF(float NdotH, float NdotV, float VdotH, float alpha)
{
    float D = D_GGX(NdotH, alpha);
    float G1 = G1_GGX(NdotV, alpha);
    return D * G1 * VdotH / (NdotV + EPSILON);
}

// ============================================================================
// Sampling Functions
// ============================================================================

// Cosine-weighted hemisphere sampling (for diffuse)
float3 CosineSampleHemisphere(float2 u)
{
    float r = sqrt(u.x);
    float phi = TWO_PI * u.y;
    
    return float3(
        r * cos(phi),
        r * sin(phi),
        sqrt(max(0.0f, 1.0f - u.x))
    );
}

// Sample GGX microfacet normal
float3 SampleGGX(float2 u, float alpha)
{
    float phi = TWO_PI * u.x;
    float cosTheta = sqrt((1.0f - u.y) / (1.0f + (alpha * alpha - 1.0f) * u.y + EPSILON));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    
    return float3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
}

// Sample GGX VNDF (better importance sampling for rough surfaces)
// Reference: "Sampling the GGX Distribution of Visible Normals" by Heitz
float3 SampleGGX_VNDF(float2 u, float3 Ve, float alpha)
{
    // Transform view direction to hemisphere configuration
    float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
    
    // Orthonormal basis
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) / sqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);
    
    // Parameterization of the projected area
    float r = sqrt(u.x);
    float phi = TWO_PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - t1 * t1)) + s * t2;
    
    // Reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    
    // Transform back to ellipsoid configuration
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

// ============================================================================
// Coordinate System Utilities
// ============================================================================

// Build orthonormal basis from normal
void BuildOrthonormalBasis(float3 n, out float3 tangent, out float3 bitangent)
{
    if (abs(n.x) > abs(n.y))
        tangent = normalize(float3(-n.z, 0.0f, n.x));
    else
        tangent = normalize(float3(0.0f, n.z, -n.y));
    bitangent = cross(n, tangent);
}

// Transform direction from local (tangent) space to world space
float3 LocalToWorld(float3 localDir, float3 normal)
{
    float3 tangent, bitangent;
    BuildOrthonormalBasis(normal, tangent, bitangent);
    return tangent * localDir.x + bitangent * localDir.y + normal * localDir.z;
}

// Transform direction from world space to local (tangent) space
float3 WorldToLocal(float3 worldDir, float3 normal)
{
    float3 tangent, bitangent;
    BuildOrthonormalBasis(normal, tangent, bitangent);
    return float3(dot(worldDir, tangent), dot(worldDir, bitangent), dot(worldDir, normal));
}

// ============================================================================
// Helper Functions
// ============================================================================

// Reflect vector about normal
float3 Reflect(float3 v, float3 n)
{
    return 2.0f * dot(v, n) * n - v;
}

// Refract vector (returns false if total internal reflection)
bool Refract(float3 v, float3 n, float eta, out float3 refracted)
{
    float cosI = dot(v, n);
    float sin2I = max(0.0f, 1.0f - cosI * cosI);
    float sin2T = eta * eta * sin2I;
    
    if (sin2T >= 1.0f)
    {
        refracted = float3(0.0f, 0.0f, 0.0f);
        return false;  // Total internal reflection
    }
    
    float cosT = sqrt(1.0f - sin2T);
    refracted = eta * (-v) + (eta * cosI - cosT) * n;
    return true;
}

// Convert roughness to alpha (squared roughness)
float RoughnessToAlpha(float roughness)
{
    return max(roughness * roughness, 0.001f);
}

// Luminance of RGB color
float Luminance(float3 color)
{
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

#endif // MATERIALS_COMMON_HLSLI
