// ============================================================================
// conductor_ior.hlsli - Conductor IOR Presets (Mitsuba3 compatible)
// 
// Complex index of refraction (eta, k) for various metals.
// Data from Mitsuba3's conductor material presets.
// Reference: https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
// ============================================================================
#ifndef MATERIALS_CONDUCTOR_IOR_HLSLI
#define MATERIALS_CONDUCTOR_IOR_HLSLI

// Note: These are approximate RGB values derived from spectral data
// For accurate results, spectral rendering would be needed

// Silver (Ag) - highly reflective
static const float3 ETA_SILVER = float3(0.155f, 0.117f, 0.138f);
static const float3 K_SILVER = float3(4.827f, 3.122f, 2.147f);

// Gold (Au) - characteristic yellow
static const float3 ETA_GOLD = float3(0.143f, 0.374f, 1.442f);
static const float3 K_GOLD = float3(3.983f, 2.387f, 1.603f);

// Copper (Cu) - reddish
static const float3 ETA_COPPER = float3(0.200f, 0.924f, 1.102f);
static const float3 K_COPPER = float3(3.912f, 2.452f, 2.142f);

// Aluminium (Al) - neutral
static const float3 ETA_ALUMINIUM = float3(1.657f, 0.880f, 0.521f);
static const float3 K_ALUMINIUM = float3(9.224f, 6.269f, 4.837f);

// Iron (Fe)
static const float3 ETA_IRON = float3(2.950f, 2.930f, 2.650f);
static const float3 K_IRON = float3(3.000f, 2.950f, 2.800f);

// Chromium (Cr)
static const float3 ETA_CHROMIUM = float3(3.180f, 3.180f, 2.010f);
static const float3 K_CHROMIUM = float3(3.300f, 3.330f, 3.040f);

// Nickel (Ni)
static const float3 ETA_NICKEL = float3(1.970f, 1.860f, 1.670f);
static const float3 K_NICKEL = float3(3.740f, 3.060f, 2.580f);

// Titanium (Ti)
static const float3 ETA_TITANIUM = float3(2.160f, 1.970f, 1.810f);
static const float3 K_TITANIUM = float3(2.930f, 2.620f, 2.350f);

// Platinum (Pt)
static const float3 ETA_PLATINUM = float3(2.180f, 2.070f, 1.860f);
static const float3 K_PLATINUM = float3(4.270f, 3.740f, 3.100f);

// Tungsten (W)
static const float3 ETA_TUNGSTEN = float3(4.350f, 3.400f, 2.850f);
static const float3 K_TUNGSTEN = float3(3.400f, 2.700f, 2.150f);

// Mercury (Hg)
static const float3 ETA_MERCURY = float3(2.400f, 1.920f, 1.540f);
static const float3 K_MERCURY = float3(5.200f, 4.260f, 3.450f);

// Lead (Pb)
static const float3 ETA_LEAD = float3(1.910f, 1.830f, 1.440f);
static const float3 K_LEAD = float3(3.500f, 3.400f, 3.170f);

// ============================================================================
// Helper function to get conductor IOR by index
// ============================================================================

// Conductor material types (for parsing)
static const uint CONDUCTOR_NONE = 0;       // Perfect mirror (100% reflective)
static const uint CONDUCTOR_SILVER = 1;
static const uint CONDUCTOR_GOLD = 2;
static const uint CONDUCTOR_COPPER = 3;
static const uint CONDUCTOR_ALUMINIUM = 4;
static const uint CONDUCTOR_IRON = 5;
static const uint CONDUCTOR_CHROMIUM = 6;
static const uint CONDUCTOR_NICKEL = 7;
static const uint CONDUCTOR_TITANIUM = 8;
static const uint CONDUCTOR_PLATINUM = 9;
static const uint CONDUCTOR_TUNGSTEN = 10;
static const uint CONDUCTOR_MERCURY = 11;
static const uint CONDUCTOR_LEAD = 12;

// Get eta and k for a conductor material preset
void GetConductorIOR(uint materialPreset, out float3 eta, out float3 k)
{
    switch (materialPreset)
    {
        case CONDUCTOR_SILVER:
            eta = ETA_SILVER;
            k = K_SILVER;
            break;
        case CONDUCTOR_GOLD:
            eta = ETA_GOLD;
            k = K_GOLD;
            break;
        case CONDUCTOR_COPPER:
            eta = ETA_COPPER;
            k = K_COPPER;
            break;
        case CONDUCTOR_ALUMINIUM:
            eta = ETA_ALUMINIUM;
            k = K_ALUMINIUM;
            break;
        case CONDUCTOR_IRON:
            eta = ETA_IRON;
            k = K_IRON;
            break;
        case CONDUCTOR_CHROMIUM:
            eta = ETA_CHROMIUM;
            k = K_CHROMIUM;
            break;
        case CONDUCTOR_NICKEL:
            eta = ETA_NICKEL;
            k = K_NICKEL;
            break;
        case CONDUCTOR_TITANIUM:
            eta = ETA_TITANIUM;
            k = K_TITANIUM;
            break;
        case CONDUCTOR_PLATINUM:
            eta = ETA_PLATINUM;
            k = K_PLATINUM;
            break;
        case CONDUCTOR_TUNGSTEN:
            eta = ETA_TUNGSTEN;
            k = K_TUNGSTEN;
            break;
        case CONDUCTOR_MERCURY:
            eta = ETA_MERCURY;
            k = K_MERCURY;
            break;
        case CONDUCTOR_LEAD:
            eta = ETA_LEAD;
            k = K_LEAD;
            break;
        case CONDUCTOR_NONE:
        default:
            // Perfect mirror - no Fresnel (100% reflective)
            eta = float3(0.0f, 0.0f, 0.0f);
            k = float3(1.0f, 1.0f, 1.0f);  // Use k > 0 to trigger conductor path but F = 1
            break;
    }
}

#endif // MATERIALS_CONDUCTOR_IOR_HLSLI
