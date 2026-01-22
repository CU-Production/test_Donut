#pragma once

// ============================================================================
// Texture Loading Utilities
// Supports: PNG, JPG, TGA, BMP, HDR, EXR
// Uses stb_image for standard formats and tinyexr for EXR
// ============================================================================

#include <string>
#include <vector>
#include <filesystem>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include <donut/core/log.h>

// stb_image for standard image formats (PNG, JPG, TGA, BMP, HDR)
#include <stb_image.h>

// tinyexr for EXR format
#include <tinyexr.h>

namespace texture_utils
{

// Texture data structure
struct TextureData
{
    std::vector<float> data;    // RGBA float data (always converted to float)
    int width = 0;
    int height = 0;
    int channels = 4;           // Always converted to 4 channels (RGBA)
    bool isHDR = false;         // True for HDR/EXR formats
    std::string path;           // Original file path
    
    bool IsValid() const { return !data.empty() && width > 0 && height > 0; }
    size_t GetPixelCount() const { return static_cast<size_t>(width) * height; }
    size_t GetDataSize() const { return data.size() * sizeof(float); }
};

// Environment map data (cubemap or equirectangular)
struct EnvironmentMap
{
    TextureData texture;
    float intensity = 1.0f;
    bool isCubemap = false;
    
    bool IsValid() const { return texture.IsValid(); }
};

// Load texture from file (auto-detects format)
inline TextureData LoadTexture(const std::filesystem::path& filePath)
{
    TextureData result;
    result.path = filePath.string();
    
    if (!std::filesystem::exists(filePath))
    {
        donut::log::error("Texture file not found: %s", filePath.string().c_str());
        return result;
    }
    
    std::string ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // EXR format - use tinyexr
    if (ext == ".exr")
    {
        float* exrData = nullptr;
        int width, height;
        const char* err = nullptr;
        
        int ret = LoadEXR(&exrData, &width, &height, filePath.string().c_str(), &err);
        
        if (ret != TINYEXR_SUCCESS)
        {
            if (err)
            {
                donut::log::error("Failed to load EXR: %s - %s", filePath.string().c_str(), err);
                FreeEXRErrorMessage(err);
            }
            return result;
        }
        
        result.width = width;
        result.height = height;
        result.channels = 4;
        result.isHDR = true;
        result.data.resize(static_cast<size_t>(width) * height * 4);
        std::memcpy(result.data.data(), exrData, result.data.size() * sizeof(float));
        
        free(exrData);
        donut::log::info("Loaded EXR texture: %s (%dx%d)", filePath.filename().string().c_str(), width, height);
    }
    // HDR format - use stb_image
    else if (ext == ".hdr")
    {
        int width, height, channels;
        float* hdrData = stbi_loadf(filePath.string().c_str(), &width, &height, &channels, 4);
        
        if (!hdrData)
        {
            donut::log::error("Failed to load HDR: %s - %s", filePath.string().c_str(), stbi_failure_reason());
            return result;
        }
        
        result.width = width;
        result.height = height;
        result.channels = 4;
        result.isHDR = true;
        result.data.resize(static_cast<size_t>(width) * height * 4);
        std::memcpy(result.data.data(), hdrData, result.data.size() * sizeof(float));
        
        stbi_image_free(hdrData);
        donut::log::info("Loaded HDR texture: %s (%dx%d)", filePath.filename().string().c_str(), width, height);
    }
    // Standard formats (PNG, JPG, TGA, BMP) - use stb_image and convert to float
    else
    {
        int width, height, channels;
        unsigned char* imgData = stbi_load(filePath.string().c_str(), &width, &height, &channels, 4);
        
        if (!imgData)
        {
            donut::log::error("Failed to load texture: %s - %s", filePath.string().c_str(), stbi_failure_reason());
            return result;
        }
        
        result.width = width;
        result.height = height;
        result.channels = 4;
        result.isHDR = false;
        result.data.resize(static_cast<size_t>(width) * height * 4);
        
        // Convert uint8 to float (with gamma to linear conversion for sRGB)
        for (size_t i = 0; i < result.data.size(); i++)
        {
            float value = imgData[i] / 255.0f;
            // Apply sRGB to linear conversion for RGB channels (not alpha)
            if ((i % 4) < 3)
            {
                value = (value <= 0.04045f) ? (value / 12.92f) : std::pow((value + 0.055f) / 1.055f, 2.4f);
            }
            result.data[i] = value;
        }
        
        stbi_image_free(imgData);
        donut::log::info("Loaded texture: %s (%dx%d, %d channels)", 
            filePath.filename().string().c_str(), width, height, channels);
    }
    
    return result;
}

// Load environment map from file
inline EnvironmentMap LoadEnvironmentMap(const std::filesystem::path& filePath, float intensity = 1.0f)
{
    EnvironmentMap result;
    result.texture = LoadTexture(filePath);
    result.intensity = intensity;
    result.isCubemap = false;  // We always load as equirectangular
    return result;
}

// Sample equirectangular environment map at given direction
inline void SampleEquirectangular(const TextureData& envMap, float dirX, float dirY, float dirZ,
                                   float& outR, float& outG, float& outB)
{
    if (!envMap.IsValid())
    {
        outR = outG = outB = 0.0f;
        return;
    }
    
    // Convert direction to spherical coordinates
    float theta = std::atan2(dirX, dirZ);  // Azimuth angle
    float phi = std::asin(std::clamp(dirY, -1.0f, 1.0f));  // Elevation angle
    
    // Convert to UV coordinates
    const float PI = 3.14159265359f;
    float u = (theta + PI) / (2.0f * PI);
    float v = (phi + PI * 0.5f) / PI;
    
    // Sample texture (bilinear)
    float fx = u * (envMap.width - 1);
    float fy = v * (envMap.height - 1);
    int x0 = static_cast<int>(fx);
    int y0 = static_cast<int>(fy);
    int x1 = std::min(x0 + 1, envMap.width - 1);
    int y1 = std::min(y0 + 1, envMap.height - 1);
    float wx = fx - x0;
    float wy = fy - y0;
    
    auto sample = [&](int x, int y) -> const float* {
        return &envMap.data[(static_cast<size_t>(y) * envMap.width + x) * 4];
    };
    
    const float* c00 = sample(x0, y0);
    const float* c10 = sample(x1, y0);
    const float* c01 = sample(x0, y1);
    const float* c11 = sample(x1, y1);
    
    outR = (1-wx)*(1-wy)*c00[0] + wx*(1-wy)*c10[0] + (1-wx)*wy*c01[0] + wx*wy*c11[0];
    outG = (1-wx)*(1-wy)*c00[1] + wx*(1-wy)*c10[1] + (1-wx)*wy*c01[1] + wx*wy*c11[1];
    outB = (1-wx)*(1-wy)*c00[2] + wx*(1-wy)*c10[2] + (1-wx)*wy*c01[2] + wx*wy*c11[2];
}

// Create mip chain for texture (simple box filter)
inline std::vector<TextureData> GenerateMipChain(const TextureData& baseTexture, int maxLevels = 0)
{
    std::vector<TextureData> mipChain;
    mipChain.push_back(baseTexture);
    
    int width = baseTexture.width;
    int height = baseTexture.height;
    int level = 0;
    
    while (width > 1 || height > 1)
    {
        if (maxLevels > 0 && level >= maxLevels - 1) break;
        
        int newWidth = std::max(1, width / 2);
        int newHeight = std::max(1, height / 2);
        
        TextureData mip;
        mip.width = newWidth;
        mip.height = newHeight;
        mip.channels = 4;
        mip.isHDR = baseTexture.isHDR;
        mip.data.resize(static_cast<size_t>(newWidth) * newHeight * 4);
        
        const TextureData& prevMip = mipChain.back();
        
        for (int y = 0; y < newHeight; y++)
        {
            for (int x = 0; x < newWidth; x++)
            {
                int srcX = x * 2;
                int srcY = y * 2;
                
                float r = 0, g = 0, b = 0, a = 0;
                int count = 0;
                
                for (int dy = 0; dy < 2 && srcY + dy < prevMip.height; dy++)
                {
                    for (int dx = 0; dx < 2 && srcX + dx < prevMip.width; dx++)
                    {
                        size_t idx = (static_cast<size_t>(srcY + dy) * prevMip.width + (srcX + dx)) * 4;
                        r += prevMip.data[idx + 0];
                        g += prevMip.data[idx + 1];
                        b += prevMip.data[idx + 2];
                        a += prevMip.data[idx + 3];
                        count++;
                    }
                }
                
                size_t dstIdx = (static_cast<size_t>(y) * newWidth + x) * 4;
                mip.data[dstIdx + 0] = r / count;
                mip.data[dstIdx + 1] = g / count;
                mip.data[dstIdx + 2] = b / count;
                mip.data[dstIdx + 3] = a / count;
            }
        }
        
        mipChain.push_back(std::move(mip));
        width = newWidth;
        height = newHeight;
        level++;
    }
    
    return mipChain;
}

} // namespace texture_utils
