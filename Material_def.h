#pragma once
#include "gdt/math/vec.h"
#include <cuda_runtime.h>

using namespace gdt;

enum material_kind
{
    DIFFUSE, METAL, DIELECTRIC
};

struct material_mes {
    material_kind mat_kind;

    vec3f diffuse = 1.0f;
    float roughness = 1.0f;
    float transparent = 0.0f;
    float ior = 1.8f;
    vec3f emitter = 0;

    int diffuseTextureID{ -1 };
    int emitterTextureID{ -1 };
    cudaTextureObject_t diffuse_texture;
    cudaTextureObject_t emitter_texture;
};