#pragma once
#include "gdt/math/vec.h"
#include <cuda_runtime.h>

using namespace gdt;

/*目前定义的所有材质类型*/
enum material_kind
{
    DIFFUSE, METAL, DIELECTRIC       //漫反射、镜面反射、透射，三大基础光追材质
};

/*材质参数*/
struct material_mes {
    material_kind mat_kind;          //材质类型，0漫1镜2透

    vec3f diffuse = 1.0f;            //漫反射颜色
    float roughness = 1.0f;          //粗糙度
    float transparent = 0.0f;        //透射度
    float ior = 1.8f;                //折射率
    vec3f emitter = 0;               //自发光

    int diffuseTextureID{ -1 };      //漫反射贴图编号
    int emitterTextureID{ -1 };      //自发光贴图编号
    cudaTextureObject_t diffuse_texture;    //漫反射贴图实例
    cudaTextureObject_t emitter_texture;    //自发光贴图实例
};