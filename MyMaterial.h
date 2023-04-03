#pragma once
#include "MyInteraction.h"
#include "MyTool.h"

#define PI 3.1415926
using namespace osc;

/*PRD是一个随机采样器，通过prd.random.init()播随机种子，然后prd.random()得到(0,1)的随机数*/
typedef gdt::LCG<16> Random;
struct PRD {
    Random random;
    vec3f  pixelColor;
};

/*计算漫反射材质的bsdf、pdf，并采样下一次弹射方向*/
__forceinline__ __device__ vec3f cal_diffuse_bsdf(const Interaction& isect, const vec3f& wi, vec3f *wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    vec3f bsdf = diffuseColor / float(PI);
    vec3f rnd;
    PRD prd;
    prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921,
        frame_id * 348593 % 43832 + iy * 324123 % 23415);
    rnd.x = prd.random() * 2 - 1;
    prd.random.init(frame_id * 972823 % 12971 + ix * 743782 % 82013,
        frame_id * 893022 % 28191 + iy * 918212 % 51321);
    rnd.y = prd.random() * 2 - 1;
    prd.random.init(frame_id * 383921 % 48839 + ix * 572131 % 47128,
        frame_id * 389291 % 29301 + iy * 716271 % 63291);
    rnd.z = prd.random() * 2 - 1;
    vec3f wos = normalize(isect.geomNormal + normalize(rnd));
    wo->x = wos.x; wo->y = wos.y; wo->z = wos.z;
    *pdf = 1 / (2 * float(PI));
    return bsdf;
}

/*计算镜面反射材质的bsdf、pdf，并采样下一次弹射方向*/
__forceinline__ __device__ vec3f cal_metal_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    vec3f bsdf = diffuseColor / float(PI);
    vec3f out;
    out = wi - 2.0f * (vec3f)dot(wi, isect.geomNormal) * isect.geomNormal;
    out = normalize(out);
    vec3f out1 = cross(out, vec3f(1.0f));
    vec3f out2 = cross(out, out1);
    PRD prd; 
    prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
    vec3f out3 = normalize(out + (prd.random() * 2 - 1) * out1 * isect.mat_mes.roughness + (prd.random() * 2 - 1) * out2 * isect.mat_mes.roughness);
    if (dot(out3, isect.geomNormal) <= 0)
        out3 = out;
    *wo = out3;
    *pdf = 1 / ((float(PI) - 1) * isect.mat_mes.roughness + 1);
    return bsdf;
}

/*计算纯镜面反射材质的bsdf、pdf，并采样下一次弹射方向*/
__forceinline__ __device__ vec3f cal_mirror_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    vec3f bsdf = diffuseColor;
    vec3f out;
    out = wi - 2.0f * (vec3f)dot(wi, isect.geomNormal) * isect.geomNormal;
    *wo = normalize(out);
    *pdf = 1;
    return bsdf;
}

/*计算透射材质的bsdf、pdf，并采样下一次弹射方向*/
__forceinline__ __device__ vec3f cal_dielectric_bsdf(const Interaction& isect, const vec3f& wi, vec3f* wo, float* pdf, const int ix, const int iy, const int frame_id) {
    vec3f diffuseColor = isect.mat_mes.diffuse;
    if (isect.mat_mes.diffuseTextureID != -1) {
        float u = isect.texcoord.x;
        float v = isect.texcoord.y;
        vec4f fromTexture = tex2D<float4>(isect.mat_mes.diffuse_texture, u, v);
        diffuseColor *= (vec3f)fromTexture;
    }
    vec3f bsdf = diffuseColor;
    *pdf = 1;
    vec3f out;
    float etai_over_etat = 0;
    if (dot(wi, isect.realNormal) > 0) etai_over_etat = isect.mat_mes.ior;
    else etai_over_etat = 1.0f / isect.mat_mes.ior;
    vec3f unit_direction = normalize(wi);
    double cos_theta = my_min(dot(-unit_direction, isect.geomNormal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) { //全内反射
        *wo = reflect(unit_direction, isect.geomNormal);
        return bsdf;
    }
    double reflect_prob = schlick(cos_theta, etai_over_etat);//反射率
    PRD prd;
    prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
    if (prd.random() < reflect_prob) {
        *wo = reflect(unit_direction, isect.geomNormal);
        return bsdf;
    }
    if (prd.random() < isect.mat_mes.transparent) {
        //*wo = reflect(unit_direction, isect.geomNormal);
        //return bsdf;
    }
    *wo = refract(unit_direction, isect.geomNormal, etai_over_etat);
    return bsdf;
}

/*计算材质的bsdf、pdf，并采样下一次弹射方向，这里主要是用于分流*/
__forceinline__ __device__ vec3f cal_bsdf(const Interaction &isect, const vec3f &wi, vec3f *wo, float *pdf,const int ix, const int iy, const int frame_id)
{
    PRD prd;
    vec3f result;
    if (isect.mat_mes.mat_kind == DIFFUSE) {
        result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }
    else if (isect.mat_mes.mat_kind == METAL) {
        //result = cal_metal_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        prd.random.init(frame_id * 234834 % 32849 + ix * 385932 % 82921, frame_id * 348593 % 43832 + iy * 324123 % 23415);
        float res = prd.random();
        if (res < isect.mat_mes.roughness)
            result = cal_diffuse_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
        else
            result = cal_mirror_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }
    else if (isect.mat_mes.mat_kind == DIELECTRIC) {
        result = cal_dielectric_bsdf(isect, wi, wo, pdf, ix, iy, frame_id);
    }
    return result;
}