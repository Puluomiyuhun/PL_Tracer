#pragma once
#include "gdt/math/vec.h"
#include "math.h"

using namespace gdt;

__forceinline__ __device__ float my_min(const float a, const float b) {
    return a < b ? a : b;
}

__forceinline__ __device__ float length_squared(const vec3f v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/*已知入射向量、法线，求反射向量*/
__forceinline__ __device__ vec3f reflect(const vec3f v, const vec3f n){
    return v - 2 * dot(v, n) * n;
}

/*已知入射向量uv、法线、折射率，求折射向量*/
__forceinline__ __device__ vec3f refract(const vec3f uv, const vec3f n, double etai_over_etat){
    auto cos_theta = dot(-uv, n);
    vec3f r_out_perp = (float)etai_over_etat * (uv + cos_theta * n);
    vec3f r_out_parallel = (float)(-sqrt(fabs(1.0 - length_squared(r_out_perp)))) * n;
    return r_out_perp + r_out_parallel;
}

/*菲涅尔项的近似拟合*/
__forceinline__ __device__ double schlick(double cosine, double ref_idx){
    auto r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 *= r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}