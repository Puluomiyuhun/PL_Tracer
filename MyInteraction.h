#pragma once
#include "gdt/math/vec.h"
#include "math.h"
#include "LaunchParams.h"

using namespace gdt;

/*射线，包含出射点坐标、出射方向、最大求交时长（这个是optix求交api的参数之一）*/
struct Ray
{
    vec3f origin;
    vec3f direction;
    float tmax = FLT_MAX;
};

/*求交信息，当射线求交到面时，将这些信息记录到Interaction中*/
struct Interaction
{
    float bias = 0.0001f;      //出射射线的偏移量，下面解释

    float distance;            //距离射线出射点的距离
    vec3f position;            //求交点的世界坐标
    vec3f geomNormal;          //求交点的法线（强行矫正成与射线反向）
    vec3f realNormal;          //求交点的法线（没被矫正方向）
    vec2f texcoord;            //求交点的uv值

    material_mes mat_mes;      //求交点的材质信息

    /*已知下一次光线的弹射方向，返回一个Ray*/
    __forceinline__ __device__ Ray spawn_ray(const vec3f& wi) const
    {
        vec3f N = geomNormal;
        if (dot(wi, geomNormal) < 0.0f)
        {
            N = -geomNormal;
        }

        Ray ray;
        ray.origin = position + N * bias;   //如果不加这个bias，可能下次弹射直接原地求交了
        ray.direction = wi;
        ray.tmax = FLT_MAX;
        return ray;
    }
};