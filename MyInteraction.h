#pragma once
#include "gdt/math/vec.h"
#include "math.h"
#include "LaunchParams.h"

using namespace gdt;

/*���ߣ�������������ꡢ���䷽�������ʱ���������optix��api�Ĳ���֮һ��*/
struct Ray
{
    vec3f origin;
    vec3f direction;
    float tmax = FLT_MAX;
};

/*����Ϣ���������󽻵���ʱ������Щ��Ϣ��¼��Interaction��*/
struct Interaction
{
    float bias = 0.0001f;      //�������ߵ�ƫ�������������

    float distance;            //�������߳����ľ���
    vec3f position;            //�󽻵����������
    vec3f geomNormal;          //�󽻵�ķ��ߣ�ǿ�н����������߷���
    vec3f realNormal;          //�󽻵�ķ��ߣ�û����������
    vec2f texcoord;            //�󽻵��uvֵ

    material_mes mat_mes;      //�󽻵�Ĳ�����Ϣ

    /*��֪��һ�ι��ߵĵ��䷽�򣬷���һ��Ray*/
    __forceinline__ __device__ Ray spawn_ray(const vec3f& wi) const
    {
        vec3f N = geomNormal;
        if (dot(wi, geomNormal) < 0.0f)
        {
            N = -geomNormal;
        }

        Ray ray;
        ray.origin = position + N * bias;   //����������bias�������´ε���ֱ��ԭ������
        ray.direction = wi;
        ray.tmax = FLT_MAX;
        return ray;
    }
};