#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include "Material_def.h"

namespace osc {
    using namespace gdt;
    //��������
    enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE = 1, RAY_TYPE_COUNT };

    //ÿ��TriangleMesh��SBT����Ҫ�󶨵Ĳ�������
    struct TriangleMeshSBTData {
        vec3f* vertex;
        vec3f* normal;
        vec2f* texcoord;
        vec3i* index;
        material_mes mat_mes;
    };

    //����Ⱦ��ʼǰ����Ҫ��ǰ����Gpu��һЩ������Gpu��shader�п��Ի�ȡ����д����Щ����
    struct LaunchParams
    {
        int numPixelSamples = 1;             //ÿ֡�Ĳ�����
        int maxBounce = 5;                   //���׷�ٴ���
        float lightness_change = 0.0f;       //���Ⱥ���
        float saturate_change = 0.0f;        //���ͶȺ���
        float contrast_change = 0.1f;        //�ԱȶȺ���

        /*frame��������ĳһ֡��������Ⱦ����ģ�Gpu����д��Cpu������*/
        struct {
            int       frameID = 0;           //֡���
            float4* colorBuffer;             //ֱ����Ⱦ�õ��Ľ��
            float4* renderBuffer;            //colorBuffer��������õ��Ľ��
            vec2i     size;                  //����ĳߴ�
        } frame;

        /*���camera����涨�����λ�á������Լ�����ӿڵĿ�߳ߴ��뷽��*/
        struct {
            vec3f position;
            vec3f direction;
            vec3f horizontal;
            vec3f vertical;
        } camera;

        /*��Դ*/
        struct {
            vec3f origin, du, dv, power;
        } light;

        OptixTraversableHandle traversable;  //�󽻼��ٽṹ����������˳���������������
    };

}