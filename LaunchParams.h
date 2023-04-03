#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"
#include "Material_def.h"

namespace osc {
    using namespace gdt;
    //射线类型
    enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE = 1, RAY_TYPE_COUNT };

    //每个TriangleMesh在SBT表中要绑定的材质数据
    struct TriangleMeshSBTData {
        vec3f* vertex;
        vec3f* normal;
        vec2f* texcoord;
        vec3i* index;
        material_mes mat_mes;
    };

    //在渲染开始前，需要提前传入Gpu的一些参数，Gpu在shader中可以获取或者写入这些参数
    struct LaunchParams
    {
        int numPixelSamples = 1;             //每帧的采样数
        int maxBounce = 5;                   //最大追踪次数
        float lightness_change = 0.0f;       //亮度后处理
        float saturate_change = 0.0f;        //饱和度后处理
        float contrast_change = 0.1f;        //对比度后处理

        /*frame是用来存某一帧的最终渲染结果的，Gpu往里写，Cpu往外拿*/
        struct {
            int       frameID = 0;           //帧编号
            float4* colorBuffer;             //直接渲染得到的结果
            float4* renderBuffer;            //colorBuffer经过后处理得到的结果
            vec2i     size;                  //画面的尺寸
        } frame;

        /*这个camera相机规定了相机位置、方向，以及相机视口的宽高尺寸与方向*/
        struct {
            vec3f position;
            vec3f direction;
            vec3f horizontal;
            vec3f vertical;
        } camera;

        /*光源*/
        struct {
            vec3f origin, du, dv, power;
        } light;

        OptixTraversableHandle traversable;  //求交加速结构，里面包含了场景的所有三角面
    };

}