#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"
#include "Model.h"

namespace osc {
    /*定义场景中的相机*/
    struct Camera {
        vec3f from;
        vec3f at;
        vec3f up;
    };
  /*! 渲染器核心 */
  class SampleRenderer
  {
  public:
    SampleRenderer(const Model* model, const QuadLight& light);

    void render();  //调用optix内核，进行渲染的函数

    void resize(const vec2i &newSize);     //如果窗口大小变化了，进行渲染画布的尺寸调整

    void downloadPixels(vec4f h_pixels[]); //从gpu端下载渲染结果

    void setCamera(const Camera& camera);  //场景中如果视角位置变化了，就要重新设定相机，并开始新的渲染任务
  protected:
    void initOptix();             //初始化optix内核
  
    void createTextures();        //创建、绑定所有要传进gpu的纹理

    void createContext();         //创建设备、上下文，绑定Gpu设备

    void createModule();          //创建Gpu模块，模块上要绑定shader代码（机器码）
    
    void createRaygenPrograms();  //创建RP Shader实例
    
    void createMissPrograms();    //创建MP Shader实例
    
    void createHitgroupPrograms();//创建HP Shader实例

    void createPipeline();        //将上述Shader实例连接起来，绑定到管线当中

    void buildSBT();              //绑定SBT，核心任务是将每个TriangleMesh和SBT中的一个Record项绑定，用于输入材质参数

    OptixTraversableHandle buildAccel();   //创建加速结构，核心任务是把所有TriangleMesh的三角面绑定成optix的加速结构形式

  /*下面是optix的管线参数，学过Dx的话应该不陌生，和Dx的管线大同小异*/
  protected:
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    OptixDeviceContext optixContext;

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions    = {};
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    /*对于每一个要在Gpu中占用内存的参数，都要申请一个对应的Buffer变量*/
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;

    CUDABuffer colorBuffer;
    CUDABuffer renderBuffer;

    Camera lastSetCamera;    //最后一次变动的相机位置

    const Model* model;
    std::vector<CUDABuffer> vertexBuffer;   //顶点数据buffer
    std::vector<CUDABuffer> indexBuffer;    //索引数据buffer
    CUDABuffer asBuffer;                    //加速结构buffer

    std::vector<CUDABuffer> normalBuffer;   //法线buffer
    std::vector<CUDABuffer> texcoordBuffer; //uv-buffer
    std::vector<cudaArray_t>         textureArrays;   //纹理数组
    std::vector<cudaTextureObject_t> textureObjects;  //已经申请好内存的纹理，可以在gpu端直接采样

  };

}
