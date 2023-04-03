#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "gdt/math/AffineSpace.h"
#include "Model.h"

namespace osc {
    /*���峡���е����*/
    struct Camera {
        vec3f from;
        vec3f at;
        vec3f up;
    };
  /*! ��Ⱦ������ */
  class SampleRenderer
  {
  public:
    SampleRenderer(const Model* model, const QuadLight& light);

    void render();  //����optix�ںˣ�������Ⱦ�ĺ���

    void resize(const vec2i &newSize);     //������ڴ�С�仯�ˣ�������Ⱦ�����ĳߴ����

    void downloadPixels(vec4f h_pixels[]); //��gpu��������Ⱦ���

    void setCamera(const Camera& camera);  //����������ӽ�λ�ñ仯�ˣ���Ҫ�����趨���������ʼ�µ���Ⱦ����
  protected:
    void initOptix();             //��ʼ��optix�ں�
  
    void createTextures();        //������������Ҫ����gpu������

    void createContext();         //�����豸�������ģ���Gpu�豸

    void createModule();          //����Gpuģ�飬ģ����Ҫ��shader���루�����룩
    
    void createRaygenPrograms();  //����RP Shaderʵ��
    
    void createMissPrograms();    //����MP Shaderʵ��
    
    void createHitgroupPrograms();//����HP Shaderʵ��

    void createPipeline();        //������Shaderʵ�������������󶨵����ߵ���

    void buildSBT();              //��SBT�����������ǽ�ÿ��TriangleMesh��SBT�е�һ��Record��󶨣�����������ʲ���

    OptixTraversableHandle buildAccel();   //�������ٽṹ�����������ǰ�����TriangleMesh��������󶨳�optix�ļ��ٽṹ��ʽ

  /*������optix�Ĺ��߲�����ѧ��Dx�Ļ�Ӧ�ò�İ������Dx�Ĺ��ߴ�ͬС��*/
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

    /*����ÿһ��Ҫ��Gpu��ռ���ڴ�Ĳ�������Ҫ����һ����Ӧ��Buffer����*/
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

    Camera lastSetCamera;    //���һ�α䶯�����λ��

    const Model* model;
    std::vector<CUDABuffer> vertexBuffer;   //��������buffer
    std::vector<CUDABuffer> indexBuffer;    //��������buffer
    CUDABuffer asBuffer;                    //���ٽṹbuffer

    std::vector<CUDABuffer> normalBuffer;   //����buffer
    std::vector<CUDABuffer> texcoordBuffer; //uv-buffer
    std::vector<cudaArray_t>         textureArrays;   //��������
    std::vector<cudaTextureObject_t> textureObjects;  //�Ѿ�������ڴ������������gpu��ֱ�Ӳ���

  };

}
