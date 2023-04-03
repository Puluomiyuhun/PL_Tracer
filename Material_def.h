#pragma once
#include "gdt/math/vec.h"
#include <cuda_runtime.h>

using namespace gdt;

/*Ŀǰ��������в�������*/
enum material_kind
{
    DIFFUSE, METAL, DIELECTRIC       //�����䡢���淴�䡢͸�䣬���������׷����
};

/*���ʲ���*/
struct material_mes {
    material_kind mat_kind;          //�������ͣ�0��1��2͸

    vec3f diffuse = 1.0f;            //��������ɫ
    float roughness = 1.0f;          //�ֲڶ�
    float transparent = 0.0f;        //͸���
    float ior = 1.8f;                //������
    vec3f emitter = 0;               //�Է���

    int diffuseTextureID{ -1 };      //��������ͼ���
    int emitterTextureID{ -1 };      //�Է�����ͼ���
    cudaTextureObject_t diffuse_texture;    //��������ͼʵ��
    cudaTextureObject_t emitter_texture;    //�Է�����ͼʵ��
};