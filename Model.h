#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include "Material_def.h"

namespace osc {
    using namespace gdt;

    /*һ��TriangleMesh��һ��SBT�������TriangleMesh��ģ�Ͳ���*/
    struct TriangleMesh {
        std::vector<vec3f> vertex;         //������
        std::vector<vec3f> normal;         //������
        std::vector<vec2f> texcoord;       //uv��
        std::vector<vec3i> index;          //�������������

        material_mes mat_mes;              //����ʵ��������Ҳ��ȷ�ˣ�һ��TriangleMeshֻ��һ������
    };

    struct Texture {
        ~Texture()
        {
            if (pixel) delete[] pixel;
        }

        uint32_t* pixel{ nullptr };
        vec2i     resolution{ -1 };
    };
    struct QuadLight {
        vec3f origin, du, dv, power;
    };

    /*�����˳��������е�TriangleMesh�����е�����������ͼ���ƹ����Ϣ*/
    struct Model {
        ~Model()
        {
            for (auto mesh : meshes) delete mesh;
            for (auto texture : textures) delete texture;
        }

        std::vector<TriangleMesh*> meshes;       //��.obj�в�����������TriangleMesh
        std::vector<Texture*>      textures;     //��.obj�з�����������������
        Texture* envmap;     //������ͼ
        box3f bounds;        //��Χ�У����ܰ�ס������������ģ�͵���С�����壬����������xyz������ƽ��
    };
    int loadEnvmap(Model* model, const std::string& Path);
    Model* loadOBJ(const std::string& objFile, material_kind mat_kind);
}
