#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include "Material_def.h"

namespace osc {
    using namespace gdt;

    /*一个TriangleMesh绑定一个SBT项。这里是TriangleMesh的模型参数*/
    struct TriangleMesh {
        std::vector<vec3f> vertex;         //顶点组
        std::vector<vec3f> normal;         //法线组
        std::vector<vec2f> texcoord;       //uv组
        std::vector<vec3i> index;          //三角面的索引号

        material_mes mat_mes;              //材质实例。这里也明确了：一个TriangleMesh只绑定一个材质
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

    /*包含了场景中所有的TriangleMesh、所有的纹理、环境贴图、灯光等信息*/
    struct Model {
        ~Model()
        {
            for (auto mesh : meshes) delete mesh;
            for (auto texture : textures) delete texture;
        }

        std::vector<TriangleMesh*> meshes;       //从.obj中拆解出来的所有TriangleMesh
        std::vector<Texture*>      textures;     //从.obj中分析出来的所有纹理
        Texture* envmap;     //环境贴图
        box3f bounds;        //包围盒，即能包住整个场景所有模型的最小立方体，该立方体与xyz三个轴平行
    };
    int loadEnvmap(Model* model, const std::string& Path);
    Model* loadOBJ(const std::string& objFile, material_kind mat_kind);
}
