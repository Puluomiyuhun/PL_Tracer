// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "gdt/math/AffineSpace.h"
#include <vector>
#include "Material_def.h"

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
    using namespace gdt;

    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;

        // material data:
        material_mes mat_mes;
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
    struct Model {
        ~Model()
        {
            for (auto mesh : meshes) delete mesh;
            for (auto texture : textures) delete texture;
        }

        std::vector<TriangleMesh*> meshes;
        std::vector<Texture*>      textures;
        Texture* envmap;
        //! bounding box of all vertices in the model
        box3f bounds;
    };
    int loadEnvmap(Model* model, const std::string& Path);
    Model* loadOBJ(const std::string& objFile, material_kind mat_kind);
}
