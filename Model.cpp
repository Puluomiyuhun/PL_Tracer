#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"
#include <set>

/*小于号重载，模型index_t的排序*/
namespace std {
    inline bool operator<(const tinyobj::index_t& a, const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

namespace osc {
    /*由于一个TriangleMesh对应一个材质，然而导入的.obj可能包含多个材质，因此要根据材质对象拆解模型。AddVertex就是拆解模型时 为新模型添加顶点的函数*/
    int addVertex(TriangleMesh* mesh,
        tinyobj::attrib_t& attributes,
        const tinyobj::index_t& idx,
        std::map<tinyobj::index_t, int>& knownVertices)
    {
        if (knownVertices.find(idx) != knownVertices.end())
            return knownVertices[idx];

        const vec3f* vertex_array = (const vec3f*)attributes.vertices.data();
        const vec3f* normal_array = (const vec3f*)attributes.normals.data();
        const vec2f* texcoord_array = (const vec2f*)attributes.texcoords.data();

        int newID = (int)mesh->vertex.size();
        knownVertices[idx] = newID;

        mesh->vertex.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normal.size() < mesh->vertex.size())
                mesh->normal.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texcoord.size() < mesh->vertex.size())
                mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
        }

        if (mesh->texcoord.size() > 0)
            mesh->texcoord.resize(mesh->vertex.size());
        if (mesh->normal.size() > 0)
            mesh->normal.resize(mesh->vertex.size());

        return newID;
    }

    /*加载环境贴图*/
    int loadEnvmap(Model* model, const std::string& Path) {
        if (Path == "")
            return -1;
        vec2i res;
        int comp;
        unsigned char* image = stbi_load(Path.c_str(),
            &res.x, &res.y, &comp, STBI_rgb_alpha);
        if (image) {
            Texture* texture = new Texture;
            texture->resolution = res;
            texture->pixel = (uint32_t*)image;
            model->envmap = texture;
            return 1;
        }
        return 0;
    }

    /*加载贴图*/
    int loadTexture(Model* model,
        std::map<std::string, int>& knownTextures,
        const std::string& inFileName,
        const std::string& modelPath)
    {
        if (inFileName == "")
            return -1;

        if (knownTextures.find(inFileName) != knownTextures.end())
            return knownTextures[inFileName];

        std::string fileName = inFileName;
        // first, fix backspaces:
        for (auto& c : fileName)
            if (c == '\\') c = '/';
        if (fileName[1] != ':')
            fileName = modelPath + "/" + fileName;

        vec2i res;
        int   comp;
        unsigned char* image = stbi_load(fileName.c_str(),
            &res.x, &res.y, &comp, STBI_rgb_alpha);
        int textureID = -1;
        if (image) {
            textureID = (int)model->textures.size();
            Texture* texture = new Texture;
            texture->resolution = res;
            texture->pixel = (uint32_t*)image;

            /*纹理上下颠倒，因为原点一个在上面一个在下面*/
            for (int y = 0; y < res.y / 2; y++) {
                uint32_t* line_y = texture->pixel + y * res.x;
                uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
                for (int x = 0; x < res.x; x++) {
                    std::swap(line_y[x], mirrored_y[x]);
                }
            }
            model->textures.push_back(texture);
        }
        else {
            std::cout << GDT_TERMINAL_RED
                << "Could not load texture from " << fileName << "!"
                << GDT_TERMINAL_DEFAULT << std::endl;
        }

        knownTextures[inFileName] = textureID;
        return textureID;
    }

    /*model的读取。通过LoadObj读取.obj文件后，会得到若干个shape，每个shape会有若干个material*/
    Model* loadOBJ(const std::string& objFile, material_kind mat_kind)
    {
        Model* model = new Model;
        const std::string modelDir
            = objFile.substr(0, objFile.rfind('/') + 1);

        const std::string mtlDir
            = objFile.substr(0, objFile.rfind('/') + 1);
        PRINT(mtlDir);

        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err = "";

        bool readOK
            = tinyobj::LoadObj(&attributes,
                &shapes,
                &materials,
                &err,
                &err,
                objFile.c_str(),
                mtlDir.c_str(),
                true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + objFile + ":" + mtlDir + " : " + err);
        }

        if (materials.empty())
            throw std::runtime_error("could not parse materials ...");

        std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
        std::map<std::string, int>      knownTextures;        //knownXXX就是记录重复用的
        /*下面这个循环干的事：遍历每个shape，遍历每个shape中的material，根据material将模型拆解成TriangleMesh和material一一对应*/
        for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
            tinyobj::shape_t& shape = shapes[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids) {
                materialIDs.insert(faceMatID);
            }

            for (int materialID : materialIDs) {
                if (materialID == -1)
                    continue;

                std::map<tinyobj::index_t, int> knownVertices;
                TriangleMesh* mesh = new TriangleMesh;

                for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                        addVertex(mesh, attributes, idx1, knownVertices),
                        addVertex(mesh, attributes, idx2, knownVertices));
                    mesh->index.push_back(idx);
                    /*根据材质类型决定要往mat_mes中写入哪些参数*/
                    if (mat_kind == DIFFUSE) {
                        mesh->mat_mes.diffuse = (const vec3f&)materials[materialID].diffuse;
                        mesh->mat_mes.diffuseTextureID = loadTexture(model,
                            knownTextures,
                            materials[materialID].diffuse_texname,
                            modelDir);
                        if (mesh->mat_mes.diffuseTextureID != -1 && mesh->mat_mes.diffuse == vec3f(0.0f))
                            mesh->mat_mes.diffuse = 1.0f;
                        mesh->mat_mes.emitter = (const vec3f&)materials[materialID].emission;
                        mesh->mat_mes.emitterTextureID = loadTexture(model,
                            knownTextures,
                            materials[materialID].emissive_texname,
                            modelDir);
                        if (mesh->mat_mes.emitterTextureID != -1 && mesh->mat_mes.emitter == vec3f(0.0f))
                            mesh->mat_mes.emitter = 1.0f;
                    }
                    if (mat_kind == METAL && !materials.empty()) {
                        mesh->mat_mes.diffuse = (const vec3f&)materials[materialID].diffuse;
                        mesh->mat_mes.diffuseTextureID = loadTexture(model,
                            knownTextures,
                            materials[materialID].diffuse_texname,
                            modelDir);
                        mesh->mat_mes.roughness = materials[materialID].roughness;
                        if (mesh->mat_mes.diffuseTextureID != -1 && mesh->mat_mes.diffuse == vec3f(0.0f))
                            mesh->mat_mes.diffuse = 1.0f;
                        mesh->mat_mes.emitter = (const vec3f&)materials[materialID].emission;
                        mesh->mat_mes.emitterTextureID = loadTexture(model,
                            knownTextures,
                            materials[materialID].emissive_texname,
                            modelDir);
                        if (mesh->mat_mes.emitterTextureID != -1 && mesh->mat_mes.emitter == vec3f(0.0f))
                            mesh->mat_mes.emitter = 1.0f;
                    }
                    if (mat_kind == DIELECTRIC && !materials.empty()) {
                        mesh->mat_mes.diffuse = (const vec3f&)materials[materialID].diffuse;
                        mesh->mat_mes.diffuseTextureID = loadTexture(model,
                            knownTextures,
                            materials[materialID].diffuse_texname,
                            modelDir);
                        mesh->mat_mes.transparent = materials[materialID].transmittance[0];
                    }
                    mesh->mat_mes.mat_kind = mat_kind;
                }

                if (mesh->vertex.empty())
                    delete mesh;
                else
                    model->meshes.push_back(mesh);
            }
        }
        /*把所有三角面放进来，计算包围盒*/
        for (auto mesh : model->meshes)
            for (auto vtx : mesh->vertex)
                model->bounds.extend(vtx);

        std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
        return model;
    }
}
