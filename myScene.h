#pragma once
#include"myShader.h"
#include"myTexture.h"
#include"myCamera.h"
#include"myModel.h"
#include"myLight.h"
class myScene {
public:
    myScene(){}
	myScene(int kind) {
		if (kind == 0) {
            /*Shader管线创建*/
            shaders.push_back(myShader("shader/standard.vsh", "shader/standard.fsh", "shader/standard.gsh"));
            shaders[0].setName("PBR_shader");
            shaders.push_back(myShader("shader/water.vsh", "shader/water.fsh", "shader/water.gsh"));
            shaders[1].setName("water_shader");
            //models.push_back(myModel("D:/blender/Cerberus_by_Andrew_Maximov/gun.obj", false));
            models.push_back(myModel("D:/blender/lyzl.obj", false));
            //models.push_back(myModel("D:/blender/DamagedHelmet/DamagedHelmet.obj", false));
            cameras.push_back(myCamera("MainCamera", glm::vec3(5.0f, -3.0f, 30.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f), Euler{0.0f,-90.0f,0.0f}, 45.0f));
            lights.push_back(new myDirLight("dir", glm::vec3(10, 10, 10), glm::vec3(-1, -1, -1), glm::vec3(1, 1, 1), 1, glm::vec2(1,500), glm::vec4(-60, 60, -60, 60), glm::vec2(1024, 1024)));
            lights.push_back(new myPointLight("point", glm::vec3(0, 8, 10), glm::vec3(1, 1, 1), 1, 1.0f, 0.1f, 0.01f, glm::vec2(1,100), glm::vec2(1024, 1024)));
            //hdrs = myHDR{loadHDR( "G:/A_project_blender/abandoned_tank_farm_04_2k.hdr"), 1 };
            hdrs = myHDR{ loadHDR("D:/mitsuba2_project/ball/5.hdr"), 1 };
            for (int i = 0; i < models.size(); i++) {
                for (int j = 0; j < models[i].meshes.size(); j++) {
                    for (int k = 0; k < models[i].meshes[j].texture_struct.size(); k++) {
                        textures.push_back(models[i].meshes[j].texture_struct[k]);
                    }
                }
            }
            /*vector<std::string> faces
            {
                "D:/cg_opengl/OpenglRenderer/resources/cubemap/right.jpg",
                "D:/cg_opengl/OpenglRenderer/resources/cubemap/left.jpg",
                "D:/cg_opengl/OpenglRenderer/resources/cubemap/top.jpg",
                "D:/cg_opengl/OpenglRenderer/resources/cubemap/bottom.jpg",
                "D:/cg_opengl/OpenglRenderer/resources/cubemap/front.jpg",
                "D:/cg_opengl/OpenglRenderer/resources/cubemap/back.jpg"
            };
            hdrs = myHDR{ loadCubemap(faces) ,0};*/
		}
	}

    vector<myShader> shaders;
    vector<myModel> models;
    vector<myCamera> cameras;
    vector<myTexture> textures;
    vector<myLight*> lights;
    myHDR hdrs;
};

glm::mat4 transform(glm::vec3 p, glm::vec3 r, glm::vec3 s ) {
    glm::mat4 model = glm::identity<glm::mat4>();
    model = glm::translate(model, p);
    model = glm::rotate(model, glm::radians(r.x), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(r.y), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(r.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model =  glm::scale(model, s);
    return model;
}