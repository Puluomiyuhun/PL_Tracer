#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include"myShader.h"
#include"myTexture.h"
#include"myCamera.h"
#include"myModel.h"
#include"myScene.h"

class myPipeline {
public:
    myPipeline(){}
    myPipeline(myScene *sc):scene(sc) 
    {
        setupMainShader();
        setupFrameBuffer();
        sphereToCubemap();
        setupCubemap();
        setupBloom();
        setupShadowMap();
	}
    void render() {
        time += 1;
        /////////////////////////这里开始平行光源视角渲染
        glEnable(GL_DEPTH_TEST);
        shader_dir_light.use();
        myLight* light = scene->lights[0];
        glViewport(0, 0, light->getShadowMap().x, light->getShadowMap().y);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        glm::mat4 lightProjection = glm::ortho(light->getOrtho().x, light->getOrtho().y, light->getOrtho().z, light->getOrtho().w, light->getPlane().x, light->getPlane().y);
        glm::mat4 lightView = glm::lookAt(light->getPos(), light->getPos() + light->getDir(), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 lightSpaceMatrix = lightProjection * lightView;
        shader_dir_light.setMatrix("lightSpaceMatrix", lightSpaceMatrix);
        for (int i = 0; i < scene->models.size(); i++) {
            glm::mat4 parent_model = transform(scene->models[i].pos, scene->models[i].rot, scene->models[i].scale);
            for (int j = 0; j < scene->models[i].meshes.size(); j++) {
                glm::mat4 leaf_model = transform(scene->models[i].meshes[j].pos, scene->models[i].meshes[j].rot, scene->models[i].meshes[j].scale);
                glm::mat4 model = parent_model * leaf_model;
                shader_dir_light.setMatrix("model", model);
                scene->models[i].meshes[j].Draw(shader_dir_light);
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        /////////////////////////这里开始点光源视角渲染
        shader_point_light.use();
        light = scene->lights[1];
        glViewport(0, 0, light->getShadowMap().x, light->getShadowMap().y);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO2);
        glClear(GL_DEPTH_BUFFER_BIT);
        GLfloat aspect = (GLfloat)light->getShadowMap().x / (GLfloat)light->getShadowMap().y;
        glm::mat4 shadowProj = glm::perspective(glm::radians(90.0f), aspect, light->getPlane().x, light->getPlane().y);
        std::vector<glm::mat4> shadowTransforms;
        glm::vec3 lightPos = light->getPos();
        shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(-1.0, 0.0, 0.0), glm::vec3(0.0, -1.0, 0.0)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 1.0)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, -1.0, 0.0), glm::vec3(0.0, 0.0, -1.0)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, -1.0, 0.0)));
        shadowTransforms.push_back(shadowProj * glm::lookAt(lightPos, lightPos + glm::vec3(0.0, 0.0, -1.0), glm::vec3(0.0, -1.0, 0.0)));
        for (GLuint i = 0; i < 6; ++i)
            shader_point_light.setMatrix("shadowMatrices[" + std::to_string(i) + "]", shadowTransforms[i]);
        shader_point_light.setFloat("far_plane", light->getPlane().y);
        for (int i = 0; i < scene->models.size(); i++) {
            glm::mat4 parent_model = transform(scene->models[i].pos, scene->models[i].rot, scene->models[i].scale);
            for (int j = 0; j < scene->models[i].meshes.size(); j++) {
                glm::mat4 leaf_model = transform(scene->models[i].meshes[j].pos, scene->models[i].meshes[j].rot, scene->models[i].meshes[j].scale);
                glm::mat4 model = parent_model * leaf_model;
                shader_point_light.setMatrix("model", model);
                scene->models[i].meshes[j].Draw(shader_point_light);
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);


        /////////////////////////这里开始场景渲染
        glViewport(0, 0, screenWidth, screenHeight);
        glBindFramebuffer(GL_FRAMEBUFFER, colorFBO);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = scene->cameras[0].getView();
        glm::mat4 projection = glm::identity<glm::mat4>();
        for (int i = 0; i < scene->models.size(); i++) {
            glm::mat4 parent_model = transform(scene->models[i].pos, scene->models[i].rot, scene->models[i].scale);
            for (int j = 0; j < scene->models[i].meshes.size(); j++) {
                myShader shader = scene->shaders[scene->models[i].meshes[j].use_shader];
                shader.use();
                shader.setFloat("time", glfwGetTime());
                /*shader光源设置*/
                shader.setVec3("dl[0].dir", scene->lights[0]->getDir());
                shader.setVec3("dl[0].color", scene->lights[0]->getColor() * scene->lights[0]->getIntensity());
                shader.setVec3("pl[0].pos", scene->lights[1]->getPos());
                shader.setVec3("pl[0].color", scene->lights[1]->getColor() * scene->lights[1]->getIntensity());
                shader.setFloat("pl[0].constant", scene->lights[1]->getConstant());
                shader.setFloat("pl[0].linear", scene->lights[1]->getLinear());
                shader.setFloat("pl[0].quadratic", scene->lights[1]->getQuadratic());

                /*shader相机设置*/
                shader.setVec3("cameraPos", scene->cameras[0].cameraPos);
                /*shader空间变换设置*/
                glm::mat4 leaf_model = transform(scene->models[i].meshes[j].pos, scene->models[i].meshes[j].rot, scene->models[i].meshes[j].scale);
                glm::mat4 model = parent_model * leaf_model;
                projection = glm::perspective(scene->cameras[0].fov, screenWidth / screenHeight, 0.1f, 500.0f);
                shader.setMatrix("model", model);
                shader.setMatrix("view", view);
                shader.setMatrix("projection", projection);

                /*shader材质参数设置*/
                glActiveTexture(GL_TEXTURE7);
                glBindTexture(GL_TEXTURE_2D, depthMap);
                shader.setInt("dir_shadowMap", 7);
                glActiveTexture(GL_TEXTURE8);
                glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubemap);
                shader.setInt("point_shadowMap", 8);
                glActiveTexture(GL_TEXTURE9);
                glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap_convolution);
                shader.setInt("diffuse_convolution", 9);
                glActiveTexture(GL_TEXTURE10);
                glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap_mipmap);
                shader.setInt("reflect_mipmap", 10);
                glActiveTexture(GL_TEXTURE11);
                glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
                shader.setInt("reflect_lut", 11);
                shader.setMatrix("lightSpaceMatrix", lightSpaceMatrix);
                shader.setFloat("far_plane", light->getPlane().y);
                shader.setFloat("totalTime", time);
                scene->models[i].meshes[j].Draw(shader);
            }
        }

        /////////////////////////这里开始立方体环境贴图渲染
        /*立方体贴图背景渲染*/
        glDepthFunc(GL_LEQUAL);
        shader_cubemap.use();
        shader_cubemap.setInt("cubeTexture", 0);
        view = glm::mat4(glm::mat3(scene->cameras[0].getView()));
        shader_cubemap.setMatrix("view", view);
        shader_cubemap.setMatrix("projection", projection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap);
        glBindVertexArray(skyboxVAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        /////////////////////////泛光的高斯模糊
        glDepthFunc(GL_LESS);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDisable(GL_DEPTH_TEST);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        GLboolean horizontal = true, first_iteration = true;
        GLuint amount = 0;
        shader_bloom.use();
        for (GLuint i = 0; i < amount; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[horizontal]);
            shader_bloom.setBool("horizontal", horizontal);
            glBindTexture(GL_TEXTURE_2D, first_iteration ? colorTexture[1] : pingpongTexture[!horizontal]);
            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            horizontal = !horizontal;
            if (first_iteration)
                first_iteration = false;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        /////////////////////////这里开始将渲染结果输出到屏幕上
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        shader_post.use();
        shader_post.setInt("screenTexture", 0);
        shader_post.setInt("bloomTexture", 1);
        shader_post.setFloat("exposure", 1.0f);
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTexture[0]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, pingpongTexture[0]);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
private:
	void setupMainShader() {
        shader_post = myShader("shader/postprocess.vsh", "shader/postprocess.fsh");
        shader_cubemap = myShader("shader/cubemap.vsh", "shader/cubemap.fsh");
        shader_hdr = myShader("shader/hdr.vsh", "shader/hdr.fsh");
        shader_hdr_convolution = myShader("shader/hdr.vsh", "shader/hdr_convolution.fsh");
        shader_hdr_mipmap = myShader("shader/hdr.vsh", "shader/hdr_mipmap.fsh");
        shader_lut = myShader("shader/lut.vsh", "shader/lut.fsh");
        shader_dir_light = myShader("shader/dir_light.vsh", "shader/dir_light.fsh");
        shader_point_light = myShader("shader/point_light.vsh", "shader/point_light.fsh", "shader/point_light.gsh");
        shader_bloom = myShader("shader/bloom.vsh", "shader/bloom.fsh");
	}
    void setupFrameBuffer() {
        /*屏幕的纹理模型设置*/
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        /*帧缓冲设置*/
        glGenFramebuffers(1, &colorFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, colorFBO);
        /*帧纹理设置，将帧缓冲的颜色信息写入*/
        glGenTextures(2, colorTexture);
        for (unsigned int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, colorTexture[i]);
            //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screenWidth, screenHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorTexture[i], 0);
        }
        GLuint attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, attachments);
        /*渲染缓冲设置，接收深度值和模板值*/
        unsigned int rbo;
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, screenWidth, screenHeight);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
    }
    void sphereToCubemap() {
        glGenVertexArrays(1, &skyboxVAO);
        glGenBuffers(1, &skyboxVBO);
        glBindVertexArray(skyboxVAO);
        glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

        glGenFramebuffers(1, &captureFBO);
        glGenRenderbuffers(1, &captureRBO);
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 1024, 1024);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);
        captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
        captureViews[0] = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        captureViews[1] = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        captureViews[2] = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        captureViews[3] = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
        captureViews[4] = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        captureViews[5] = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f));
        if (scene->hdrs.type == 0) {
            hdrCubemap = scene->hdrs.id;
            return;
        }
        /*球状HDR的设置*/
        glGenTextures(1, &hdrCubemap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap);
        for (unsigned int i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                1024, 1024, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        /*将球状贴图读成cubemap*/
        unsigned int hdrTexture = scene->hdrs.id;
        shader_hdr.use();
        shader_hdr.setInt("equirectangularMap", 0);
        shader_hdr.setMatrix("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, hdrTexture);
        glViewport(0, 0, 1024, 1024);
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        for (unsigned int i = 0; i < 6; ++i)
        {
            shader_hdr.setMatrix("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, hdrCubemap, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glBindVertexArray(skyboxVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void setupCubemap() {
        /*环境贴图模型设置*/
        const int convolution_size = 32;
        glGenTextures(1, &hdrCubemap_convolution);
        glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap_convolution);
        for (unsigned int i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F,
                convolution_size, convolution_size, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, convolution_size, convolution_size);
        glViewport(0, 0, convolution_size, convolution_size);
        shader_hdr_convolution.use();
        shader_hdr_convolution.setInt("environmentMap", 0);
        shader_hdr_convolution.setMatrix("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap);
        for (unsigned int i = 0; i < 6; ++i)
        {
            shader_hdr_convolution.setMatrix("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, hdrCubemap_convolution, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glBindVertexArray(skyboxVAO);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        /*预处理，镜面反射项的卷积*/
        glGenTextures(1, &hdrCubemap_mipmap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap_mipmap);
        for (unsigned int i = 0; i < 6; ++i)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, 128, 128, 0, GL_RGB, GL_FLOAT, nullptr);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

        shader_hdr_mipmap.use();
        shader_hdr_mipmap.setInt("environmentMap", 0);
        shader_hdr_mipmap.setMatrix("projection", captureProjection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, hdrCubemap);

        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        unsigned int maxMipLevels = 5;
        for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
        {
            unsigned int mipWidth = 128 * std::pow(0.5, mip);
            unsigned int mipHeight = 128 * std::pow(0.5, mip);
            glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
            glViewport(0, 0, mipWidth, mipHeight);

            float roughness = (float)mip / (float)(maxMipLevels - 1);
            shader_hdr_mipmap.setFloat("roughness", roughness);
            for (unsigned int i = 0; i < 6; ++i)
            {
                shader_hdr_mipmap.setMatrix("view", captureViews[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, hdrCubemap_mipmap, mip);

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glBindVertexArray(skyboxVAO);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        /*Lut预计算*/
        glGenTextures(1, &brdfLUTTexture);

        glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

        glViewport(0, 0, 512, 512);
        shader_lut.use();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void setupBloom() {
        /*泛光模糊缓冲*/
        glGenFramebuffers(2, pingpongFBO);
        glGenTextures(2, pingpongTexture);
        for (GLuint i = 0; i < 2; i++)
        {
            glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i]);
            glBindTexture(GL_TEXTURE_2D, pingpongTexture[i]);
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB16F, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, NULL
            );
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongTexture[i], 0
            );
        }
    }
    void setupShadowMap() {
        /*平行光源阴影缓冲设置*/
        glGenFramebuffers(1, &depthMapFBO);
        /*平行光源阴影纹理设置*/
        glGenTextures(1, &depthMap);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, scene->lights[0]->shadowmap_resolution.x, scene->lights[0]->shadowmap_resolution.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        GLfloat borderColor[] = { 1.0, 1.0, 1.0, 1.0 };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        /*点光源阴影缓冲设置*/
        glGenFramebuffers(1, &depthMapFBO2);
        /*点光源阴影纹理设置*/
        glGenTextures(1, &depthCubemap);
        glBindTexture(GL_TEXTURE_CUBE_MAP, depthCubemap);
        for (GLuint i = 0; i < 6; ++i)
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT,
                scene->lights[0]->shadowmap_resolution.x, scene->lights[0]->shadowmap_resolution.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO2);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthCubemap, 0);
        glDrawBuffer(GL_NONE);
        glReadBuffer(GL_NONE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    

private:
    float screenWidth = 1920, screenHeight = 960;

    myShader shader_post;
    myShader shader_cubemap;
    myShader shader_hdr;
    myShader shader_hdr_convolution;
    myShader shader_hdr_mipmap;
    myShader shader_lut;
    myShader shader_dir_light;
    myShader shader_point_light;
    myShader shader_bloom;

    myScene *scene;

    /*矩形、正方体的绑定数据*/
    unsigned int quadVAO, quadVBO;
    unsigned int skyboxVAO, skyboxVBO;

    /*颜色缓冲*/
    unsigned int colorFBO;
    unsigned int colorTexture[2];

    /*环境贴图*/
    unsigned int captureFBO, captureRBO;
    glm::mat4 captureViews[6];
    glm::mat4 captureProjection;
    unsigned int hdrCubemap;
    unsigned int hdrCubemap_convolution;
    unsigned int hdrCubemap_mipmap;
    unsigned int brdfLUTTexture;

    /*bloom数据*/
    GLuint pingpongFBO[2];
    GLuint pingpongTexture[2];

    /*shadoow数据*/
    GLuint depthMapFBO, depthMapFBO2;
    GLuint depthMap, depthCubemap;

    float time = 0;

};