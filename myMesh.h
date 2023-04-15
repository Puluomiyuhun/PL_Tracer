#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "myTexture.h"
#include "myShader.h"
using namespace std;

struct vertice {
	glm::vec3 pos;
	glm::vec2 uv;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec3 bitangent;
};

class myMesh {
public:
	myMesh() {
		
	}
	myMesh(glm::vec3 pos_offset, glm::vec3 size, myTexture tex) {
		Cube(pos_offset, size, tex);
	}

	myMesh(string str, vector<vertice> vertices, vector<unsigned int> indices, vector<myTexture> textures)
	{
		this->name = str;
		this->vertice_struct = vertices;
		this->indice_struct = indices;
		this->texture_struct = textures;
		this->pos = glm::vec3(0, 0, 0);
		this->rot = glm::vec3(0, 0, 0);
		this->scale = glm::vec3(1, 1, 1);
	}
	void Draw(myShader shader) {
		shader.setVec4("material.diffuse", diffuse);
		shader.setVec3("material.specular", specular);
		shader.setFloat("material.roughness", roughness);
		shader.setFloat("material.metallic", metallic);
		shader.setFloat("material.ambient", ambient);
		shader.setBool("material.diffuse_texture_use", false);
		shader.setBool("material.specular_texture_use", false);
		shader.setBool("material.ambient_texture_use", false);
		shader.setBool("material.normal_texture_use", false);
		shader.setBool("material.metallic_texture_use", false);
		shader.setBool("material.roughness_texture_use", false);
		for (unsigned int i = 0; i < texture_struct.size(); i++)
		{
			glActiveTexture(GL_TEXTURE0 + i);
			string name = texture_struct[i].type;
			shader.setInt(("material." + name).c_str(), i);
			shader.setBool(("material." + name + "_use").c_str(), true);
			glBindTexture(GL_TEXTURE_2D, texture_struct[i].id);
			//cout << name << endl;
		}
		glActiveTexture(GL_TEXTURE0);

		// 绘制网格
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, indice_struct.size(), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
	void Setup() {
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		glBufferData(GL_ARRAY_BUFFER, vertice_struct.size() * sizeof(vertice), &vertice_struct[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indice_struct.size() * sizeof(unsigned int), &indice_struct[0], GL_STATIC_DRAW);

		// 顶点位置
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertice), (void*)0);
		// 顶点纹理坐标
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertice), (void*)offsetof(vertice, uv));
		// 顶点法线
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertice), (void*)offsetof(vertice, normal));
		// 顶点切线
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertice), (void*)offsetof(vertice, tangent));
		// 顶点副切线
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(vertice), (void*)offsetof(vertice, bitangent));

		glBindVertexArray(0);
	}
public:
	string name;
	vector<vertice> vertice_struct;
	vector<unsigned int> indice_struct;

	unsigned int VAO, VBO, EBO;
	int use_shader = 0;

	glm::vec3 pos, rot, scale;
	vector<myTexture> texture_struct;
	glm::vec4 diffuse;
	glm::vec3 specular;
	float roughness;
	float metallic;
	float ambient = 0.8f;

private:
	void Cube(glm::vec3 pos_offset, glm::vec3 size, myTexture tex) {
		vertice_struct.clear();
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(0.0f,0.0f,-1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f, -0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(0.0f,0.0f,-1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f, -0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(0.0f,0.0f,-1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f, -0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(0.0f,0.0f,-1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(0.0f,0.0f,-1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(0.0f,0.0f,-1.0f) });

		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f,  0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(0.0f,0.0f,1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(0.0f,0.0f,1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(0.0f,0.0f,1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(0.0f,0.0f,1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f,  0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(0.0f,0.0f,1.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f,  0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(0.0f,0.0f,1.0f) });

		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(-1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f, -0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(-1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(-1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(-1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f,  0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(-1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(-1.0f,0.0f,0.0f) });

		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f, -0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f,  0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(1.0f,0.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(1.0f,0.0f,0.0f) });

		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(0.0f,-1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f, -0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(0.0f,-1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(0.0f,-1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f, -0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(0.0f,-1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f,  0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(0.0f,-1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f, -0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(0.0f,-1.0f,0.0f) });

		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(0.0f,1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f, -0.5f),glm::vec2(1.0f, 1.0f),glm::vec3(0.0f,1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(0.0f,1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(0.5f,  0.5f,  0.5f),glm::vec2(1.0f, 0.0f),glm::vec3(0.0f,1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f,  0.5f),glm::vec2(0.0f, 0.0f),glm::vec3(0.0f,1.0f,0.0f) });
		vertice_struct.push_back(vertice{ glm::vec3(-0.5f,  0.5f, -0.5f),glm::vec2(0.0f, 1.0f),glm::vec3(0.0f,1.0f,0.0f) });

		for (int i = 0; i < vertice_struct.size(); i++) {
			vertice_struct[i].pos = vertice_struct[i].pos * size + pos_offset;
		}
		for (unsigned int i = 0; i < vertice_struct.size(); i++)
			indice_struct.push_back(i);
		texture_struct.push_back(tex);
	}

};

float quadVertices[] = {
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		 1.0f, -1.0f,  1.0f, 0.0f,
		 1.0f,  1.0f,  1.0f, 1.0f
};

float skyboxVertices[] = {
	// positions          
	-1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,

	-1.0f, -1.0f,  1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	-1.0f,  1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,

	 1.0f, -1.0f, -1.0f,
	 1.0f, -1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,

	-1.0f, -1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,

	-1.0f,  1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f, -1.0f,

	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f
};