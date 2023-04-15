#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct Euler {
	double pitch;
	double yaw;
	double roll;
};

class myCamera {
public:
	myCamera(string str, glm::vec3 Pos, glm::vec3 Front, glm::vec3 Up, Euler eu, double fov_) {
		name = str;
		cameraPos = Pos;
		cameraFront = Front;
		cameraUp = Up;
		cameraEuler = eu;
		fov = fov_;
	}
	glm::mat4 getView() {
		return glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
	}
	glm::vec3 cameraPos;
	glm::vec3 cameraFront;
	glm::vec3 cameraUp;
	Euler cameraEuler;
	float fov = 45.0f;

	string name;
};