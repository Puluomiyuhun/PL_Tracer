#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
class myLight {
public:
	myLight(){}
	virtual ~myLight() {}
	virtual int getKind() { return -1; }
	virtual string getName() { return ""; }
	virtual glm::vec3 getDir() { return glm::vec3(0, 0, 0); }
	virtual glm::vec3 getColor() { return glm::vec3(0, 0, 0); }
	virtual float getIntensity() { return 0; }
	virtual glm::vec3 getPos() { return glm::vec3(0, 0, 0); }
	virtual float getConstant() { return 1; }
	virtual float getLinear() { return 0; }
	virtual float getQuadratic() { return 0; }
	virtual glm::vec2 getPlane() { return glm::vec2(0, 0); }
	virtual glm::vec4 getOrtho() { return glm::vec4(0, 0, 0, 0); }
	virtual glm::vec2 getShadowMap() { return glm::vec2(256, 256); }

public:
	string name = "";
	int kind;
	glm::vec3 direction;
	glm::vec3 color;
	float intensity;
	glm::vec3 position;
	glm::vec2 plane;
	glm::vec4 ortho;
	glm::vec2 shadowmap_resolution;
	float constant;
	float linear;
	float quadratic;
};

class myDirLight : public myLight {
public:
	myDirLight(string str, glm::vec3 pos, glm::vec3 dir, glm::vec3 col, float inte, glm::vec2 pl, glm::vec4 orth,glm::vec2 shadowmap){
		kind = 0;
		name = str;
		position = pos;
		direction = dir;
		color = col;
		intensity = inte;
		plane = pl;
		ortho = orth;
		shadowmap_resolution = shadowmap;
	}
	virtual int getKind() { return kind; }
	virtual string getName() { return name; }
	virtual glm::vec3 getColor() { return color; }
	virtual float getIntensity() { return intensity; }
	virtual glm::vec3 getPos() { return position; }
	virtual glm::vec3 getDir() { return direction; }
	virtual glm::vec2 getPlane() { return plane; }
	virtual glm::vec4 getOrtho() { return ortho; }
	virtual glm::vec2 getShadowMap() { return shadowmap_resolution; }
};

class myPointLight : public myLight {
public:
	myPointLight(string str, glm::vec3 pos, glm::vec3 col, float inte, float c, float l, float q, glm::vec2 pl, glm::vec2 shadowmap) {
		kind = 1;
		name = str;
		position = pos;
		color = col;
		intensity = inte;
		plane = pl;
		shadowmap_resolution = shadowmap;
		constant = c;
		linear = l;
		quadratic = q;
		plane = pl;
	}

	virtual int getKind() { return kind; }
	virtual string getName() { return name; }
	virtual glm::vec3 getColor() { return color; }
	virtual float getIntensity() { return intensity; }
	virtual glm::vec3 getPos() { return position; }
	virtual float getConstant() { return constant; }
	virtual float getLinear() { return linear; }
	virtual float getQuadratic() { return quadratic; }
	virtual glm::vec2 getPlane() { return plane; }
	virtual glm::vec2 getShadowMap() { return shadowmap_resolution; }
};