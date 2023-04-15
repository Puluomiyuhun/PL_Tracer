#version 330 core
out vec4 FragColor;

in vec3 pos;

uniform samplerCube cubeTexture;

void main()
{ 
    FragColor = texture(cubeTexture, pos);
    //FragColor = textureLod(cubeTexture, pos, 3);
}