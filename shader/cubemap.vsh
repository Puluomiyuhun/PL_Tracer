#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 pos;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    pos = aPos;
    vec4 p = projection * view * vec4(aPos, 1.0);
    gl_Position = p.xyww;
}