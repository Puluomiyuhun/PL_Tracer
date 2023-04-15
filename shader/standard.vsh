#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

out VS_OUT {
   vec3 normal;
   vec2 texcoord;
   vec3 FragPos;  
   mat3 TBN;
}vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
   gl_Position = projection * view * model * vec4(aPos, 1.0);
   vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
   vs_out.normal = mat3(transpose(inverse(model))) * aNormal;;
   vs_out.texcoord = aUV;

   vec3 T = normalize(vec3(model * vec4(aTangent,   0.0)));
   vec3 N = normalize(vec3(model * vec4(aNormal,    0.0)));
   T = normalize(T - dot(T, N) * N);
   vec3 B = cross(T, N);
   vs_out.TBN = mat3(T, B, N);
}