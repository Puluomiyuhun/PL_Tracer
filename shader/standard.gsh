#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT {
   vec3 normal;
   vec2 texcoord;
   vec3 FragPos;  
   mat3 TBN;
} gs_in[];

out vec3 f_normal;
out vec2 f_texcoord;
out vec3 f_FragPos; 
out mat3 f_TBN;

uniform float time;

vec4 explode(vec4 position, vec3 normal)
{
    return position;
    float magnitude = 2.0;
    vec3 direction = normal * ((sin(time) + 1.0) / 2.0) * magnitude; 
    return position + vec4(direction, 0.0);
}

vec3 GetNormal()
{
   vec3 a = vec3(gl_in[0].gl_Position) - vec3(gl_in[1].gl_Position);
   vec3 b = vec3(gl_in[2].gl_Position) - vec3(gl_in[1].gl_Position);
   return normalize(cross(a, b));
}

void main() {    
    vec3 normal = GetNormal();

    gl_Position = explode(gl_in[0].gl_Position, normal);
    f_normal = gs_in[0].normal;
    f_texcoord = gs_in[0].texcoord;
    f_FragPos = gs_in[0].FragPos;
    f_TBN = gs_in[0].TBN;
    EmitVertex();
    gl_Position = explode(gl_in[1].gl_Position, normal);
    f_normal = gs_in[1].normal;
    f_texcoord = gs_in[1].texcoord;
    f_FragPos = gs_in[1].FragPos;
    f_TBN = gs_in[1].TBN;
    EmitVertex();
    gl_Position = explode(gl_in[2].gl_Position, normal);
    f_normal = gs_in[2].normal;
    f_texcoord = gs_in[2].texcoord;
    f_FragPos = gs_in[2].FragPos;
    f_TBN = gs_in[2].TBN;
    EmitVertex();
    EndPrimitive();
}