#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;
uniform sampler2D bloomTexture;
uniform float exposure;

void main()
{ 
    vec3 colors = vec3(texture(screenTexture, TexCoords));
    vec3 bloomColor = texture(bloomTexture, TexCoords).rgb;
    colors += bloomColor;
    //这里对colors进行后处理
    vec3 mapped = vec3(1.0) - exp(-colors * exposure);
    FragColor.rgb = pow(mapped,vec3(1/2.2));
    FragColor.a = 1.0;
}