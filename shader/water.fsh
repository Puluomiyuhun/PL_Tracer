#version 330 core
layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;
in vec3 f_normal;
in vec2 f_texcoord;
in vec3 f_FragPos;  
in mat3 f_TBN;
out vec4 FragColor;
uniform vec3 cameraPos;
uniform mat4 lightSpaceMatrix;
uniform sampler2D dir_shadowMap;
uniform samplerCube point_shadowMap;
uniform samplerCube diffuse_convolution;
uniform samplerCube reflect_mipmap;
uniform sampler2D reflect_lut;
uniform float far_plane;
uniform float totalTime;

struct Material {
    vec4 diffuse;
    bool diffuse_texture_use;
    sampler2D diffuse_texture;

    vec3 specular;
    bool specular_texture_use;
    sampler2D specular_texture;

    float metallic;
    bool metallic_texture_use;
    sampler2D metallic_texture;

    float roughness;
    bool roughness_texture_use;
    sampler2D roughness_texture;

    bool normal_texture_use;
    sampler2D normal_texture;

    float ambient;
    bool ambient_texture_use;
    sampler2D ambient_texture;
}; 
uniform Material material;

struct DirectionLight {
    vec3 dir;
    vec3 color;
};

struct PointLight{
    vec3 pos;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
};

uniform DirectionLight dl[6];
uniform PointLight pl[6];
const float PI = 3.14159265359;

vec3 sampleOffsetDirections[20] = vec3[]
(
   vec3( 1,  1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1,  1,  1), 
   vec3( 1,  1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1,  1, -1),
   vec3( 1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1,  1,  0),
   vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
   vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0,  1, -1)
);

float D_GGX_TR(vec3 N, vec3 H, float a)
{
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom    = a2;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
float GeometrySchlickGGX(float NdotV, float k)
{
    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float k)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, k);
    float ggx2 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}   

float cal_dir_shadow()
{
    vec4 fragPosLightSpace = lightSpaceMatrix * vec4(f_FragPos,1.0);
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float currentDepth = projCoords.z;
    float bias = 0.005;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(dir_shadowMap, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(dir_shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    return shadow;
}

float cal_point_shadow()
{
    vec3 fragToLight = f_FragPos - pl[0].pos; 
    float currentDepth = length(fragToLight);
    float bias = 11.0; 
    float shadow = 0.0;
    int samples = 20;
    float viewDistance = length(cameraPos - f_FragPos);
    float diskRadius = (1.0 + (viewDistance / far_plane)) / 2.0;
    for(int i = 0; i < samples; ++i)
    {
        float closestDepth = texture(point_shadowMap, fragToLight + sampleOffsetDirections[i] * diskRadius).r;
        closestDepth *= far_plane;   // Undo mapping [0;1]
        if(currentDepth - bias > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);

    return shadow;
}

vec3 BlendAngleCorrectedNormals(vec3 n1, vec3 n2)
{
    return normalize(vec3(n1.xy + n2.xy, n1.z));
}

vec3 GetBaseNormal()
{
    vec2 texcoord = f_texcoord * vec2(2, 2) + totalTime * vec2(-0.0006,-0.0004);
    vec3 normal = vec3(texture(material.normal_texture, texcoord));
    normal = normalize(2 * normal - 1);
    return normal;
}

vec3 GetAdditionNormal(float level)
{
    float time = sin(f_FragPos.x / 150 + 0.01 * totalTime);
    vec2 texcoord = f_texcoord * vec2(1, 1) * level + time * vec2(-0.0006,-0.0004) * level;
    vec3 normal = vec3(texture(material.normal_texture, texcoord));
    normal = normalize(2 * normal - 1);
    return normal;
}

void main()
{
    FragColor = vec4(0,0,0,1);
    BrightColor = vec4(0,0,0,1);

    vec4 diffuse_;
    if(material.diffuse_texture_use == true) {
        diffuse_ = texture(material.diffuse_texture, f_texcoord);
        diffuse_.rgb = pow(diffuse_.rgb,vec3(2.2));
    }
    else diffuse_ = material.diffuse;

    vec3 specular_;
    if(material.specular_texture_use == true) specular_ = vec3(texture(material.specular_texture, f_texcoord));
    else specular_ = material.specular;

    float metallic_;
    if(material.metallic_texture_use == true) metallic_ = texture(material.metallic_texture, f_texcoord).r;
    else metallic_ = material.metallic;

    float roughness_;
    if(material.roughness_texture_use == true) roughness_ = texture(material.roughness_texture, f_texcoord).r;
    else roughness_ = material.roughness;
    if(roughness_<0.01)roughness_ = 0.01;

    vec3 N = normalize(f_normal);
    if(material.normal_texture_use == true) {
        N = GetBaseNormal();
        vec3 N_1 = GetAdditionNormal(3);
        //vec3 N_2 = GetAdditionNormal(10);
        N = BlendAngleCorrectedNormals(N,N_1);
        //N = BlendAngleCorrectedNormals(N,N_2);
        N = normalize(f_TBN * N);
    }

    float ambient_;
    if(material.ambient_texture_use == true) ambient_ = texture(material.ambient_texture, f_texcoord).r;
    else ambient_ = material.ambient;

    vec3 F0 = mix(vec3(0.04), diffuse_.rgb, metallic_);
    vec3 V = normalize(cameraPos - f_FragPos);
    for(int i = 0; i < 1; i++){
        if(dl[i].color == vec3(0,0,0)) continue;
        vec3 L = normalize(-dl[i].dir);
        vec3 H = normalize(L + V);
        // cook-torrance brdf
        float NDF = D_GGX_TR(N, H, roughness_);        
        float G = GeometrySmith(N, V, L, roughness_);      
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 kD = vec3(1.0) - F;
        kD *= 1.0 - metallic_;

        vec3 nominator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        vec3 specular = specular_ * nominator / denominator;

        // add to outgoing radiance Lo
        float shadow = cal_dir_shadow();
        float NdotL = max(dot(N, L), 0.0);                
        FragColor.rgb += (1 - shadow) * (kD * diffuse_.rgb / PI + specular) * dl[i].color * NdotL; 
    }
    for(int i = 0; i < 1; i++){
        if(pl[i].color == vec3(0,0,0)) continue;
        vec3 L = normalize(pl[i].pos - f_FragPos);
        vec3 H = normalize(L + V);
        float distance = length(pl[i].pos - f_FragPos);
        vec3 pl_color = pl[i].color / (pl[i].constant + pl[i].linear * distance + pl[i].quadratic * distance * distance);

        float NDF = D_GGX_TR(N, H, roughness_);        
        float G = GeometrySmith(N, V, L, roughness_);      
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 kD = vec3(1.0) - F;
        kD *= 1.0 - metallic_;

        vec3 nominator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        vec3 specular = specular_ * nominator / denominator;

        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);   

        float shadow = cal_point_shadow();
        FragColor.rgb += ((1 - shadow) * (kD * diffuse_.rgb / PI + specular) * pl_color * NdotL); 
    }

    vec3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness_); 
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic_;
    //kD = 0;
    vec3 environment_diffuse = ambient_ * kD * texture(diffuse_convolution, N).rgb * diffuse_.rgb;
    vec3 R = reflect(-V, N);
    vec3 prefilteredColor = textureLod(reflect_mipmap, R,  roughness_ * 4).rgb;
    vec2 envBRDF  = texture(reflect_lut, vec2(max(dot(N, V), 0.0), roughness_)).rg;
    vec3 environment_reflect = prefilteredColor * (F0 * envBRDF.x + envBRDF.y);
    FragColor.rgb += ambient_ * (environment_diffuse + specular_ * environment_reflect);
    FragColor.rgb *= fresnelSchlick(dot(V,normalize(f_normal)),F0);
    FragColor.a = fresnelSchlick(dot(V,normalize(f_normal)),F0).r + 0.3;

    float brightness = dot(FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    if(brightness > 1.0)
        BrightColor = vec4(FragColor.rgb, 1.0);
}