#version 420

layout(std140) uniform Material
{
    vec4 diffuse;
    vec4 ambient;
    vec4 specular;
    vec4 emissive;
    float shininess;
    int texCount;
};

uniform sampler2D texUnit;
uniform sampler2D clouds;
uniform sampler2D texture_poi;

in vec3 Normal;
in vec2 TexCoord;

in float proj_depth;

out vec4 out_color;

out float gl_FragDepth;

void main()
{
    vec4 color;
    vec4 amb;
    float intensity;
    vec3 lightDir;
    vec3 n;

    lightDir = normalize(vec3(1.0, 1.0, 1.0));
    n = /*normalize(*/ Normal /*)*/;
    intensity = max(dot(lightDir, n), 0.0);

    /*if(texCount == 0)
    {
        color = diffuse;
        amb = ambient;
    }
    else
    {
        color = texture(texUnit, TexCoord);
        amb = color * 0.33;
    }*/

    /*color = vec4(0.4f, 0.4f, 0.4f, 1.f);
    amb = vec4(0.1f, 0.1f, 0.1f, 1.f);

    out_color = (color * intensity) + amb;*/

    vec3 intensity_rgb = texture(texture_poi, TexCoord).rgb;// (n * 0.5 + 0.5) * (0.5 + texture(clouds_earth, TexCoord).r * 0.5 + texture(clouds, TexCoord).r * 0.);
    out_color = vec4(intensity_rgb, 1.);

    // out_color = vec4(1.0f, 0.f, 0.f, 1.f);

    gl_FragDepth = proj_depth;
}