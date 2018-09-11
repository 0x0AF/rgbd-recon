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

in vec3 Normal;
in vec2 TexCoord;
out vec4 out_color;

void main()
{
    vec4 color;
    vec4 amb;
    float intensity;
    vec3 lightDir;
    vec3 n;

    lightDir = normalize(vec3(1.0, 1.0, 1.0));
    n = normalize(Normal);
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
    }

    if(color.r < 0.001f){*/
    color = vec4(0.75f, 0.75f, 0.75f, 1.f);
    amb = vec4(0.2f, 0.2f, 0.2f, 1.f);
    /*}*/

    out_color = (color * intensity) + amb;

    // out_color = vec4(1.0f, 0.f, 0.f, 1.f);
}