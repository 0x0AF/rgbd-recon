#version 420

layout(std140) uniform Matrices
{
    mat4 projMatrix;
    mat4 viewMatrix;
    mat4 modelMatrix;
};

in vec3 position;
in vec3 normal;
in vec2 texCoord;

uniform int layer;
uniform sampler3D[5] cv_xyz;
uniform sampler3D[5] cv_xyz_inv;
uniform sampler3D[5] cv_uv;

out vec4 vertexPos;
out vec2 TexCoord;
out vec3 Normal;

out float proj_depth;

void main()
{
    Normal = normalize((vec4(normal, 1.)).rgb) * 0.5 + vec3(0.5);
    TexCoord = vec2(texCoord);

    vec3 coords = (modelMatrix * vec4(position, 1.0)).rgb;
    coords.x = (coords.x + 0.1999f) / 1.4f;
    coords.y = (coords.y - 0.f) / 1.4f;
    coords.z = (coords.z + 1.f) / 1.4f;
    coords = clamp(coords, vec3(0.), vec3(1.));

    vec4 projection = texture(cv_xyz_inv[layer], coords);

    //float n = 0.1f;
    //float f = 3.f;
    //float real_depth = -(projection.z * (n - f) + n);
    //float non_linear_depth = (f + n - 2. * n * f / real_depth) / (f - n);
    //non_linear_depth = (non_linear_depth + 1.) / 2.;

    proj_depth = projection.z;

    gl_Position = vec4((projection.xy - 0.5) * 2., 0., 1.);

    // vec4 back_projection = texture(cv_xyz[layer], projection.xyz);

    // gl_Position = projMatrix * viewMatrix * vec4(back_projection.xyz, 1.0);
}