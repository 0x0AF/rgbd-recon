#version 420

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_normal;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 world_to_vol;
uniform mat4 vol_to_world;

out vec3 pass_Position;
out vec3 pass_Normal;

void main()
{
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * world_to_vol * in_position;
    pass_Position = (inverse(vol_to_world) * world_to_vol * in_position).xyz;
    pass_Normal = normalize(in_normal.xyz);
}