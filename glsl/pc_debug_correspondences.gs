#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable
#extension GL_EXT_texture_array : enable

layout(points) in;
layout(line_strip, max_vertices = 2) out;

struct Correspondence
{
    vec3 previous;
    uint layer;
    vec3 current;
    uint cell_id;
    uvec2 previous_proj;
    uvec2 current_proj;
};

layout(std430, binding = 10) restrict buffer CorrespondenceBuffer { Correspondence correspondences[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];

out float pass_Length;
out vec3 pass_Color;

#include </inc_bbox_test.glsl>

void main()
{
    Correspondence correspondence = correspondences[uint(geo_Position[0].x * 1000000.f)];

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(correspondence.previous, 1.0f);
    pass_Length = 10.f * length(correspondence.current - correspondence.previous);
    pass_Color = vec3(0.f, 0.0f, 1.f);
    EmitVertex();

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(correspondence.current, 1.0f);
    pass_Length = 10.f * length(correspondence.current - correspondence.previous);
    pass_Color = vec3(0.f, 1.f, 0.f);
    EmitVertex();

    EndPrimitive();
}