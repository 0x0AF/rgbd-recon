#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable
#extension GL_EXT_texture_array : enable

layout(points) in;
layout(line_strip, max_vertices = 2) out;

uniform sampler2DArray kinect_depths;
uniform sampler3D[4] cv_xyz;

struct Correspondence
{
    vec2 previous;
    vec2 current;
    uint layer;
    uint pad;
    float depth_prev;
    float depth_curr;
    /*vec3 previous;
    uint pad1;
    vec3 current;
    uint pad2;*/
};

layout(std430, binding = 10) restrict buffer CorrespondenceBuffer { Correspondence correspondences[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];
out float pass_Length;
out vec3 pass_Color;

vec3 get_volume_coordinates(vec2 in_pos, float depth, uint layer) { return texture(cv_xyz[layer], vec3(in_pos, depth)).rgb; }

#include </inc_bbox_test.glsl>

void main()
{
    Correspondence correspondence = correspondences[uint(geo_Position[0].x * 1000000.f)];

    vec3 coords_curr = get_volume_coordinates(correspondence.current, correspondence.depth_curr, correspondence.layer);
    vec3 coords_prev = get_volume_coordinates(correspondence.previous, correspondence.depth_prev, correspondence.layer);

    if(!in_bbox(coords_curr) || !in_bbox(coords_prev) || length(coords_curr - coords_prev) > 0.2f)
    {
        return;
    }

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix /** vol_to_world*/ * vec4(coords_prev, 1.0f);
    pass_Length = 10.f * length(correspondence.current - correspondence.previous);
    pass_Color = vec3(0.f, 1.0f, 0.f);
    EmitVertex();

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix /** vol_to_world*/ * vec4(coords_curr, 1.0f);
    pass_Length = 10.f * length(correspondence.current - correspondence.previous);
    pass_Color = vec3(0.f, 0.0f, 1.f);
    EmitVertex();

    EndPrimitive();
}