#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

layout(points) in;
layout(points, max_vertices = 1) out;

struct EDNode
{
    vec3 position;
    uint brick_id;
    vec3 translation;
    uint ed_cell_id;
    vec4 affine;

    uint vx_offset;
    uint vx_length;
    float misalignment_error;
    uint pad;
};

layout(std430, binding = 8) restrict buffer EDNodeBuffer { EDNode ed_nodes[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];

out vec3 pass_Position;
out vec3 pass_Error;

#include </inc_bbox_test.glsl>

void main()
{
    EDNode node = ed_nodes[uint(geo_Position[0].x * 1000000.f)];

    pass_Position = node.position;
    pass_Error = vec3(node.misalignment_error, 0.f, 0.f);

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(pass_Position, 1.0);
    gl_PointSize = 3.5f;

    EmitVertex();
    EndPrimitive();
}