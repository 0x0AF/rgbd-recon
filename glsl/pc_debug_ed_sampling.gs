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
    float data_term;
    float hull_term;
    float correspondence_term;
    float regularization_term;
    int pad[1];
};

layout(std430, binding = 8) restrict buffer EDNodeBuffer { EDNode ed_nodes[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

uniform int mode;

in vec3 geo_Position[];

out vec3 pass_Position;
out float pass_Error;
out float pass_Range;

#include </inc_bbox_test.glsl>

void main()
{
    EDNode ed_node = ed_nodes[uint(geo_Position[0].x * 1000000.f)];

    pass_Position = ed_node.position;

    if(mode == 0)
    {
        pass_Error = ed_node.misalignment_error;
        pass_Range = 0.03f;
    }

    if(mode == 1)
    {
        pass_Error = ed_node.data_term;
        pass_Range = 0.01f;
    }

    if(mode == 2)
    {
        pass_Error = ed_node.hull_term;
        pass_Range = 1.f;
    }

    if(mode == 3)
    {
        pass_Error = ed_node.correspondence_term;
        pass_Range = 0.1f;
    }

    if(mode == 4)
    {
        pass_Error = ed_node.regularization_term;
        pass_Range = 2.f;
    }

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(pass_Position, 1.0);
    gl_PointSize = 3.5f;

    EmitVertex();
    EndPrimitive();
}