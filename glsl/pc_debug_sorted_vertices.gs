#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

layout(points) in;
layout(points, max_vertices = 1) out;

struct Vertex
{
    vec3 position;
    uint brick_id;
    vec3 normal;
    uint ed_cell_id;
};

layout(std430, binding = 9) restrict buffer SortedVertexBuffer { Vertex vertices[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];

out vec3 pass_Position;
out float pass_BrickIdColor;
out vec3 pass_Normal;
out float pass_EDCellIdColor;

#include </inc_bbox_test.glsl>

void main()
{
    Vertex vertex = vertices[uint(geo_Position[0].x * 100000.f)];

    pass_Position = vertex.position;
    pass_BrickIdColor = (vertex.brick_id % 256) / 256.0f;
    pass_Normal = vertex.normal;
    pass_EDCellIdColor = (vertex.ed_cell_id % 27) / 27.0f;

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(pass_Position + bbox_min, 1.0);
    gl_PointSize = 2.0f;

    EmitVertex();
    EndPrimitive();
}